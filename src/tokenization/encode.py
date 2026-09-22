from __future__ import annotations

from dataclasses import dataclass, field

from src.preprocessing.canonical.events import normalize_text
from src.preprocessing.read import ClientEvent, ClientHistory
from src.preprocessing.keys import CATEGORICAL, NUMERIC, TEXT

from .layout import FrozenArtifacts
from .numeric import FOUND_BUCKET, FOUND_INVALID
from .scan import value_text, value_type
from .specials import EMPTY, EVT, INVALID, MISSING, UNK, USR


# ============================================================
# ИДЕЯ
# ============================================================
#
# Кодирование применяет готовый словарь и ничего не обучает.
#
# Содержательная позиция это связанная пара ключ/значение.
# Число и категория занимают одну позицию, текст — столько,
# сколько кусков дал BPE, и все они делят один key_id.
#
# positions это номер куска ВНУТРИ значения, а не порядок полей
# события. Порядок полей смысла не несёт вовсе: событие это
# набор пар, а его границы задаются отдельно.
#
# Ничего не обрезается. Значение, которое не помещается в
# объявленный предел кусков, это явная ошибка, а не молчаливо
# укороченный текст.
#
# Ссылки на сущности в embedding не входят: они едут рядом
# служебной колонкой для будущего Masker и датасета.
# ============================================================


class EncodeError(ValueError):
    """
    Значение закодировать нельзя.
    """


@dataclass
class EncodedRecord:
    """
    Одна запись: событие или представление профиля.

    Позиция 0 это ведущий маркер. Он занимает место в массивах
    модели, потому что модель обязана его видеть, но
    СОДЕРЖАТЕЛЬНЫМ значением не является: в n_values он не
    входит и собственного span не получает.

    Разница не косметическая. Маркер не имеет ключа, его нечего
    предсказывать, и маскировать его нельзя. Если бы он лежал
    среди значений, любой потребитель, который ходит по spans,
    считал бы его обычным полем и однажды спрятал бы под маской.
    """

    lead: str = ""
    key_ids: list[int] = field(default_factory=list)
    value_ids: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    value_starts: list[int] = field(default_factory=list)
    value_lengths: list[int] = field(default_factory=list)
    value_keys: list[str] = field(default_factory=list)
    unknown_keys: list[str] = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        return len(self.key_ids)

    @property
    def n_values(self) -> int:
        """
        Сколько содержательных значений в записи. Маркер сюда не
        входит.
        """

        return len(self.value_starts)

    @property
    def content_start(self) -> int:
        """
        Первая позиция, с которой начинается содержимое.
        """

        return 1 if self.lead else 0

    def set_lead(self, name: str, token_id: int) -> None:
        """
        Ведущий маркер: одинаковый код в обоих слотах, позиция 0.
        """

        if self.key_ids:
            raise EncodeError("маркер ставится первым, до любого значения")

        self.lead = name
        self.key_ids.append(token_id)
        self.value_ids.append(token_id)
        self.positions.append(0)

    def add(self, key: str, key_id: int, value_ids: list[int]) -> None:

        self.value_starts.append(len(self.key_ids))
        self.value_lengths.append(len(value_ids))
        self.value_keys.append(key)

        for position, value_id in enumerate(value_ids):
            self.key_ids.append(key_id)
            self.value_ids.append(value_id)
            self.positions.append(position)

    def check(self) -> None:
        """
        Длины согласованы, маркер на своём месте, а span'ы
        покрывают все позиции содержимого без дыр и нахлёстов.
        """

        if not (len(self.key_ids) == len(self.value_ids) == len(self.positions)):
            raise EncodeError("массивы записи разной длины")

        if not (len(self.value_starts) == len(self.value_lengths) == len(self.value_keys)):
            raise EncodeError("массивы span'ов разной длины")

        if self.lead:

            if not self.key_ids:
                raise EncodeError("запись объявила маркер, но массивы пусты")

            if self.key_ids[0] != self.value_ids[0]:
                raise EncodeError("маркер обязан занимать оба слота одним кодом")

            if self.positions[0] != 0:
                raise EncodeError("маркер обязан стоять на позиции 0")

        covered = self.content_start

        for start, length in zip(self.value_starts, self.value_lengths):
            if start != covered:
                raise EncodeError(f"span значения начинается в {start}, а покрыто {covered}")
            covered += length

        if covered != len(self.key_ids):
            raise EncodeError(f"span'ы покрывают {covered} позиций из {len(self.key_ids)}")


def _text_value_ids(artifacts: FrozenArtifacts, key: str, value: object, limit: int) -> list[int]:
    """
    Куски текста в общем пространстве ID.
    """

    if not isinstance(value, str):
        raise EncodeError(f"ключ {key} объявлен текстом, а значение пришло как {type(value).__name__}")

    normalized = normalize_text(value)

    if normalized is None:
        # Пустой текст это не отсутствие значения: поле пришло.
        return [artifacts.special(EMPTY)]

    if not artifacts.bpe.enabled:
        return [artifacts.special(UNK)]

    pieces = artifacts.bpe.pieces(normalized)

    if len(pieces) > limit:
        raise EncodeError(
            f"ключ {key}: значение разбилось на {len(pieces)} кусков при пределе {limit}. "
            "Текст не обрезается: поднимите предел осознанно"
        )

    return [artifacts.bpe_offset + piece for piece in pieces]


def _value_ids(artifacts: FrozenArtifacts, key: str, value: object, limit: int) -> list[int]:
    """
    Значение одного ключа в общем пространстве ID.
    """

    kind = artifacts.key_info[key]["value_kind"]

    if kind == TEXT:
        return _text_value_ids(artifacts, key, value, limit)

    if kind == CATEGORICAL:

        found = artifacts.categorical_id(key, value_type(value), value_text(value))

        return [artifacts.special(UNK) if found is None else found]

    if kind == NUMERIC:

        encoder = artifacts.encoders[key]

        found, index = encoder.locate(value)

        if found == FOUND_BUCKET:
            return [artifacts.bucket_id(key, index)]

        return [artifacts.special(INVALID if found == FOUND_INVALID else UNK)]

    raise EncodeError(f"ключ {key}: неизвестный вид значения {kind!r}")


def encode_values(
    artifacts: FrozenArtifacts,
    values: dict[str, object],
    declared: tuple[str, ...],
    lead: str,
    limit: int,
) -> EncodedRecord:
    """
    Пары одной записи: ведущий маркер, затем значения по
    возрастанию key_id.
    """

    record = EncodedRecord()

    # Маркер занимает и слот ключа, и слот значения: так
    # потребитель узнаёт служебную позицию по их равенству.
    # Содержательным значением он при этом не становится.
    record.set_lead(lead, artifacts.special(lead))

    emit: list[tuple[int, str, object, bool]] = []

    for key, value in values.items():

        if key in artifacts.link_keys:
            # Ссылка кодом не становится: она уезжает в
            # метаданные связи рядом с записью.
            continue

        info = artifacts.key_info.get(key)

        if info is None:
            record.unknown_keys.append(key)
            continue

        emit.append((info["id"], key, value, False))

    present = set(values)

    for key in declared:

        if key in present:
            continue

        info = artifacts.key_info.get(key)

        if info is None:
            continue

        emit.append((info["id"], key, None, True))

    for key_id, key, value, absent in sorted(emit):

        if absent:
            record.add(key, key_id, [artifacts.special(MISSING)])
            continue

        record.add(key, key_id, _value_ids(artifacts, key, value, limit))

    record.unknown_keys.sort()
    record.check()

    return record


def encode_event(artifacts: FrozenArtifacts, event: ClientEvent, limit: int) -> EncodedRecord:
    """
    Одно событие: ведущий [EVT] и пары его значений.
    """

    values = event.model_values()

    event_type = values.get("event_type")

    declared = artifacts.declared_by_event_type.get(event_type, ()) if event_type else ()

    return encode_values(artifacts, values, declared, EVT, limit)


def encode_profile(artifacts: FrozenArtifacts, history: ClientHistory, limit: int) -> EncodedRecord:
    """
    Представление профиля на cutoff: ведущий [USR] и пары его
    значений.

    Пока версия профиля известна, объявленным считается весь
    набор ключей профиля: пустой доход у клиента с анкетой это
    факт о клиенте. Но если анкеты нет вовсе, записи с двумя
    десятками [MISSING] не появляется: «банк ещё ничего не знает
    о клиенте» и «банк знает, но поля пусты» это разные вещи, и
    состояние профиля названо метаданными.

    Решает именно СОСТОЯНИЕ, а не пустота словаря значений:
    известная версия, у которой все поля пусты, это анкета, и
    все её ключи обязаны получить [MISSING].
    """

    declared = artifacts.profile_keys if profile_known(history) else ()

    return encode_values(artifacts, history.profile or {}, declared, USR, limit)


def profile_known(history: ClientHistory) -> bool:
    """
    Есть ли у клиента анкета вообще.

    Пустой словарь значений ответом не является: у известной
    анкеты все поля могут оказаться незаполненными.
    """

    return history.has_profile


def references(artifacts: FrozenArtifacts, event: ClientEvent) -> dict[str, str]:
    """
    Локальные ссылки события: связь, а не значение.
    """

    return {
        key: value
        for key, value in sorted(event.values.items())
        if key in artifacts.link_keys
    }


def decode_record(artifacts: FrozenArtifacts, record: EncodedRecord) -> list[dict]:
    """
    Читаемая расшифровка записи: что стоит в каждой позиции.
    """

    out: list[dict] = []

    if record.lead:
        out.append(
            {
                "key": record.lead,
                "key_id": record.key_ids[0],
                "value_ids": [record.value_ids[0]],
                "positions": [record.positions[0]],
                "decoded": [artifacts.describe(record.value_ids[0])["label"]],
                "marker": True,
            }
        )

    for key, start, length in zip(record.value_keys, record.value_starts, record.value_lengths):

        ids = record.value_ids[start : start + length]

        item = {
            "key": key,
            "key_id": record.key_ids[start],
            "value_ids": ids,
            "positions": record.positions[start : start + length],
            "decoded": [artifacts.describe(value_id)["label"] for value_id in ids],
        }

        # У текста куски читаются по отдельности плохо: рядом
        # кладётся собранный обратно текст, и он обязан совпасть
        # с нормализованным исходным.
        if artifacts.bpe.enabled and all(value_id >= artifacts.bpe_offset for value_id in ids):
            item["decoded_text"] = artifacts.bpe.decode([value_id - artifacts.bpe_offset for value_id in ids])

        out.append(item)

    return out


__all__ = [
    "EncodeError",
    "EncodedRecord",
    "decode_record",
    "encode_event",
    "encode_profile",
    "encode_values",
    "profile_known",
    "references",
]
