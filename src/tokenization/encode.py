from __future__ import annotations

from dataclasses import dataclass, field

from src.preprocessing.canonical.events import normalize_text
from src.preprocessing.read import ClientEvent, ClientHistory

from .finalvocab import FrozenArtifacts
from .numeric import BucketsError
from .scan import value_text
from .specials import EVT, UNK, USR


# ============================================================
# КОДИРОВАНИЕ
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
# Пары существуют только у того, что есть. Отсутствующее поле в
# последовательность не попадает вовсе, и пустой после
# нормализации текст это то же отсутствие: служебного токена
# «значения нет» больше нет. Неизвестное, но корректное значение
# кодируется [UNK].
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


def _text_value_ids(artifacts: FrozenArtifacts, key: str, value: object,
                    limit: int) -> list[int] | None:
    """
    Куски текста в общем пространстве ID.

    None означает, что значения нет: пустой после нормализации
    текст поля не создаёт.
    """

    if not isinstance(value, str):
        raise EncodeError(f"ключ {key} объявлен текстом, а значение пришло как {type(value).__name__}")

    normalized = normalize_text(value)

    if normalized is None:
        return None

    if not artifacts.bpe.enabled:
        return [artifacts.special(UNK)]

    pieces = artifacts.bpe.pieces(normalized)

    if len(pieces) > limit:
        raise EncodeError(
            f"ключ {key}: значение разбилось на {len(pieces)} кусков при пределе {limit}. "
            "Текст не обрезается: поднимите предел осознанно"
        )

    return [artifacts.piece_id(piece) for piece in pieces]


def _value_ids(artifacts: FrozenArtifacts, key: str, value: object,
               limit: int) -> list[int] | None:
    """
    Значение одного ключа в общем пространстве ID.

    None означает, что пары у этого ключа не будет.
    """

    kind = artifacts.kind(key)

    if kind == "text":
        return _text_value_ids(artifacts, key, value, limit)

    if kind == "categorical":

        found = artifacts.categorical_id(key, value_text(value))

        return [artifacts.special(UNK) if found is None else found]

    try:
        found = artifacts.bucket_id(key, value)
    except (BucketsError, TypeError, ValueError) as error:
        raise EncodeError(f"ключ {key}: {error}") from error

    return [artifacts.special(UNK) if found is None else found]


def encode_values(
    artifacts: FrozenArtifacts,
    values: dict[str, object],
    lead: str,
    limit: int,
    links: frozenset[str] = frozenset(),
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

    emit: list[tuple[int, str, object]] = []

    for key, value in values.items():

        if key in links:
            # Ссылка кодом не становится: она уезжает в
            # метаданные связи рядом с записью.
            continue

        if value is None:
            # Поля нет — и пары нет.
            continue

        key_id = artifacts.key_id(key)

        if key_id is None:
            record.unknown_keys.append(key)
            continue

        emit.append((key_id, key, value))

    for key_id, key, value in sorted(emit, key=lambda item: item[0]):

        ids = _value_ids(artifacts, key, value, limit)

        if ids is None:
            continue

        record.add(key, key_id, ids)

    record.unknown_keys.sort()
    record.check()

    return record


def encode_event(artifacts: FrozenArtifacts, event: ClientEvent, limit: int,
                 links: frozenset[str] = frozenset()) -> EncodedRecord:
    """
    Одно событие: ведущий [EVT] и пары его значений.
    """

    return encode_values(artifacts, event.model_values(), EVT, limit, links)


def encode_profile(artifacts: FrozenArtifacts, history: ClientHistory, limit: int,
                   links: frozenset[str] = frozenset()) -> EncodedRecord:
    """
    Представление профиля: ведущий [USR] и пары его значений.

    У клиента без анкеты остаётся только маркер: банк о нём ещё
    ничего не знает, и выдумывать пустые поля незачем.
    """

    return encode_values(artifacts, history.profile or {}, USR, limit, links)


def references(event: ClientEvent, links: frozenset[str]) -> dict[str, str]:
    """
    Локальные ссылки события: связь, а не значение.
    """

    return {key: value for key, value in sorted(event.values.items()) if key in links}


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
                "decoded": [artifacts.describe(record.value_ids[0])],
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
            "decoded": [artifacts.describe(value_id) for value_id in ids],
        }

        # У текста куски читаются по отдельности плохо: рядом
        # кладётся собранный обратно текст.
        if artifacts.kind(key) == "text" and artifacts.bpe.enabled:
            item["decoded_text"] = artifacts.decode_text(ids)

        out.append(item)

    return out


__all__ = [
    "EncodeError",
    "EncodedRecord",
    "decode_record",
    "encode_event",
    "encode_profile",
    "encode_values",
    "references",
]
