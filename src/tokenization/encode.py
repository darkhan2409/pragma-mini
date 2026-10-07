from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from operator import itemgetter

from src.preprocessing.canonical.events import normalize_text
from src.preprocessing.keys import PROFILE_LIFELONG_KEY
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
# Границы значений задаёт positions, и другого их описания в
# формате нет: ноль начинает новое значение, дальше идут 1, 2, …
# — куски того же значения. Разбор однозначен потому, что
# значений нулевой длины не бывает: отсутствующее поле пары не
# создаёт вовсе, поэтому нулей ровно столько, сколько значений.
#
# Два соседних значения с одним ключом остаются двумя: каждое
# начинает свой ноль, и на key_id правило не смотрит.
#
# Порядок полей смысла не несёт: событие это набор пар, а его
# границы задаются отдельно. Пары записи идут по возрастанию
# key_id; у анкеты за ними следуют вехи — по времени.
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
# Идентификаторов сущностей в значениях нет вовсе: их не
# пропускает препроцессинг, и связи между договорами, счетами
# и картами модели не передаются.
# ============================================================


# Сколько разных строк помнит разбор текста (_text_value_ids).
TEXT_CACHE_SIZE = 1 << 16

_MISSING = object()

_first = itemgetter(0)


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
    входит.

    Разница не косметическая. Маркер стоит на нулевой позиции,
    как и начало любого значения, и по одному positions его не
    отличить. Поэтому разбор всегда начинается со следующего
    токена: иначе потребитель, ищущий значения, однажды принял
    бы маркер за обычное поле и спрятал бы его под маской.
    """

    lead: str = ""
    key_ids: list[int] = field(default_factory=list)
    value_ids: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    unknown_keys: list[str] = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        return len(self.key_ids)

    @property
    def n_values(self) -> int:
        """
        Сколько содержательных значений в записи. Маркер сюда не
        входит: его ноль в разбор не попадает.
        """

        return self.positions[self.content_start:].count(0)

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

    def add(self, key_id: int, value_ids: list[int]) -> None:
        """
        Пара ключ/значение: ноль открывает значение, дальше идут
        номера его кусков.
        """

        # Число и категория — один кусок: так чаще всего.
        if len(value_ids) == 1:
            self.key_ids.append(key_id)
            self.value_ids.append(value_ids[0])
            self.positions.append(0)
            return

        self.key_ids.extend([key_id] * len(value_ids))
        self.value_ids.extend(value_ids)
        self.positions.extend(range(len(value_ids)))

    def check(self) -> None:
        """
        Длины согласованы, маркер на своём месте, а positions
        разбираются на значения без разрывов.
        """

        if not (len(self.key_ids) == len(self.value_ids) == len(self.positions)):
            raise EncodeError("массивы записи разной длины")

        if self.lead:

            if not self.key_ids:
                raise EncodeError("запись объявила маркер, но массивы пусты")

            if self.key_ids[0] != self.value_ids[0]:
                raise EncodeError("маркер обязан занимать оба слота одним кодом")

            if self.positions[0] != 0:
                raise EncodeError("маркер обязан стоять на позиции 0")

        # Содержимое обязано начаться с нуля, а куски значения —
        # идти подряд: позиция либо открывает значение, либо
        # продолжает предыдущую ровно на единицу. Первый токен с
        # positions = 3 не пройдёт ни одну из двух веток.
        content = self.positions[self.content_start:]

        # Одни нули — каждое значение из одного куска: обе ветки
        # проходят на каждой позиции, и обходить их незачем.
        if content.count(0) == len(content):
            return

        expected = 0

        for index, position in enumerate(content, self.content_start):

            if position != 0 and position != expected:
                raise EncodeError(
                    f"в позиции {index} стоит {position}, а куски значения идут "
                    f"подряд от нуля (ожидалось {expected})"
                )

            expected = position + 1


def _text_value_ids(artifacts: FrozenArtifacts, key: str, value: object,
                    limit: int) -> list[int] | None:
    """
    Куски текста в общем пространстве ID.

    None означает, что значения нет: пустой после нормализации
    текст поля не создаёт.
    """

    if not isinstance(value, str):
        raise EncodeError(f"ключ {key} объявлен текстом, а значение пришло как {type(value).__name__}")

    if not artifacts.bpe.enabled:
        return None if normalize_text(value) is None else [artifacts.special(UNK)]

    # Нормализация и разбиение — чистые функции строки, а названия
    # мерчантов и экранов повторяются: каждая разная строка
    # разбирается один раз. Предел проверяется на каждом значении.
    texts = artifacts.texts

    ids = texts.get(value, _MISSING)

    if ids is _MISSING:

        normalized = normalize_text(value)

        if normalized is None:
            ids = None
        else:
            pieces = artifacts.bpe.pieces(normalized)
            # Предел — до перевода кусков в номера, как без запаса.
            _check_pieces(key, len(pieces), limit)
            ids = tuple(artifacts.piece_id(piece) for piece in pieces)

        # Память ограничена: переполненный запас начинается заново,
        # ответ от этого не меняется.
        if len(texts) >= TEXT_CACHE_SIZE:
            texts.clear()

        texts[value] = ids

    if ids is None:
        return None

    _check_pieces(key, len(ids), limit)

    return list(ids)


def _check_pieces(key: str, count: int, limit: int) -> None:

    if count > limit:
        raise EncodeError(
            f"ключ {key}: значение разбилось на {count} кусков при пределе {limit}. "
            "Текст не обрезается: поднимите предел осознанно"
        )


def _value_ids(artifacts: FrozenArtifacts, key: str, value: object,
               limit: int, record: dict | None = None, kind: str | None = None) -> list[int] | None:
    """
    Значение одного ключа в общем пространстве ID. record — все
    значения записи: шкала числа может зависеть от соседнего ключа.
    kind — уже известный artifacts.kind(key).

    None означает, что пары у этого ключа не будет.
    """

    if kind is None:
        kind = artifacts.kind(key)

    if kind == "text":
        return _text_value_ids(artifacts, key, value, limit)

    if kind == "categorical":

        # Запись строки это она сама: value_text(str) == str.
        found = artifacts.categorical_id(key, value if value.__class__ is str else value_text(value))

        return [artifacts.special(UNK) if found is None else found]

    try:
        found = artifacts.bucket_id(key, value, record)
    except (BucketsError, TypeError, ValueError) as error:
        raise EncodeError(f"ключ {key}: {error}") from error

    return [artifacts.special(UNK) if found is None else found]


def encode_values(
    artifacts: FrozenArtifacts,
    values: dict[str, object],
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

    encoders = artifacts.encoders

    emit: list[tuple[int, str, object, str]] = []

    for key, value in values.items():

        if value is None:
            # Поля нет — и пары нет.
            continue

        encoder = encoders.get(key)

        if encoder is None:
            record.unknown_keys.append(key)
            continue

        emit.append((encoder[0], key, value, encoder[1]))

    # key_id у разных ключей разные: порядок полный.
    emit.sort(key=_first)

    for key_id, key, value, kind in emit:

        ids = _value_ids(artifacts, key, value, limit, values, kind)

        if ids is None:
            continue

        record.add(key_id, ids)

    record.unknown_keys.sort()
    record.check()

    return record


def encode_event(artifacts: FrozenArtifacts, event: ClientEvent, limit: int) -> EncodedRecord:
    """
    Одно событие: ведущий [EVT] и пары его значений.
    """

    # Значения читаются, но не меняются (encode_values только
    # смотрит в них), поэтому копия model_values не нужна.
    return encode_values(artifacts, event.values, EVT, limit)


def encode_profile(
    artifacts: FrozenArtifacts, history: ClientHistory, limit: int
) -> tuple[EncodedRecord, list[datetime | None]]:
    """
    Представление анкеты: ведущий [USR], пары Attributes, затем
    по паре на каждую веху Lifelong, и время каждого токена.

    Время есть только у вех: у [USR] и Attributes оно None — это
    состояние на cutoff, а не событие. У всех кусков одного
    значения время одно.

    Вехи идут под одним ключом, значение — тип вехи. Ключ
    повторяется, и каждое значение начинает свой ноль в positions.

    У клиента без анкеты остаётся только маркер: банк о нём ещё
    ничего не знает, и выдумывать пустые поля незачем.
    """

    record = encode_values(artifacts, history.profile or {}, USR, limit)

    times: list[datetime | None] = [None] * record.n_tokens

    if history.lifelong:

        key = PROFILE_LIFELONG_KEY.key

        key_id = artifacts.key_id(key)

        # Словарь без ключа вех собран прежним кодом: молча
        # потерять вехи хуже, чем остановиться.
        if key_id is None:
            raise EncodeError(f"в словаре нет ключа {key}: словарь собран прежним кодом")

        for kind, moment in history.lifelong:

            ids = _value_ids(artifacts, key, kind, limit)

            if ids is None:
                continue

            record.add(key_id, ids)
            times.extend([moment] * len(ids))

    record.check()

    return record, times


__all__ = [
    "EncodeError",
    "EncodedRecord",
    "encode_event",
    "encode_profile",
    "encode_values",
]
