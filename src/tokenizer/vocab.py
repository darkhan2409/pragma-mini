from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.generator.config import PROFILE_FIELDS
from src.preprocessing.artifacts import read_json, sha256_ints, write_json
from src.preprocessing.buckets import STATUS_NO_FIT_DATA, load_specs
from src.preprocessing.config import (
    KIND_BOOLEAN,
    KIND_NUMERIC,
    SCHEMA_VERSION,
    FieldSpec,
    feature_specs,
)
from src.preprocessing.cutoffs import CUTOFF_INDEX_SCHEMA
from src.preprocessing.fit import FitScope, fit_scope
from src.preprocessing.stats import value_key

from .config import (
    FIELD_VALUE_IDS_FILE,
    KEY_VOCAB_FILE,
    N_SPECIAL,
    SPECIAL_IDS,
    SPECIAL_NOTES,
    SPECIAL_TOKENS,
    SPECIAL_TOKENS_FILE,
    VALUE_VOCAB_FILE,
    IncompatibleArtifactsError,
)
from .semantics import (
    DEFAULT_KEY_MODE,
    DEFAULT_VALUE_MODE,
    check_mode,
    is_baseline,
    registry_digest,
    semantic_key_name,
    shares_values,
    validate_registry,
    value_class,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Словарь обучается ТОЛЬКО на train и ровно на том же наборе
# записей, на котором preprocessing считал свои статистики:
# fit_scope переиспользуется как есть. Поэтому повторение одной
# записи в нескольких cutoff-примерах не увеличивает её частоту:
# fit-набор это префикс ленты клиента, а не объединение историй.
#
# ДВА ПРОСТРАНСТВА, КОТОРЫЕ НЕЛЬЗЯ ПУТАТЬ
#
#   field_id   идентичность физического поля. Индекс в реестре
#              preprocessing, 0..n_fields-1, ОДИН И ТОТ ЖЕ во
#              всех режимах. По нему живут кандидаты, маски,
#              predictable, корзины, unigram и все отчёты.
#
#   token_id   идентификатор токена в словаре ЭТОГО режима.
#              Только embedding lookup. В semantic-режиме два
#              поля могут нести один key token, в shared —
#              одинаковое значение из разных полей один value
#              token.
#
# Раскладка token_id:
#
#   [0, 6)                      special
#   [6, 6 + n_key_tokens)       key tokens
#   [6 + n_key_tokens, size)    value tokens
#
# Порядок не зависит от частот: поля идут в порядке реестра,
# значения внутри поля по возрастанию самого значения, а новый
# token выдаётся на первое появление класса склейки.
#
# Локальный индекс кандидата это позиция в ТИПИЗИРОВАННОМ
# порядке значений своего поля. У numeric он равен номеру
# корзины — на этом держится совпадение с unigram_baselines.
# Поэтому в field_value_ids кандидаты лежат в порядке поля, а
# не по возрастанию token_id: в shared-режиме это разные вещи.
# ============================================================


KEY_FORMAT = "{namespace}__{field}"

ORDER_RULES = {
    "ids": "сначала special, затем key tokens, затем value tokens",
    "fields": "порядок реестра preprocessing: timeline.event_type, поля событий по EVENT_TYPES, затем profile",
    "key_tokens": "по первому появлению: физический ключ либо имя семантической группы",
    "values": {
        "numeric": "корзины 0..actual_bucket_count-1 из bucket_edges.json по возрастанию",
        "boolean": "false, затем true (среди встреченных на train)",
        "categorical": "по возрастанию типизированного значения: целые численно, строки по кодпоинтам",
    },
    "value_tokens": "token выдаётся на первое появление класса склейки при обходе полей в порядке реестра",
    "local_index": "позиция в типизированном порядке значений своего поля, не позиция token_id",
    "frequency": "count хранится рядом со значением, но на порядок ID не влияет",
    "repeated_keys": "повторяющиеся ключи внутри события сохраняют исходный порядок значений (стабильная сортировка)",
    "unknown_key": "пара с ключом вне реестра ставится после известных полей, в порядке ввода",
}

VALUE_RULES = {
    "missing": "null значение -> [MISSING] в value_ids; field_id и key token при этом настоящие",
    "unknown": "непустое значение вне frozen vocab -> [UNK]; словарь не расширяется",
    "numeric": "numeric берётся уже bucketized из preprocessing: значение это номер корзины",
    "no_bpe": "остальные значения кодируются целиком, без BPE",
    "metadata": "metadata (client_id, ts, seq, snapshot_month, payload) в словарь не входит",
    "special_position": "[EVT], [USR] и неизвестный ключ несут один и тот же special ID в key_ids и в value_ids",
    "membership": "принадлежность значения полю живёт в field_value_ids, а не в записи токена: один token может принадлежать нескольким полям",
}


ARROW_TYPES: dict[str, pa.DataType] = {
    "int64": pa.int64(),
    "int32": pa.int32(),
    "int16": pa.int16(),
    "double": pa.float64(),
    "float": pa.float32(),
    "string": pa.string(),
    "bool": pa.bool_(),
}


# ============================================================
# ПРОСТРАНСТВО ПОЛЕЙ
# ============================================================
#
# Оно не зависит ни от режима, ни от данных: это реестр
# preprocessing. Специальная позиция ([EVT], [USR], неизвестный
# ключ) поля не имеет и несёт сентинел NO_FIELD.
#
# Сентинел это n_fields, а не -1: массивы по полю имеют длину
# n_fields + 1 с ложью в последней строке, и отрицательный
# индекс никуда не уезжает.
# ============================================================


def key_specs() -> list[FieldSpec]:
    """
    Поля словаря: все содержательные поля реестра, metadata нет.
    """

    return list(feature_specs())


N_FIELDS: int = len(key_specs())

NO_FIELD: int = N_FIELDS

FIELD_ID_RULE = (
    "field_id это индекс поля в реестре preprocessing, одинаковый во всех режимах; "
    f"NO_FIELD = {NO_FIELD} у специальных позиций"
)


# ============================================================
# ЗАПИСИ СЛОВАРЯ
# ============================================================
#
# Три вида записей, и путать их нельзя:
#
#   FieldEntry  одна на ФИЗИЧЕСКОЕ ПОЛЕ (их всегда n_fields)
#   KeyToken    одна на key token (их 57 или 47)
#   ValueEntry  одна на value token
#
# Принадлежность значения полю это отношение «многие ко многим»,
# поэтому у ValueEntry нет field_id. Порядок кандидатов внутри
# поля живёт в FieldEntry.candidates и в field_value_ids.json:
# множество порядка не несёт, а локальный индекс от него зависит.
# ============================================================


@dataclass(frozen=True)
class FieldEntry:
    field_id: int
    key: str
    namespace: str
    field: str
    kind: str
    predictable: bool
    arrow_type: str
    key_token_id: int
    candidates: tuple[int, ...]

    @property
    def n_values(self) -> int:
        return len(self.candidates)

    @property
    def is_contiguous(self) -> bool:
        """
        Кандидаты поля идут подряд по token_id.

        В baseline это верно всегда, в shared — только у полей,
        ни одного значения которых не забрал кто-то раньше.
        Numeric обязан быть непрерывным: на этом стоит
        арифметическое кодирование корзин.
        """

        if not self.candidates:
            return True

        start = self.candidates[0]

        return tuple(range(start, start + len(self.candidates))) == self.candidates

    @property
    def value_start(self) -> int:
        return self.candidates[0] if (self.candidates and self.is_contiguous) else -1

    @property
    def value_end(self) -> int:
        start = self.value_start
        return -1 if start < 0 else start + len(self.candidates)

    def to_json(self) -> dict:
        return {
            "id": self.key_token_id,
            "field_id": self.field_id,
            "key": self.key,
            "key_token_id": self.key_token_id,
            "namespace": self.namespace,
            "field": self.field,
            "kind": self.kind,
            "predictable": self.predictable,
            "arrow_type": self.arrow_type,
            "n_values": self.n_values,
            "contiguous": self.is_contiguous,
            "value_start": self.value_start,
            "value_end": self.value_end,
            "candidates": list(self.candidates),
        }

    @staticmethod
    def from_json(data: dict) -> "FieldEntry":
        return FieldEntry(
            field_id=int(data["field_id"]),
            key=data["key"],
            namespace=data["namespace"],
            field=data["field"],
            kind=data["kind"],
            predictable=bool(data["predictable"]),
            arrow_type=data["arrow_type"],
            key_token_id=int(data["key_token_id"]),
            candidates=tuple(int(value) for value in data["candidates"]),
        )


@dataclass(frozen=True)
class KeyToken:
    token_id: int
    name: str
    fields: tuple[int, ...]

    @property
    def shared(self) -> bool:
        return len(self.fields) > 1

    def to_json(self) -> dict:
        return {
            "token_id": self.token_id,
            "name": self.name,
            "shared": self.shared,
            "fields": list(self.fields),
        }

    @staticmethod
    def from_json(data: dict) -> "KeyToken":
        return KeyToken(
            token_id=int(data["token_id"]),
            name=data["name"],
            fields=tuple(int(value) for value in data["fields"]),
        )


@dataclass(frozen=True)
class ValueEntry:
    """
    Одна запись на value token.

    field_id здесь НЕТ намеренно: общий токен принадлежит
    нескольким полям. fields это производная принадлежность для
    отчётов; авторитет по составу и порядку кандидатов —
    field_value_ids / CandidateIndex.
    """

    id: int
    value: str
    arrow_type: str
    shared: bool
    count: int
    fields: tuple[int, ...]

    def to_json(self, owner: dict[int, str] | None = None) -> dict:

        data = {
            "id": self.id,
            "value": self.value,
            "arrow_type": self.arrow_type,
            "shared": self.shared,
            "count": self.count,
            "fields": list(self.fields),
        }

        # У токена ровно одного владельца имя поля пишется
        # рядом: так читается и baseline-словарь, и отчёт.
        if owner is not None and len(self.fields) == 1:
            data["key"] = owner[self.fields[0]]

        return data

    @staticmethod
    def from_json(data: dict) -> "ValueEntry":
        return ValueEntry(
            id=int(data["id"]),
            value=data["value"],
            arrow_type=data["arrow_type"],
            shared=bool(data["shared"]),
            count=int(data["count"]),
            fields=tuple(int(value) for value in data["fields"]),
        )


def parse_value(kind: str, arrow_type: str, text: str) -> Any:
    """
    Строка из value_vocab обратно в типизированное значение.
    """

    if kind == KIND_BOOLEAN or arrow_type == "bool":
        return text == "true"

    if arrow_type.startswith("int"):
        return int(text)

    if arrow_type in ("double", "float"):
        return float(text)

    return text


# ============================================================
# СЛОВАРЬ
# ============================================================


class Vocab:
    """
    Frozen словарь одного режима: special-токены, key tokens и
    value tokens в едином пространстве ID, плюс пространство
    полей, которое от режима не зависит.
    """

    def __init__(
        self,
        fields: Iterable[FieldEntry],
        key_tokens: Iterable[KeyToken],
        values: Iterable[ValueEntry],
        key_mode: str = DEFAULT_KEY_MODE,
        value_mode: str = DEFAULT_VALUE_MODE,
    ):

        check_mode(key_mode, value_mode)

        self.key_mode = key_mode
        self.value_mode = value_mode


        self.fields: tuple[FieldEntry, ...] = tuple(fields)
        self.key_tokens: tuple[KeyToken, ...] = tuple(key_tokens)
        self.values: tuple[ValueEntry, ...] = tuple(values)

        self._by_key_name: dict[str, FieldEntry] = {entry.key: entry for entry in self.fields}
        self._by_field_id: dict[int, FieldEntry] = {entry.field_id: entry for entry in self.fields}
        self._by_key_token: dict[int, KeyToken] = {token.token_id: token for token in self.key_tokens}

        self._value_token: dict[tuple[int, str], int] = {}

        for entry in self.fields:
            for token, value in zip(entry.candidates, self._value_strings(entry)):
                self._value_token[(entry.field_id, value)] = token

        self._check_layout()

        self._typed_cache: dict[int, pa.Array] = {}

        self._build_tables()

    # --------------------------------------------------------

    def _value_strings(self, entry: FieldEntry) -> list[str]:
        """
        Строки значений поля в порядке кандидатов.
        """

        first = self.first_value_id

        return [self.values[token - first].value for token in entry.candidates]

    def _build_tables(self) -> None:
        """
        Плотные таблицы для горячих путей.

        local_lookup это [n_fields + 1, size] int32, -1 значит
        «значение не кандидат этого поля». Ради него всё и
        затевалось: «домены полей не смешиваются» становится
        данными, а не рассуждением, и проверка стоит одно
        индексирование.

        Граница применимости: на реальных HCB с сотнями полей и
        десятками тысяч значений таблицу придётся сделать
        разреженной. При 57 полях и 738 токенах это 171 КБ.
        """

        size = self.size
        rows = N_FIELDS + 1

        self.predictable_by_field = np.zeros(rows, dtype=bool)
        self.n_candidates_by_field = np.zeros(rows, dtype=np.int64)
        self.key_token_by_field = np.zeros(rows, dtype=np.int64)

        # Специальная позиция поля не имеет: key token у неё
        # свой собственный special ID, он приходит в потоке.
        self.key_token_by_field[NO_FIELD] = SPECIAL_IDS["[UNK]"]

        self.local_lookup = np.full((rows, size), -1, dtype=np.int32)

        offsets = np.zeros(rows + 1, dtype=np.int64)
        flat: list[int] = []

        for entry in self.fields:

            field_id = entry.field_id

            self.predictable_by_field[field_id] = entry.predictable
            self.n_candidates_by_field[field_id] = entry.n_values
            self.key_token_by_field[field_id] = entry.key_token_id

            for local, token in enumerate(entry.candidates):
                self.local_lookup[field_id, token] = local

            flat.extend(entry.candidates)
            offsets[field_id + 1] = len(flat)

        offsets[NO_FIELD + 1] = len(flat)

        self.candidate_offsets = offsets
        self.candidates_flat = np.asarray(flat, dtype=np.int64)

    # --------------------------------------------------------

    def _check_layout(self) -> None:
        """
        ID непрерывны и сгруппированы, кандидаты покрывают все
        value token ровно один раз.
        """

        # Полнота реестра здесь НЕ проверяется: build_vocab
        # строит поля по key_specs() и иначе не умеет, а тесты
        # законно собирают крошечные словари из трёх полей.
        # Сверку с реестром делает FieldTable.load по
        # field_value_ids.json.
        if len(self.fields) > N_FIELDS:
            raise IncompatibleArtifactsError(
                f"полей в словаре {len(self.fields)}, а в реестре preprocessing их {N_FIELDS}"
            )

        for index, entry in enumerate(self.fields):
            if entry.field_id != index:
                raise IncompatibleArtifactsError(
                    f"поле {entry.key} имеет field_id {entry.field_id}, ожидался {index}"
                )

        for index, token in enumerate(self.key_tokens):
            if token.token_id != N_SPECIAL + index:
                raise IncompatibleArtifactsError(
                    f"key token {token.name} имеет ID {token.token_id}, ожидался {N_SPECIAL + index}"
                )

        first = self.first_value_id

        for entry in self.fields:

            token = self._by_key_token.get(entry.key_token_id)

            if token is None:
                raise IncompatibleArtifactsError(
                    f"поле {entry.key} ссылается на key token {entry.key_token_id}, которого нет"
                )

            if entry.field_id not in token.fields:
                raise IncompatibleArtifactsError(
                    f"key token {token.name} не числит среди своих полей {entry.key}"
                )

            if len(set(entry.candidates)) != len(entry.candidates):
                raise IncompatibleArtifactsError(
                    f"поле {entry.key} дважды называет один и тот же value token"
                )

            for candidate in entry.candidates:
                if not (first <= candidate < self.size):
                    raise IncompatibleArtifactsError(
                        f"кандидат {candidate} поля {entry.key} вне диапазона значений "
                        f"[{first}, {self.size})"
                    )

            # Корзины кодируются арифметикой, поэтому numeric
            # обязан остаться непрерывным в любом режиме.
            if entry.kind == KIND_NUMERIC and not entry.is_contiguous:
                raise IncompatibleArtifactsError(
                    f"numeric-поле {entry.key} получило разрывный набор корзин: "
                    "корзины не делятся между полями ни в одном режиме"
                )

        for index, entry in enumerate(self.values):
            if entry.id != first + index:
                raise IncompatibleArtifactsError(f"значение {entry.value} имеет разрывный ID {entry.id}")

        covered = {token for entry in self.fields for token in entry.candidates}

        if len(covered) != self.n_values:
            missing = sorted(set(range(first, self.size)) - covered)
            raise IncompatibleArtifactsError(
                f"value token без владельца: {missing[:8]}; кандидаты полей обязаны покрывать словарь"
            )

    # --------------------------------------------------------

    @property
    def n_fields(self) -> int:
        return len(self.fields)

    @property
    def n_key_tokens(self) -> int:
        return len(self.key_tokens)

    @property
    def n_values(self) -> int:
        return len(self.values)

    @property
    def first_value_id(self) -> int:
        return N_SPECIAL + self.n_key_tokens

    @property
    def size(self) -> int:
        return N_SPECIAL + self.n_key_tokens + self.n_values

    @property
    def is_baseline(self) -> bool:
        return is_baseline(self.key_mode, self.value_mode)

    @property
    def modes(self) -> dict[str, str]:
        return {"key_mode": self.key_mode, "categorical_value_mode": self.value_mode}

    # --------------------------------------------------------

    def field_entry(self, key: str) -> FieldEntry | None:
        return self._by_key_name.get(key)

    def field_entry_by_id(self, field_id: int) -> FieldEntry | None:
        return self._by_field_id.get(int(field_id))

    def field_id(self, namespace: str, field: str) -> int | None:
        entry = self._by_key_name.get(KEY_FORMAT.format(namespace=namespace, field=field))
        return None if entry is None else entry.field_id

    def key_token_id(self, namespace: str, field: str) -> int | None:
        entry = self._by_key_name.get(KEY_FORMAT.format(namespace=namespace, field=field))
        return None if entry is None else entry.key_token_id

    def value_token(self, field_id: int, value: str) -> int | None:
        """
        Токен значения ВНУТРИ поля.

        Ключ поиска это поле, а не key token: в semantic-режиме
        два поля делят key token, но множества значений у них
        по-прежнему свои.
        """

        return self._value_token.get((int(field_id), value))

    def decode(self, token_id: int) -> str:
        """
        Человекочитаемое имя токена: для отчётов и golden-векторов.
        """

        token_id = int(token_id)

        if 0 <= token_id < N_SPECIAL:
            return SPECIAL_TOKENS[token_id]

        token = self._by_key_token.get(token_id)

        if token is not None:
            return token.name

        index = token_id - self.first_value_id

        if 0 <= index < self.n_values:
            return self.values[index].value

        raise KeyError(f"ID {token_id} вне словаря")

    def typed_values(self, field_id: int) -> pa.Array:
        """
        Значения поля в типе исходного поля, в порядке
        кандидатов: для pc.index_in.

        Для numeric не используется: там колонка __bucket.
        """

        field_id = int(field_id)

        cached = self._typed_cache.get(field_id)

        if cached is not None:
            return cached

        entry = self._by_field_id[field_id]

        arrow_type = ARROW_TYPES.get(entry.arrow_type, pa.string())

        parsed = [
            parse_value(entry.kind, entry.arrow_type, value)
            for value in self._value_strings(entry)
        ]

        array = pa.array(parsed, type=arrow_type)

        self._typed_cache[field_id] = array

        return array

    # --------------------------------------------------------

    def field_value_ids(self) -> dict:
        return {
            "rule": (
                "поле -> допустимые value token в ПОРЯДКЕ ПОЛЯ (типизированном), а не по возрастанию токена: "
                "локальный индекс кандидата это позиция в этом списке, у numeric он равен номеру корзины. "
                "numeric это все корзины train-artifact, остальные поля это значения, встреченные на train; "
                "special токены в кандидаты не входят"
            ),
            "field_id_rule": FIELD_ID_RULE,
            "modes": self.modes,
            "fields": {
                entry.key: {
                    "field_id": entry.field_id,
                    "key_token_id": entry.key_token_id,
                    "kind": entry.kind,
                    "predictable": entry.predictable,
                    "n_candidates": entry.n_values,
                    "value_ids": list(entry.candidates),
                }
                for entry in self.fields
            },
        }

    def candidates(self) -> "CandidateIndex":
        return CandidateIndex(self)

    def sharing_report(self) -> dict:
        """
        Что именно склеилось: для artifacts и для отчёта.
        """

        key_groups = [
            {"token_id": token.token_id, "name": token.name, "keys": [self.fields[f].key for f in token.fields]}
            for token in self.key_tokens
            if token.shared
        ]

        value_groups = [
            {
                "token_id": entry.id,
                "value": entry.value,
                "arrow_type": entry.arrow_type,
                "keys": [self.fields[f].key for f in entry.fields],
            }
            for entry in self.values
            if len(entry.fields) > 1
        ]

        return {
            "modes": self.modes,
            "n_fields": self.n_fields,
            "n_key_tokens": self.n_key_tokens,
            "n_value_tokens": self.n_values,
            "size": self.size,
            "n_merged_key_tokens": len(key_groups),
            "n_merged_value_tokens": len(value_groups),
            "merged_keys": key_groups,
            "merged_values": value_groups,
        }

    # --------------------------------------------------------

    def save(self, directory: Path) -> None:

        directory = Path(directory)

        owner = {entry.field_id: entry.key for entry in self.fields}

        write_json(
            directory / SPECIAL_TOKENS_FILE,
            {
                "n_special": N_SPECIAL,
                "ids": dict(SPECIAL_IDS),
                "tokens": [
                    {"id": SPECIAL_IDS[name], "token": name, "note": SPECIAL_NOTES[name]}
                    for name in SPECIAL_TOKENS
                ],
            },
        )

        write_json(
            directory / KEY_VOCAB_FILE,
            {
                "schema_version": SCHEMA_VERSION,
                "key_format": KEY_FORMAT,
                "first_key_id": N_SPECIAL,
                "modes": self.modes,
                "field_id_rule": FIELD_ID_RULE,
                "n_fields": self.n_fields,
                "n_keys": self.n_key_tokens,
                "keys": [entry.to_json() for entry in self.fields],
                "key_tokens": [token.to_json() for token in self.key_tokens],
            },
        )

        write_json(
            directory / VALUE_VOCAB_FILE,
            {
                "schema_version": SCHEMA_VERSION,
                "first_value_id": self.first_value_id,
                "n_values": self.n_values,
                "modes": self.modes,
                "values": [entry.to_json(owner) for entry in self.values],
            },
        )

        write_json(directory / FIELD_VALUE_IDS_FILE, self.field_value_ids())

    @staticmethod
    def load(directory: Path) -> "Vocab":

        directory = Path(directory)

        specials = read_json(directory / SPECIAL_TOKENS_FILE)

        if specials["ids"] != SPECIAL_IDS:
            raise IncompatibleArtifactsError("special-токены словаря не совпадают с контрактом")

        keys = read_json(directory / KEY_VOCAB_FILE)
        values = read_json(directory / VALUE_VOCAB_FILE)

        modes = keys["modes"]

        return Vocab(
            fields=[FieldEntry.from_json(item) for item in keys["keys"]],
            key_tokens=[KeyToken.from_json(item) for item in keys["key_tokens"]],
            values=[ValueEntry.from_json(item) for item in values["values"]],
            key_mode=modes["key_mode"],
            value_mode=modes["categorical_value_mode"],
        )


class CandidateIndex:
    """
    Перевод между value token и локальным индексом кандидата
    внутри ПОЛЯ.

    Локальный индекс это позиция в типизированном порядке поля.
    Вычитанием смещения он больше не выражается: в shared-режиме
    кандидаты поля не обязаны идти подряд.
    """

    def __init__(self, vocab: Vocab):
        self.local_lookup = vocab.local_lookup
        self.n_candidates = vocab.n_candidates_by_field
        self.offsets = vocab.candidate_offsets
        self.flat = vocab.candidates_flat

    def to_local(self, field_ids, value_ids) -> np.ndarray:
        """
        -1 означает «значение не кандидат этого поля». Решение о
        том, ошибка это или нет, принимает вызывающий.
        """

        fields = np.asarray(field_ids, dtype=np.int64)
        values = np.asarray(value_ids, dtype=np.int64)

        return self.local_lookup[fields, values].astype(np.int64)

    def to_global(self, field_ids, local) -> np.ndarray:
        fields = np.asarray(field_ids, dtype=np.int64)
        index = np.asarray(local, dtype=np.int64)

        return self.flat[self.offsets[fields] + index]

    def size_of(self, field_ids) -> np.ndarray:
        return self.n_candidates[np.asarray(field_ids, dtype=np.int64)]


# ============================================================
# КОЛОНКИ PROCESSED
# ============================================================


def events_column(spec: FieldSpec) -> str:
    """
    Из какой колонки events.parquet берётся значение поля.
    """

    if spec.namespace == "timeline":
        return spec.field

    return spec.bucket_column if spec.is_numeric else spec.column


def profile_column(spec: FieldSpec) -> str:
    return f"{spec.field}__bucket" if spec.is_numeric else spec.field


def event_key_specs() -> list[FieldSpec]:
    return [spec for spec in key_specs() if spec.namespace != "profile"]


def profile_key_specs() -> list[FieldSpec]:
    order = {name: index for index, name in enumerate(PROFILE_FIELDS)}

    specs = [spec for spec in key_specs() if spec.namespace == "profile"]

    return sorted(specs, key=lambda spec: order[spec.field])


def field_ids_by_key() -> dict[str, int]:
    """
    Физическое имя поля -> field_id. Не зависит от режима.
    """

    return {spec.column: index for index, spec in enumerate(key_specs())}


# ============================================================
# FIT
# ============================================================


@dataclass(frozen=True)
class FitReport:
    n_clients: int
    n_events: int
    n_snapshots: int
    cutoff_max: str | None

    def as_dict(self) -> dict:
        return {
            "dataset": "train",
            "n_fit_clients": self.n_clients,
            "n_fit_events": self.n_events,
            "n_fit_snapshots": self.n_snapshots,
            "fit_cutoff_max": self.cutoff_max,
            "rule": (
                "fit-набор словаря это fit-набор preprocessing: события train-клиентов с "
                "ts < последний валидный train-cutoff клиента и as-of снимки валидных train-примеров, "
                "каждая запись один раз"
            ),
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IncompatibleArtifactsError(message)


def load_fit_scope(processed_dir: Path, split_manifest: dict) -> FitScope:
    """
    Fit-набор словаря это fit-набор preprocessing: тот же код,
    та же выборка. Состав сверяется со split manifest.
    """

    index = pq.read_table(Path(processed_dir) / "cutoff_index.parquet")

    _require(
        index.schema.equals(CUTOFF_INDEX_SCHEMA),
        "cutoff_index.parquet имеет чужую схему: preprocessing другой версии",
    )

    scope = fit_scope(index)

    train_ids = sorted(
        {
            int(client_id)
            for client_id, group in zip(
                index.column("client_id").to_pylist(), index.column("client_group").to_pylist()
            )
            if group == "train"
        }
    )

    clients = split_manifest.get("clients", {})

    _require(
        sha256_ints(train_ids) == clients.get("sha256", {}).get("train"),
        "состав train-клиентов не совпадает со split_manifest.json",
    )

    _require(
        scope.n_clients == clients.get("fit_clients"),
        f"fit-клиентов {scope.n_clients}, в split_manifest {clients.get('fit_clients')}",
    )

    return scope


def _count_column(counter: Counter, column) -> None:
    """
    Считает непустые значения колонки типизированными ключами:
    строка появляется только при записи словаря.
    """

    present = pc.drop_null(column)

    if len(present) == 0:
        return

    for item in pc.value_counts(present).to_pylist():
        counter[item["values"]] += int(item["counts"])


def count_events(processed_dir: Path, scope: FitScope) -> tuple[dict[tuple[str, str], Counter], int]:
    """
    Значения полей событий на fit-наборе. Колонка чужого типа
    события всегда null, поэтому фильтр по типу не нужен.
    """

    specs = event_key_specs()

    counters: dict[tuple[str, str], Counter] = {spec.key: Counter() for spec in specs}

    parquet = pq.ParquetFile(Path(processed_dir) / "clients" / "train_clients" / "events.parquet")

    columns = sorted({"client_id", "ts", *(events_column(spec) for spec in specs)})

    total = 0

    for index in range(parquet.num_row_groups):

        batch = parquet.read_row_group(index, columns=columns)

        if batch.num_rows == 0:
            continue

        mask = scope.mask_for(batch.column("client_id").to_numpy(), batch.column("ts").to_numpy())

        if not mask.any():
            continue

        rows = batch.filter(pa.array(mask))

        total += rows.num_rows

        for spec in specs:
            _count_column(counters[spec.key], rows.column(events_column(spec)))

    return counters, total


def count_profile(processed_dir: Path, scope: FitScope) -> tuple[dict[tuple[str, str], Counter], int]:
    """
    Значения профиля на тех снимках, которые реально выбраны
    как as-of валидными train-примерами.
    """

    specs = profile_key_specs()

    counters: dict[tuple[str, str], Counter] = {spec.key: Counter() for spec in specs}

    profile = pq.read_table(Path(processed_dir) / "clients" / "train_clients" / "profile.parquet")

    if profile.num_rows == 0 or not scope.snapshots:
        return counters, 0

    wanted = {
        (int(client_id), int(np.datetime64(stamp, "us").astype(np.int64)))
        for client_id, stamp in scope.snapshots
    }

    client_id = profile.column("client_id").to_numpy()
    ts = profile.column("ts").to_numpy().astype("datetime64[us]").astype(np.int64)

    mask = np.array(
        [(int(cid), int(stamp)) in wanted for cid, stamp in zip(client_id, ts)],
        dtype=bool,
    )

    rows = profile.filter(pa.array(mask))

    for spec in specs:
        _count_column(counters[spec.key], rows.column(profile_column(spec)))

    return counters, rows.num_rows


def _ordered_values(spec: FieldSpec, counter: Counter, bucket_count: int | None) -> list[tuple[str, int]]:
    """
    Значения поля в типизированном порядке.

    Это и есть порядок локальных индексов: у numeric он совпадает
    с номером корзины, у остального — с возрастанием значения.
    Выдача token_id идёт поверх него и порядок не меняет.
    """

    if spec.kind == KIND_NUMERIC:

        if bucket_count is None:
            return []

        empty = [index for index in range(bucket_count) if counter.get(index, 0) == 0]

        if empty:
            raise IncompatibleArtifactsError(
                f"{spec.namespace}.{spec.field}: корзины {empty} пусты на train, "
                "это противоречит инварианту preprocessing"
            )

        return [(value_key(index), int(counter[index])) for index in range(bucket_count)]

    return [(value_key(value), int(count)) for value, count in sorted(counter.items())]


def build_vocab(
    counters: dict[tuple[str, str], Counter],
    bucket_counts: dict[tuple[str, str], int | None],
    key_mode: str = DEFAULT_KEY_MODE,
    value_mode: str = DEFAULT_VALUE_MODE,
) -> Vocab:
    """
    Словарь режима из счётчиков train.

    Счётчики одни и те же для всех четырёх режимов: они считаются
    по ПОЛЯМ, а склейка касается только выдачи token_id. Поэтому
    один проход по данным даёт сразу четыре словаря.
    """

    check_mode(key_mode, value_mode)

    specs = key_specs()

    validate_registry(spec.column for spec in specs)

    # ---- key tokens -------------------------------------------------

    token_of_name: dict[str, int] = {}
    members: list[list[int]] = []
    names: list[str] = []

    key_token_of_field: list[int] = []

    for field_id, spec in enumerate(specs):

        name = semantic_key_name(spec.column, key_mode)

        token = token_of_name.get(name)

        if token is None:
            token = N_SPECIAL + len(names)
            token_of_name[name] = token
            names.append(name)
            members.append([])

        members[token - N_SPECIAL].append(field_id)
        key_token_of_field.append(token)

    key_tokens = [
        KeyToken(token_id=N_SPECIAL + index, name=names[index], fields=tuple(members[index]))
        for index in range(len(names))
    ]

    # ---- value tokens -----------------------------------------------

    first_value_id = N_SPECIAL + len(key_tokens)

    token_of_class: dict[tuple, int] = {}

    value_strings: list[str] = []
    value_types: list[str] = []
    value_shared: list[bool] = []
    value_counts: list[int] = []
    value_fields: list[list[int]] = []

    fields: list[FieldEntry] = []

    next_token = first_value_id

    for field_id, spec in enumerate(specs):

        arrow_type = str(spec.arrow_type)

        ordered = _ordered_values(spec, counters.get(spec.key, Counter()), bucket_counts.get(spec.key))

        candidates: list[int] = []

        for value, count in ordered:

            group = value_class(field_id, spec.kind, arrow_type, value, value_mode)

            token = token_of_class.get(group)

            if token is None:

                token = next_token
                next_token += 1

                token_of_class[group] = token

                value_strings.append(value)
                value_types.append(arrow_type)
                value_shared.append(shares_values(spec.kind, arrow_type, value_mode))
                value_counts.append(0)
                value_fields.append([])

            index = token - first_value_id

            value_counts[index] += count
            value_fields[index].append(field_id)

            candidates.append(token)

        fields.append(
            FieldEntry(
                field_id=field_id,
                key=spec.column,
                namespace=spec.namespace,
                field=spec.field,
                kind=spec.kind,
                predictable=spec.predictable,
                arrow_type=arrow_type,
                key_token_id=key_token_of_field[field_id],
                candidates=tuple(candidates),
            )
        )

    values = [
        ValueEntry(
            id=first_value_id + index,
            value=value_strings[index],
            arrow_type=value_types[index],
            shared=value_shared[index],
            count=value_counts[index],
            fields=tuple(value_fields[index]),
        )
        for index in range(len(value_strings))
    ]

    return Vocab(
        fields=fields,
        key_tokens=key_tokens,
        values=values,
        key_mode=key_mode,
        value_mode=value_mode,
    )


@dataclass(frozen=True)
class FitCounters:
    """
    Результат единственного прохода по train.

    Счётчики не зависят от режима, поэтому четыре словаря
    строятся из одного и того же объекта.
    """

    counters: dict[tuple[str, str], Counter]
    bucket_counts: dict[tuple[str, str], int | None]
    report: FitReport

    def build(self, key_mode: str = DEFAULT_KEY_MODE, value_mode: str = DEFAULT_VALUE_MODE) -> Vocab:
        return build_vocab(self.counters, self.bucket_counts, key_mode, value_mode)


def fit_counters(processed_dir: Path, artifacts_dir: Path) -> FitCounters:
    """
    Один проход по train: частоты значений по ПОЛЯМ.
    """

    processed_dir = Path(processed_dir)
    artifacts_dir = Path(artifacts_dir)

    split_manifest = read_json(artifacts_dir / "split_manifest.json")
    field_stats = read_json(artifacts_dir / "field_stats.json")
    bucket_edges = read_json(artifacts_dir / "bucket_edges.json")

    _require(
        split_manifest.get("schema_version") == SCHEMA_VERSION,
        "split_manifest.json другой версии схемы preprocessing",
    )

    scope = load_fit_scope(processed_dir, split_manifest)

    event_counters, n_events = count_events(processed_dir, scope)
    profile_counters, n_snapshots = count_profile(processed_dir, scope)

    counters = {**event_counters, **profile_counters}

    expected_events = field_stats["fields"]["timeline"]["event_type"]["n_total"]
    expected_snapshots = field_stats["fields"]["profile"]["age"]["n_total"]

    _require(
        n_events == expected_events,
        f"fit-событий {n_events}, в field_stats {expected_events}",
    )

    _require(
        n_snapshots == expected_snapshots,
        f"fit-снимков {n_snapshots}, в field_stats {expected_snapshots}",
    )

    specs = load_specs(bucket_edges)

    bucket_counts: dict[tuple[str, str], int | None] = {}

    for spec in key_specs():

        if spec.kind != KIND_NUMERIC:
            continue

        bucket = specs.get(spec.key)

        _require(bucket is not None, f"в bucket_edges.json нет поля {spec.namespace}.{spec.field}")

        bucket_counts[spec.key] = None if bucket.status == STATUS_NO_FIT_DATA else bucket.actual_bucket_count

    cutoff_max = scope.max_cutoff

    report = FitReport(
        n_clients=scope.n_clients,
        n_events=n_events,
        n_snapshots=n_snapshots,
        cutoff_max=cutoff_max.isoformat() if isinstance(cutoff_max, datetime) else None,
    )

    return FitCounters(counters=counters, bucket_counts=bucket_counts, report=report)


def fit_vocab(
    processed_dir: Path,
    artifacts_dir: Path,
    key_mode: str = DEFAULT_KEY_MODE,
    value_mode: str = DEFAULT_VALUE_MODE,
) -> tuple[Vocab, FitReport]:
    """
    Полный fit словаря: только train, каждая запись один раз.
    """

    counters = fit_counters(processed_dir, artifacts_dir)

    return counters.build(key_mode, value_mode), counters.report


__all__ = [
    "ARROW_TYPES",
    "CandidateIndex",
    "FIELD_ID_RULE",
    "FieldEntry",
    "FitCounters",
    "FitReport",
    "KEY_FORMAT",
    "KeyToken",
    "NO_FIELD",
    "N_FIELDS",
    "ORDER_RULES",
    "VALUE_RULES",
    "ValueEntry",
    "Vocab",
    "build_vocab",
    "count_events",
    "count_profile",
    "event_key_specs",
    "events_column",
    "field_ids_by_key",
    "fit_counters",
    "fit_vocab",
    "key_specs",
    "load_fit_scope",
    "parse_value",
    "profile_column",
    "profile_key_specs",
    "registry_digest",
]
