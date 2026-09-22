from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.preprocessing.artifacts import read_json

from .numeric import FittedEncoder, load_encoders
from .settings import TOKENIZER_FILE, tokenizer_path
from .specials import KIND_BUCKET, KIND_CATEGORICAL, SPECIAL_ROLES, SPECIAL_TOKENS
from .text import BpeModel, load_bpe
from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


# ============================================================
# ЭТАП 6: ФИНАЛЬНЫЙ СЛОВАРЬ
# ============================================================
#
# Все виды токенов живут в одном пространстве ID непересекающимися
# диапазонами, и порядок диапазонов задан раз и навсегда:
#
#   специальные -> ключи -> категории -> диапазоны чисел -> BPE
#
# Логически словари ключей и значений раздельны: ключ отвечает
# на вопрос «что это», значение — «чему равно». Физически их ID
# могут лежать в одной таблице embedding, и ровно поэтому
# диапазоны обязаны не пересекаться.
#
# Номера здесь не назначаются заново: каждый этап уже выдал их
# своим токенам, а финальный словарь только складывает части
# вместе и проверяет, что ни один номер не повторился и ни один
# не потерялся. После этого словарь не меняется.
# ============================================================


class LayoutError(ValueError):
    """
    Финальный словарь собрать или прочитать нельзя.
    """


def build_tokenizer(specials: dict, key_vocab: dict, value_vocab: dict, buckets: dict,
                    bpe: dict) -> dict:
    """
    Один словарь из пяти видов токенов.
    """

    _check_chain(specials, key_vocab, value_vocab, buckets)

    first_key_id = int(key_vocab["first_key_id"])
    first_value_id = int(value_vocab["first_value_id"])
    first_bucket_id = int(buckets["first_bucket_id"])
    bpe_offset = int(buckets["next_id"])

    size = bpe_offset + int(bpe.get("size") or 0)

    # --- значения одним списком в порядке ID ---

    values: list[dict] = list(value_vocab["values"])

    for key in sorted(buckets["encoders"]):

        entry = buckets["encoders"][key]

        for bucket in entry["buckets"]:
            values.append(
                {
                    "id": bucket["id"],
                    "kind": KIND_BUCKET,
                    "domain": None,
                    "key": key,
                    "value_type": KIND_BUCKET,
                    "value": str(bucket["index"]),
                    "label": bucket["label"],
                }
            )

    _check_ids(key_vocab["keys"], values, first_key_id, first_value_id, size)

    # --- кандидаты по ключам ---
    #
    # Что вообще может стоять в слоте значения этого ключа.
    # Нужно и для проверки, и для масок допустимых целей.

    candidates: dict[str, dict] = {}

    for row in key_vocab["keys"]:

        key = row["key"]

        if row["value_kind"] == KIND_CATEGORICAL:
            candidates[key] = {
                "kind": KIND_CATEGORICAL,
                "domain": row["domain"],
                "value_ids": list(value_vocab["by_key"].get(key, ())),
            }

        elif row["value_kind"] == "numeric":
            candidates[key] = {
                "kind": KIND_BUCKET,
                "domain": None,
                "value_ids": list(buckets["by_key"].get(key, ())),
            }

        else:
            candidates[key] = {
                "kind": "text",
                "domain": None,
                "value_ids": [],
                "bpe": bool(bpe.get("enabled")),
            }

    layout = {
        "format_version": FORMAT_VERSION,
        "ranges": {
            "special": [0, first_key_id],
            "keys": [first_key_id, first_value_id],
            "values": [first_value_id, bpe_offset],
            "bpe": [bpe_offset, size],
        },
        "bpe_offset": bpe_offset,
        "size": size,
        "sizes": {
            "special": int(specials["size"]),
            "keys": int(key_vocab["size"]),
            "categorical": int(value_vocab["size"]),
            "buckets": int(buckets["size"]),
            "values": int(value_vocab["size"]) + int(buckets["size"]),
            "bpe": int(bpe.get("size") or 0),
            "total": size,
        },
        "invariants": [
            "диапазоны не пересекаются и покрывают всё пространство без дыр",
            "ID куска BPE это bpe_offset плюс его номер в модели разбиения",
            "специальный токен в слоте значения означает не значение, а его отсутствие или невозможность",
            "ключи и значения логически раздельны, физически делят одну таблицу embedding",
        ],
        "rule": "специальные -> ключи -> категории -> диапазоны чисел -> BPE",
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "format_version": FORMAT_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "fit": value_vocab["fit"],
        "config_sha256": value_vocab["config_sha256"],
        "layout": layout,
        "specials": {
            "ids": {row["token"]: row["id"] for row in specials["tokens"]},
            "roles": {row["token"]: row["role"] for row in specials["tokens"]},
            "never_written": specials["never_written"],
            "marker_owner": specials["marker_owner"],
            "boundaries": specials["boundaries"],
        },
        "keys": {
            "first_key_id": first_key_id,
            "size": int(key_vocab["size"]),
            "rows": key_vocab["keys"],
            "link_keys": key_vocab["link_keys"],
        },
        "values": {
            "first_value_id": first_value_id,
            "first_bucket_id": first_bucket_id,
            "size": len(values),
            "rows": values,
        },
        "candidates": {"keys": candidates, "rule": "кандидаты ключа в порядке его домена"},
        "numeric": {"encoders": buckets["encoders"]},
        "declared_by_event_type": key_vocab["declared_by_event_type"],
        "bpe": {**bpe, "offset": bpe_offset},
        "tokens": _token_rows(specials, key_vocab, values, bpe, bpe_offset),
        "versions": {
            "keys": key_vocab["keys_version"],
            "projection": key_vocab["projection_version"],
        },
        "rules": {
            "frozen": "после сборки словарь не меняется: кодирование не добавляет ни одного токена",
            "refit": "новый fit вправе переставить ID; совместимость со старым checkpoint "
                     "автоматически не обещается",
            "markers": "маркеры события и профиля ставит токенизатор, marker_owner = tokenizer",
            "reuse": "val и test кодируются этим же словарём: второй раз он не учится",
        },
    }


def _token_rows(specials: dict, key_vocab: dict, values: list[dict], bpe: dict,
                bpe_offset: int) -> list[dict]:
    """
    Одна строка на КАЖДЫЙ токен словаря, в порядке ID.

    По этой таблице любой ID читается без знания того, как
    он был назначен: тип, строковое обозначение и ключ, если
    токен принадлежит конкретному полю.
    """

    rows: list[dict] = []

    for item in specials["tokens"]:
        rows.append(
            {
                "id": item["id"],
                "token": item["token"],
                "type": "special",
                "key": None,
                "role": item["role"],
            }
        )

    for item in key_vocab["keys"]:
        rows.append(
            {
                "id": item["id"],
                "token": item["key"],
                "type": "key",
                "key": item["key"],
                "value_kind": item["value_kind"],
                "unit": item["unit"],
                "domain": item["domain"],
            }
        )

    for item in values:

        categorical = item["kind"] == KIND_CATEGORICAL

        rows.append(
            {
                "id": item["id"],
                "token": item["label"],
                "type": "categorical" if categorical else "numeric_bucket",
                "key": item["key"],
                "domain": item["domain"],
                "value_type": item["value_type"],
                "value": item["value"],
                **({"keys": item["keys"]} if categorical and "keys" in item else {}),
            }
        )

    model = bpe.get("model") or {}

    vocab = (model.get("model") or {}).get("vocab") or {}

    for piece, local in sorted(vocab.items(), key=lambda pair: pair[1]):
        rows.append(
            {
                "id": bpe_offset + int(local),
                "token": piece,
                "type": "bpe",
                "key": None,
                "piece": int(local),
            }
        )

    return rows



def _check_chain(specials: dict, key_vocab: dict, value_vocab: dict, buckets: dict) -> None:
    """
    Части словаря обязаны быть собраны одна поверх другой.

    Иначе диапазоны наложились бы друг на друга, и один номер
    означал бы два разных токена.
    """

    if int(key_vocab["first_key_id"]) != int(specials["next_id"]):
        raise LayoutError(
            "словарь ключей собран поверх другого набора специальных токенов: "
            f"ключи начинаются с {key_vocab['first_key_id']}, а специальные заканчиваются "
            f"на {specials['next_id']}. Выполните key-vocab заново"
        )

    if int(value_vocab["first_value_id"]) != int(key_vocab["next_id"]):
        raise LayoutError(
            "каталог значений собран поверх другого словаря ключей: значения начинаются с "
            f"{value_vocab['first_value_id']}, а ключи заканчиваются на {key_vocab['next_id']}. "
            "Выполните value-vocab заново"
        )

    if int(buckets["first_bucket_id"]) != int(value_vocab["next_id"]):
        raise LayoutError(
            "числовые диапазоны собраны поверх другого каталога значений: диапазоны начинаются с "
            f"{buckets['first_bucket_id']}, а значения заканчиваются на {value_vocab['next_id']}. "
            "Выполните buckets заново"
        )

    if buckets["fit"]["fit_content_sha256"] != value_vocab["fit"]["fit_content_sha256"]:
        raise LayoutError(
            "значения и диапазоны посчитаны по разным данным train: выполните этапы заново"
        )


def _check_ids(keys: list[dict], values: list[dict], first_key_id: int, first_value_id: int,
               size: int) -> None:
    """
    Каждый номер встречается один раз, и дыр между диапазонами
    нет.
    """

    key_ids = [row["id"] for row in keys]
    value_ids = [row["id"] for row in values]

    if key_ids != list(range(first_key_id, first_key_id + len(key_ids))):
        raise LayoutError("номера ключей идут не подряд: словарь ключей собран не этим кодом")

    if value_ids != list(range(first_value_id, first_value_id + len(value_ids))):
        raise LayoutError(
            "номера значений идут не подряд: каталог значений и диапазоны собраны не одной цепочкой"
        )

    if size < first_value_id + len(value_ids):
        raise LayoutError("размер словаря меньше числа выданных номеров")


# ------------------------------------------------------------
# ЗАМОРОЖЕННЫЙ СЛОВАРЬ
# ------------------------------------------------------------


@dataclass
class FrozenArtifacts:
    """
    Готовый словарь, которым кодируют и расшифровывают.

    Ничего не обучает и не меняет: при загрузке проверяет, что
    пространство ID цело.
    """

    path: Path | None
    bundle: dict
    layout: dict
    specials: dict[str, int]
    key_ids: dict[str, int]
    key_info: dict[str, dict]
    domain_of: dict[str, str]
    value_ids: dict[tuple, int]
    value_rows: list[dict]
    candidates: dict[str, dict]
    encoders: dict[str, FittedEncoder]
    declared_by_event_type: dict[str, tuple[str, ...]]
    link_keys: frozenset[str]
    profile_keys: tuple[str, ...]
    bpe: BpeModel

    @property
    def first_value_id(self) -> int:
        return self.layout["ranges"]["values"][0]

    @property
    def bpe_offset(self) -> int:
        return self.layout["bpe_offset"]

    @property
    def size(self) -> int:
        return self.layout["size"]

    def special(self, name: str) -> int:
        return self.specials[name]

    def key_id(self, key: str) -> int | None:
        return self.key_ids.get(key)

    def categorical_id(self, key: str, value_type: str, value: str) -> int | None:
        return self.value_ids.get((self.domain_of[key], value_type, value))

    def bucket_id(self, key: str, index: int) -> int:
        return self.value_ids[(key, KIND_BUCKET, str(index))]

    def describe(self, token_id: int) -> dict:
        """
        Человекочитаемая расшифровка одного ID.
        """

        if token_id < len(SPECIAL_TOKENS):
            name = SPECIAL_TOKENS[token_id]
            return {"id": token_id, "kind": "special", "label": name, "role": SPECIAL_ROLES[name]}

        keys_range = self.layout["ranges"]["keys"]

        if keys_range[0] <= token_id < keys_range[1]:
            key = next(name for name, value in self.key_ids.items() if value == token_id)
            return {"id": token_id, "kind": "key", "label": key}

        values_range = self.layout["ranges"]["values"]

        if values_range[0] <= token_id < values_range[1]:
            row = self.value_rows[token_id - values_range[0]]
            return {"id": token_id, "kind": row["kind"], "label": row["label"]}

        piece = self.bpe.piece(token_id - self.bpe_offset)

        return {"id": token_id, "kind": "bpe", "label": piece}

    def verify(self) -> None:
        """
        Пространство ID цело: диапазоны идут подряд и номера не
        повторяются.
        """

        ranges = self.layout["ranges"]

        edges = [ranges["special"], ranges["keys"], ranges["values"], ranges["bpe"]]

        if edges[0][0] != 0:
            raise LayoutError("пространство ID начинается не с нуля")

        for left, right in zip(edges, edges[1:]):
            if left[1] != right[0]:
                raise LayoutError(f"диапазоны словаря не стыкуются: {left} и {right}")

        if edges[-1][1] != self.layout["size"]:
            raise LayoutError("последний диапазон не доходит до размера словаря")

        if len(self.value_rows) != ranges["values"][1] - ranges["values"][0]:
            raise LayoutError("число значений не совпадает с их диапазоном")

    @staticmethod
    def from_bundle(bundle: dict, path: Path | None = None) -> "FrozenArtifacts":

        keys = bundle["keys"]["rows"]

        value_ids: dict[tuple, int] = {}

        for row in bundle["values"]["rows"]:
            if row["kind"] == KIND_CATEGORICAL:
                value_ids[(row["domain"], row["value_type"], row["value"])] = row["id"]
            else:
                value_ids[(row["key"], KIND_BUCKET, row["value"])] = row["id"]

        section = bundle["bpe"]

        bpe = (
            load_bpe(section["model"])
            if section.get("enabled")
            else BpeModel(enabled=False, keys=(), info=section)
        )

        artifacts = FrozenArtifacts(
            path=path,
            bundle=bundle,
            layout=bundle["layout"],
            specials=bundle["specials"]["ids"],
            key_ids={row["key"]: row["id"] for row in keys},
            key_info={row["key"]: row for row in keys},
            domain_of={row["key"]: row["domain"] for row in keys if row["domain"]},
            value_ids=value_ids,
            value_rows=bundle["values"]["rows"],
            candidates=bundle["candidates"]["keys"],
            encoders=load_encoders(bundle["numeric"]),
            declared_by_event_type={
                event_type: tuple(item)
                for event_type, item in bundle["declared_by_event_type"].items()
            },
            link_keys=frozenset(bundle["keys"]["link_keys"]),
            profile_keys=tuple(row["key"] for row in keys if row["origin"] == "profile"),
            bpe=bpe,
        )

        artifacts.verify()

        return artifacts

    @staticmethod
    def load(path: Path | None = None) -> "FrozenArtifacts":
        """
        Словарь из data/tokenizer/tokenizer.json.
        """

        path = Path(path) if path is not None else tokenizer_path(TOKENIZER_FILE)

        if not path.exists():
            raise LayoutError(
                f"нет {path}: выполните python -m src.tokenization.run final-vocab"
            )

        return FrozenArtifacts.from_bundle(read_json(path), path)


__all__ = [
    "FrozenArtifacts",
    "LayoutError",
    "build_tokenizer",
]
