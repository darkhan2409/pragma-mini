from __future__ import annotations

from pathlib import Path

from src.preprocessing.artifacts import read_json

from .fit import TrainCorpus
from .keyvocab import domains_of
from .scan import TYPE_BOOL, TYPE_FLOAT, TYPE_INT, TYPE_ORDER
from .schema import SemanticSchema
from .settings import DECLINED_DOMAINS, VALUE_VOCAB_FILE, TokenizerConfig, tokenizer_path
from .specials import KIND_CATEGORICAL
from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


# ============================================================
# ЭТАП 3: КАТЕГОРИАЛЬНЫЕ ЗНАЧЕНИЯ
# ============================================================
#
# Здесь лежит ответ на вопрос «какие значения бывают у этого
# смысла» и ни на какой другой: чисел и кусков текста в этом
# словаре нет.
#
# Домен это множество значений, из которого берёт свои значения
# один или несколько ключей. По умолчанию домен у каждого ключа
# свой; объединение делается явным списком с причиной, и реестр
# смысла имеет право его запретить.
#
# Значение это пара «тип, запись». Булево, целое и строка не
# сравниваются между собой, даже когда записываются одинаково.
#
# Все наблюдавшиеся на train категории сохраняются целиком.
# Порог редкости существует, но по умолчанию выключен, и поиска
# «лучшего порога» здесь нет: это была бы подгонка словаря под
# данные под видом правила.
#
# Значения, которых на train не было, словаря не получают
# никогда: на val и test они кодируются как [UNK].
# ============================================================


class ValuesError(ValueError):
    """
    Каталог значений собрать или прочитать нельзя.
    """


def typed_value(value_type: str, text: str) -> object:
    """
    Значение в своём типе: по нему и сортируется домен.
    """

    if value_type == TYPE_BOOL:
        return text == "true"

    if value_type == TYPE_INT:
        return int(text)

    if value_type == TYPE_FLOAT:
        return float(text)

    return text


def sort_key(value_type: str, text: str) -> tuple:
    """
    Порядок значений внутри домена: сначала тип, потом само
    значение.

    Числа сортируются как числа: иначе 10 оказалось бы между 1 и
    2, и соседние ID достались бы далёким значениям.
    """

    return (TYPE_ORDER[value_type], typed_value(value_type, text))


def build_value_vocab(
    train: TrainCorpus,
    key_vocab: dict,
    config: TokenizerConfig,
    schema: SemanticSchema,
) -> dict:
    """
    Категориальные значения train, их ID, частоты и связь с
    ключом.
    """

    domain_of, domains = domains_of(config, schema)

    stats = train.statistics

    collected: dict[str, dict[tuple[str, str], dict]] = {}
    per_key: dict[str, list[tuple[str, str]]] = {}

    for (key, value_type, value), entry in sorted(stats.categorical.items()):

        domain = domain_of.get(key)

        if domain is None:
            raise ValuesError(
                f"ключ {key} встретился в статистике, но категорией не объявлен: "
                "статистика и реестр разошлись"
            )

        slot = collected.setdefault(domain, {})
        item = (value_type, value)

        row = slot.get(item)

        if row is None:
            row = {"value_type": value_type, "value": value, "count": 0, "clients": 0, "keys": []}
            slot[item] = row

        row["count"] += entry.count
        row["clients"] += entry.clients
        row["keys"].append(key)

        per_key.setdefault(key, []).append(item)

    # --- номера ---

    first_value_id = int(key_vocab["next_id"])

    rows: list[dict] = []
    rare_total = 0

    for name in sorted(domains):

        ordered = sorted(
            collected.get(name, {}).values(),
            key=lambda item: sort_key(item["value_type"], item["value"]),
        )

        ids: list[int] = []

        for item in ordered:

            item["keys"] = sorted(set(item["keys"]))
            item["rare"] = config.rare_min_count is not None and item["count"] < config.rare_min_count
            rare_total += int(item["rare"])

            row = {
                "id": first_value_id + len(rows),
                "kind": KIND_CATEGORICAL,
                "domain": name,
                "key": None,
                "value_type": item["value_type"],
                "value": item["value"],
                "label": f"{name}={item['value']}",
                "count": item["count"],
                "clients": item["clients"],
                "rare": item["rare"],
                "keys": item["keys"],
            }

            rows.append(row)
            ids.append(row["id"])

        domains[name]["values"] = len(ids)
        domains[name]["value_ids"] = ids

    # --- связь с ключом ---

    index_of = {(row["domain"], row["value_type"], row["value"]): row["id"] for row in rows}

    by_key: dict[str, list[int]] = {}

    for row in key_vocab["keys"]:

        if row["value_kind"] != KIND_CATEGORICAL:
            continue

        key = row["key"]
        domain = row["domain"]

        observed = sorted(set(per_key.get(key, [])), key=lambda item: sort_key(*item))

        by_key[key] = [index_of[(domain, value_type, value)] for value_type, value in observed]

    unobserved = sorted(key for key, ids in by_key.items() if not ids)

    return {
        "schema_version": SCHEMA_VERSION,
        "format_version": FORMAT_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "fit": train.as_dict(),
        "config_sha256": config.sha256(),
        "first_value_id": first_value_id,
        "size": len(rows),
        "next_id": first_value_id + len(rows),
        "rules": {
            "identity": "значение это пара «тип, запись»: true, 1 и \"1\" разными не выглядят, "
                        "но разными являются",
            "domain": "по умолчанию домен у ключа свой; объединение делается явным списком с причиной "
                      "и запрещено там, где смысловой реестр объявил ключи несовместимыми",
            "order": "значения домена упорядочены по типу и по самому значению, а не по его записи",
            "rare": (
                "все наблюдавшиеся категории сохраняются целиком"
                if config.rare_min_count is None
                else f"значение реже {config.rare_min_count} наблюдений помечено редким"
            ),
            "unknown": "UNK это категория, которой на train не было; RARE это известная train-категория, "
                       "объединённая по правилу редкости. Это разные вещи",
            "digits": "цифровые коды это категории: MCC, версия продукта и версия тарифа величинами не являются",
        },
        "counts": {
            "domains": len(domains),
            "domains_shared": sum(1 for item in domains.values() if item["shared"]),
            "values": len(rows),
            "rare": rare_total,
            "keys": len(by_key),
            "unobserved_in_train": len(unobserved),
        },
        "domains": [domains[name] for name in sorted(domains)],
        "values": rows,
        "by_key": by_key,
        "unobserved_in_train": unobserved,
        "declined_domains": [{"keys": list(keys), "reason": reason} for keys, reason in DECLINED_DOMAINS],
        "ambiguous": [{"keys": list(keys), "reason": reason} for keys, reason in schema.ambiguous],
    }


def load_value_vocab(directory: Path | None = None) -> dict:
    """
    Каталог значений предыдущего этапа.
    """

    path = (Path(directory) / VALUE_VOCAB_FILE) if directory else tokenizer_path(VALUE_VOCAB_FILE)

    if not path.exists():
        raise ValuesError(f"нет {path}: выполните python -m src.tokenization.run value-vocab")

    return read_json(path)


__all__ = [
    "ValuesError",
    "build_value_vocab",
    "load_value_vocab",
    "sort_key",
    "typed_value",
]
