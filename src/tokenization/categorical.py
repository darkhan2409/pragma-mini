from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from src.preprocessing.artifacts import read_json, sha256_file, write_json, write_text
from src.preprocessing.semantic.keys import CATEGORICAL, NUMERIC, TEXT

from .contract import CATEGORICAL_FILE, FIT_MANIFEST_FILE, STATISTICS_DIR
from .report import render_values_md
from .scan import TYPE_BOOL, TYPE_FLOAT, TYPE_INT, TYPE_ORDER
from .schema import SemanticSchema
from .settings import DECLINED_DOMAINS, TokenizerConfig
from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


# ============================================================
# ИДЕЯ
# ============================================================
#
# Этап 2 отвечает на вопрос «какие значения бывают у этого
# смысла» и ни на какой другой. Номеров здесь нет: их выдаёт
# этап 4, когда известны все виды токенов сразу.
#
# Домен это множество значений, из которого берёт свои значения
# один или несколько ключей. По умолчанию домен у каждого ключа
# свой: одинаковое написание ничего не доказывает, и active у
# карты не то же самое, что active у обращения. Объединение
# делается только явным списком с причиной, и реестр смысла
# имеет право его запретить.
#
# Значение это пара «тип, запись». Булево, целое и строка не
# сравниваются между собой, даже когда записываются одинаково.
#
# Все наблюдавшиеся категории сохраняются целиком. Порог
# редкости существует, но по умолчанию выключен, и поиска
# «лучшего порога» здесь нет: это была бы подгонка словаря под
# данные под видом правила.
# ============================================================


STAGE = "values"

CATALOG_FILE = "value_catalog.json"
REPORT_FILE = "values_report.md"


class ValuesError(ValueError):
    """
    Каталог значений собрать нельзя.
    """


@dataclass
class ValuesResult:
    report: dict
    outputs: list[Path] = field(default_factory=list)


def typed_value(value_type: str, text: str) -> object:
    """
    Значение своего типа из его записи.
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
    Порядок значений внутри домена: по типу, затем по самому
    значению.

    Сортировка по записи поставила бы 10 перед 9, а «истина»
    между строками. Порядок должен быть свойством значения, а не
    его текста.
    """

    return (TYPE_ORDER[value_type], typed_value(value_type, text))


def _verify_inputs(target: Path, config: TokenizerConfig) -> dict:
    """
    Этап 2 читает выходы этапа 1 и обязан убедиться, что они те
    же самые.
    """

    path = target / FIT_MANIFEST_FILE

    if not path.exists():
        raise ValuesError(
            f"нет {path}: каталог значений строится поверх контракта входа, выполните contract"
        )

    manifest = read_json(path)

    if manifest.get("config_sha256") != config.sha256():
        raise ValuesError(
            "конфигурация изменилась после контракта входа: выполните contract заново, "
            "иначе словарь учился бы по одним правилам, а объявлены были бы другие"
        )

    for name, digest in sorted((manifest.get("artifacts") or {}).items()):

        file = target / name

        if not file.exists() or sha256_file(file) != digest:
            raise ValuesError(
                f"статистика {name} изменилась после контракта входа: выполните contract заново"
            )

    return manifest


def _domains(config: TokenizerConfig, schema: SemanticSchema) -> tuple[dict[str, str], dict[str, dict]]:
    """
    Ключ -> имя домена и описание каждого домена.
    """

    declared: dict[str, dict] = {}
    domain_of: dict[str, str] = {}

    for domain in config.value_domains:

        for key in domain.keys:

            info = schema.info(key)

            if info.value_kind != CATEGORICAL:
                raise ValuesError(
                    f"домен {domain.name} объединяет {key}, а он {info.value_kind}: "
                    "общий домен бывает только у категорий"
                )

            domain_of[key] = domain.name

        declared[domain.name] = {
            "name": domain.name,
            "keys": list(domain.keys),
            "shared": True,
            "reason": domain.reason,
        }

    for key in schema.categorical_keys:

        if key in domain_of:
            continue

        domain_of[key] = key
        declared[key] = {
            "name": key,
            "keys": [key],
            "shared": False,
            "reason": "домен по умолчанию: значения ключа ни с кем не делятся",
        }

    return domain_of, declared


def build_values(
    target: Path,
    processed_dir: Path,
    config: TokenizerConfig,
    group: str = "train",
) -> ValuesResult:
    """
    Каталог категориальных значений по train-наблюдениям.
    """

    target = Path(target)

    manifest = _verify_inputs(target, config)

    schema = SemanticSchema.open(processed_dir, group)

    domain_of, domains = _domains(config, schema)

    rows = pq.read_table(target / STATISTICS_DIR / CATEGORICAL_FILE).to_pylist()

    # --- значения по доменам ---

    collected: dict[str, dict[tuple[str, str], dict]] = {}
    per_key: dict[str, list[tuple[str, str]]] = {}

    for row in rows:

        key = row["key"]

        domain = domain_of.get(key)

        if domain is None:
            raise ValuesError(
                f"ключ {key} встретился в статистике, но категорией не объявлен: "
                "статистика и реестр разошлись"
            )

        slot = collected.setdefault(domain, {})
        item = (row["value_type"], row["value"])

        entry = slot.get(item)

        if entry is None:
            entry = {"value_type": row["value_type"], "value": row["value"], "count": 0, "keys": []}
            slot[item] = entry

        entry["count"] += row["count"]
        entry["keys"].append(key)

        per_key.setdefault(key, []).append(item)

    rare_total = 0

    for name, slot in collected.items():

        ordered = sorted(slot.values(), key=lambda entry: sort_key(entry["value_type"], entry["value"]))

        for entry in ordered:
            entry["keys"] = sorted(set(entry["keys"]))
            entry["rare"] = (
                config.rare_min_count is not None and entry["count"] < config.rare_min_count
            )
            rare_total += int(entry["rare"])

        domains[name]["values"] = ordered

    for name, domain in domains.items():
        domain.setdefault("values", [])

    # --- ключи ---

    key_rows: list[dict] = []

    for key in sorted(schema.keys):

        info = schema.info(key)

        if not info.is_model_feature:
            continue

        observed = per_key.get(key, [])

        key_rows.append(
            {
                "key": key,
                "value_kind": info.value_kind,
                "unit": info.unit,
                "origin": info.origin,
                "weight_rule": info.weight_rule,
                "domain": domain_of.get(key),
                "physical_fields": list(info.physical_fields),
                "observed_in_train": bool(observed) or key in _observed_non_categorical(manifest),
                "values": len(observed),
                "candidates": [
                    {"value_type": kind, "value": text}
                    for kind, text in sorted(set(observed), key=lambda item: sort_key(*item))
                ],
            }
        )

    unobserved = sorted(row["key"] for row in key_rows if not row["observed_in_train"])

    report = {
        "stage": STAGE,
        "schema_version": SCHEMA_VERSION,
        "format_version": FORMAT_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "group": group,
        "fit_content_sha256": manifest["fit_content_sha256"],
        "config_sha256": config.sha256(),
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
            "missing": "объявленный у типа события ключ без значения кодируется парой ключ/[MISSING]; "
                       "расчётный ключ пары не получает, его отсутствие объясняет причина",
            "digits": "цифровые коды это категории: MCC, версия продукта и версия тарифа величинами не являются",
        },
        "counts": {
            "keys_model_feature": len(key_rows),
            "categorical": len(schema.categorical_keys),
            "numeric": len(schema.numeric_keys),
            "text": len(schema.text_keys),
            "link": len(schema.link_keys),
            "domains": len(domains),
            "domains_shared": sum(1 for item in domains.values() if item["shared"]),
            "values": sum(len(item["values"]) for item in domains.values()),
            "rare": rare_total,
            "observed_in_train": sum(1 for row in key_rows if row["observed_in_train"]),
            "unobserved_in_train": len(unobserved),
        },
        "domains": [domains[name] for name in sorted(domains)],
        "keys": key_rows,
        "unobserved_in_train": unobserved,
        "declined_domains": [{"keys": list(keys), "reason": reason} for keys, reason in DECLINED_DOMAINS],
        "ambiguous": [{"keys": list(keys), "reason": reason} for keys, reason in schema.ambiguous],
        "allowed_sharing": [
            {"key": key, "sources": list(sources), "reason": reason}
            for key, sources, reason in schema.allowed_sharing
        ],
        "declared_by_event_type": {
            event_type: list(keys) for event_type, keys in schema.declared_payload.items()
        },
    }

    outputs: list[Path] = []

    path = target / CATALOG_FILE
    write_json(path, report)
    outputs.append(path)

    path = target / REPORT_FILE
    write_text(path, render_values_md(report))
    outputs.append(path)

    return ValuesResult(report=report, outputs=outputs)


def _observed_non_categorical(manifest: dict) -> set[str]:
    """
    Ключи, которые наблюдались на train, по данным контракта
    входа: числа и тексты в каталоге категорий не лежат.
    """

    return {row["key"] for row in manifest["keys"]["rows"] if row["observed_in_train"]}


__all__ = ["CATALOG_FILE", "REPORT_FILE", "STAGE", "ValuesError", "ValuesResult", "build_values",
           "sort_key", "typed_value"]
