from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa

from src.preprocessing.artifacts import read_json, sha256_file, write_json, write_table, write_text
from src.preprocessing.manifest import MANIFEST_FILE
from src.preprocessing.projection import PROJECTION_VERSION
from src.preprocessing.semantic.as_of import SEMANTIC_VERSION
from src.preprocessing.semantic.keys import KEYS_VERSION

from .corpus import FitCorpus
from .report import render_compatibility_md, render_contract_md
from .scan import FitStatistics, scan
from .schema import SemanticSchema
from .settings import METHOD_UNFITTED, TokenizerConfig
from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


# ============================================================
# ИДЕЯ
# ============================================================
#
# Этап 1 отвечает на два вопроса и ни на один больше.
#
#   Что токенизатору вообще разрешено читать?
#   На чём из этого ему разрешено учиться?
#
# Словарей здесь нет. Есть список ключей с их видом, единицей и
# правилом веса, один проход статистики по разрешённому корпусу
# и список противоречий, которые обязан решить человек.
#
# Противоречие не исправляется тихо. Реестр смысла ведёт
# препроцессинг; если он называет структурированный код
# свободным текстом, токенизатор следует реестру и громко
# сообщает об этом, а не переклассифицирует поле у себя внутри.
# ============================================================


STAGE = "contract"

CONFIG_FILE = "tokenizer_config.json"
FIT_MANIFEST_FILE = "fit_corpus_manifest.json"
CONTRACT_REPORT_FILE = "contract_report.md"
COMPATIBILITY_REPORT_FILE = "compatibility_report.md"

STATISTICS_DIR = "fit_statistics"
CATEGORICAL_FILE = "categorical.parquet"
NUMERIC_SAMPLES_FILE = "numeric_samples.parquet"
NUMERIC_SUMMARY_FILE = "numeric_summary.json"
TEXT_FILE = "text.parquet"
MISSING_FILE = "missing.parquet"

# Текст, похожий на код: короткое множество значений, каждое из
# букв, цифр и подчёркиваний, и каждое встречается не раз.
CODE_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
CODE_MAX_DISTINCT = 32
CODE_MIN_REPEAT = 2.0

# Идентификатор внутри текста: четыре цифры подряд и больше.
IDENTIFIER_PATTERN = re.compile(r"\d{4,}")
IDENTIFIER_SHARE = 0.05


CATEGORICAL_SCHEMA = pa.schema(
    [
        ("key", pa.string()),
        ("value_type", pa.string()),
        ("value", pa.string()),
        ("count", pa.int64()),
        ("clients", pa.int64()),
    ]
)

NUMERIC_SAMPLES_SCHEMA = pa.schema([("key", pa.string()), ("value", pa.float64())])

TEXT_SCHEMA = pa.schema(
    [
        ("key", pa.string()),
        ("text_norm", pa.string()),
        ("count", pa.int64()),
        ("clients", pa.int64()),
        ("max_bytes", pa.int64()),
        ("example_raw", pa.string()),
    ]
)

MISSING_SCHEMA = pa.schema(
    [
        ("event_type", pa.string()),
        ("key", pa.string()),
        ("missing_events", pa.int64()),
        ("events_of_type", pa.int64()),
    ]
)


class ContractError(ValueError):
    """
    Контракт входа не выполняется: учиться на таком корпусе
    нельзя.
    """


@dataclass
class ContractResult:
    report: dict
    statistics: FitStatistics
    outputs: list[Path] = field(default_factory=list)


# ------------------------------------------------------------
# ПРОВЕРКИ
# ------------------------------------------------------------


def _check_versions(schema: SemanticSchema) -> None:
    """
    Реестр и код обязаны быть одной версии.

    Иначе fit пошёл бы по одному описанию смысла, а значения
    пришли бы по другому, и разошлись бы они молча.
    """

    pairs = (
        ("keys_version", schema.keys_version, KEYS_VERSION),
        ("semantic_version", schema.semantic_version, SEMANTIC_VERSION),
        ("projection_version", schema.projection_version, PROJECTION_VERSION),
    )

    problems = [
        f"{name}: в реестре {stored}, в коде {actual}"
        for name, stored, actual in pairs
        if stored != actual
    ]

    if problems:
        raise ContractError(
            "смысловой реестр собран другой версией кода (" + "; ".join(problems) + "): "
            "выполните этап semantic заново"
        )


def _check_config(config: TokenizerConfig, schema: SemanticSchema) -> None:
    """
    Конфигурация обязана описывать ровно те ключи, что есть в
    реестре.
    """

    config.validate()

    numeric = set(schema.numeric_keys)
    declared = set(config.numeric_encoders)

    missing = sorted(numeric - declared)

    if missing:
        raise ContractError(
            "числовые ключи без кодировщика: " + ", ".join(missing) + ". "
            "Каждое число обязано иметь объявленный способ кодирования"
        )

    extra = sorted(declared - numeric)

    if extra:
        raise ContractError(
            "кодировщики объявлены для ключей, которых нет среди числовых: " + ", ".join(extra)
        )

    unknown_domain_keys = sorted(
        key for domain in config.value_domains for key in domain.keys if key not in schema.keys
    )

    if unknown_domain_keys:
        raise ContractError("домены значений ссылаются на неизвестные ключи: " + ", ".join(unknown_domain_keys))

    for domain in config.value_domains:

        kinds = {schema.info(key).value_kind for key in domain.keys}

        if len(kinds) > 1:
            raise ContractError(f"домен {domain.name} объединяет ключи разных видов значения: {sorted(kinds)}")

        for group, reason in schema.ambiguous:
            shared = sorted(set(domain.keys) & set(group))
            if len(shared) > 1:
                raise ContractError(
                    f"домен {domain.name} объединяет ключи, объявленные несовместимыми в реестре "
                    f"({', '.join(shared)}): {reason}"
                )

    unknown_overrides = sorted(set(config.text_keys_as_categorical) - set(schema.text_keys))

    if unknown_overrides:
        raise ContractError(
            "text_keys_as_categorical называет ключи, которые реестр текстом не объявлял: "
            + ", ".join(unknown_overrides)
        )


def _check_scan(stats: FitStatistics, corpus: FitCorpus, schema: SemanticSchema) -> None:
    """
    Прочитанное обязано совпасть с тем, что объявило разделение.
    """

    if stats.unknown_keys:
        raise ContractError(
            "в значениях встретились ключи вне смыслового реестра: "
            + ", ".join(sorted(stats.unknown_keys))
            + ". Смысл обязан быть объявлен заранее"
        )

    if stats.events != corpus.declared_rows:
        raise ContractError(
            f"событий прочитано {stats.events}, а разделение объявило {corpus.declared_rows}: "
            "корпус и разделение разошлись, выполните этап split заново"
        )

    if stats.clients != len(corpus.client_ids):
        raise ContractError(
            f"клиентов прочитано {stats.clients}, а разрешено {len(corpus.client_ids)}: "
            "молчащий клиент из группы не убирается, его история просто пуста"
        )

    if stats.clients_without_profile != corpus.declared_clients_without_profile:
        raise ContractError(
            f"клиентов без профиля прочитано {stats.clients_without_profile}, а разделение объявило "
            f"{corpus.declared_clients_without_profile}: профиль на fit_end читается не так, как при разделении"
        )

    if stats.unknown_event_types:
        raise ContractError(
            "типы событий без объявленных полей: " + ", ".join(sorted(stats.unknown_event_types))
            + ". Реестр полей собран не на этих данных"
        )

    conflicts = {
        key: sorted(types)
        for key, types in sorted(stats.key_types.items())
        if len(types) > 1 and schema.info(key).value_kind != "numeric"
    }

    if conflicts:
        raise ContractError(
            "у ключа больше одного физического типа значения: "
            + "; ".join(f"{key}: {types}" for key, types in conflicts.items())
            + ". Объединять их молча нельзя: true и \"true\" это разные значения"
        )


# ------------------------------------------------------------
# ПРОТИВОРЕЧИЯ
# ------------------------------------------------------------


def _text_contradictions(stats: FitStatistics, schema: SemanticSchema) -> list[dict]:
    """
    Текстовые ключи, которые на train ведут себя не как текст.

    Решение принимает человек: токенизатор следует реестру и
    только называет несоответствие.
    """

    out: list[dict] = []

    for key in schema.text_keys:

        bucket = stats.text.get(key)

        if not bucket:
            continue

        distinct = len(bucket)
        total = sum(entry.count for entry in bucket.values())
        code_like = sum(1 for text in bucket if CODE_PATTERN.match(text))
        with_identifier = sum(entry.count for text, entry in bucket.items() if IDENTIFIER_PATTERN.search(text))

        examples = sorted(bucket)[:5]

        if distinct <= CODE_MAX_DISTINCT and code_like == distinct and total / distinct >= CODE_MIN_REPEAT:
            out.append(
                {
                    "key": key,
                    "kind": "text_looks_like_code",
                    "detail": (
                        f"реестр объявил текст, но на train у ключа {distinct} различных значений "
                        f"на {total} наблюдений, и все они выглядят кодом"
                    ),
                    "examples": examples,
                    "proposal": (
                        "перевести ключ в categorical в semantic/keys.py и пересобрать этап 5 "
                        "либо подтвердить, что это свободный текст"
                    ),
                }
            )

        if total and with_identifier / total > IDENTIFIER_SHARE:
            out.append(
                {
                    "key": key,
                    "kind": "text_contains_identifiers",
                    "detail": (
                        f"в {with_identifier} наблюдениях из {total} внутри текста стоит число "
                        "из четырёх и более цифр: BPE будет резать идентификаторы"
                    ),
                    "examples": [text for text in sorted(bucket) if IDENTIFIER_PATTERN.search(text)][:5],
                    "proposal": "признать это свойством данных и записать в ограничения либо изменить смысл поля",
                }
            )

    return out


def _unit_findings(schema: SemanticSchema, config: TokenizerConfig) -> tuple[list[dict], list[dict]]:
    """
    Числовые ключи, у которых единица это не единица.

    Возвращает две разные вещи: открытые противоречия, ждущие
    решения человека, и уже принятые ограничения V1. Смешивать
    их нельзя — иначе решённое годами выглядит нерешённым, а
    настоящий вопрос тонет среди записей.
    """

    open_items: list[dict] = []
    accepted: list[dict] = []

    for key in schema.numeric_keys:

        info = schema.info(key)

        if info.unit is None:
            open_items.append(
                {
                    "key": key,
                    "kind": "numeric_without_unit",
                    "detail": "числовой ключ без объявленной единицы: сравнивать его не с чем",
                    "examples": [],
                    "proposal": "объявить единицу в canonical/registry.py",
                }
            )
            continue

        if info.unit not in schema.keys:
            continue

        detail = (
            f"единица значения лежит в соседнем ключе {info.unit}: одной шкалы на разные единицы "
            "быть не может"
        )

        if config.numeric_encoders[key].method == METHOD_UNFITTED:
            accepted.append(
                {
                    "key": key,
                    "kind": "unit_is_another_key",
                    "detail": detail,
                    "decision": (
                        f"в V1 шкалы у ключа нет: значение получает числовое [UNK] и попадает в "
                        f"счётчики. Денежный масштаб операции уже представлен ключом "
                        f"transaction_amount в тенге, а валюта — ключом {info.unit}; строить "
                        "валютные диапазоны по нескольким наблюдениям train было бы выдумыванием "
                        "шкалы, а не её измерением"
                    ),
                }
            )
        else:
            open_items.append(
                {
                    "key": key,
                    "kind": "unit_is_another_key",
                    "detail": detail,
                    "examples": [],
                    "proposal": "объявить кодировщик unfitted либо привести значения к одной единице",
                }
            )

    return open_items, accepted


# ------------------------------------------------------------
# АРТЕФАКТЫ
# ------------------------------------------------------------


def _write_statistics(target: Path, stats: FitStatistics, schema: SemanticSchema) -> list[Path]:

    outputs: list[Path] = []

    directory = target / STATISTICS_DIR

    rows = [
        {"key": key, "value_type": kind, "value": text, "count": entry.count, "clients": entry.clients}
        for (key, kind, text), entry in sorted(stats.categorical.items())
    ]

    path = directory / CATEGORICAL_FILE
    write_table(path, pa.Table.from_pylist(rows, schema=CATEGORICAL_SCHEMA), CATEGORICAL_SCHEMA)
    outputs.append(path)

    samples = [
        {"key": key, "value": value}
        for key in sorted(stats.numeric)
        for value in stats.numeric[key].values()
    ]

    path = directory / NUMERIC_SAMPLES_FILE
    write_table(path, pa.Table.from_pylist(samples, schema=NUMERIC_SAMPLES_SCHEMA), NUMERIC_SAMPLES_SCHEMA)
    outputs.append(path)

    summary = {
        key: {
            **stats.numeric[key].summary(),
            "unit": schema.info(key).unit,
            "weight_rule": schema.info(key).weight_rule,
        }
        for key in sorted(stats.numeric)
    }

    path = directory / NUMERIC_SUMMARY_FILE
    write_json(path, summary)
    outputs.append(path)

    rows = [
        {
            "key": key,
            "text_norm": text,
            "count": entry.count,
            "clients": entry.clients,
            "max_bytes": entry.max_bytes,
            "example_raw": entry.example_raw,
        }
        for key in sorted(stats.text)
        for text, entry in sorted(stats.text[key].items())
    ]

    path = directory / TEXT_FILE
    write_table(path, pa.Table.from_pylist(rows, schema=TEXT_SCHEMA), TEXT_SCHEMA)
    outputs.append(path)

    rows = [
        {
            "event_type": event_type,
            "key": key,
            "missing_events": count,
            "events_of_type": stats.event_types.get(event_type, 0),
        }
        for (event_type, key), count in sorted(stats.missing.items())
    ]

    path = directory / MISSING_FILE
    write_table(path, pa.Table.from_pylist(rows, schema=MISSING_SCHEMA), MISSING_SCHEMA)
    outputs.append(path)

    return outputs


def _stage_versions(processed_dir: Path, group: str) -> dict:

    manifest = read_json(Path(processed_dir) / MANIFEST_FILE)

    stages = manifest.get("stages", {})

    out: dict[str, str | None] = {}

    for stage in ("passport", "canonical", "history", "semantic"):
        entry = stages.get(stage, {}).get(group, {})
        out[stage] = entry.get("stage_version")

    out["split"] = stages.get("split", {}).get("stage_version")

    return out


def _key_rows(schema: SemanticSchema, stats: FitStatistics, config: TokenizerConfig) -> list[dict]:

    rows: list[dict] = []

    for key in sorted(schema.keys):

        info = schema.info(key)
        counter = stats.key_counts.get(key)

        rows.append(
            {
                "key": key,
                "value_kind": info.value_kind,
                "unit": info.unit,
                "origin": info.origin,
                "weight_rule": info.weight_rule,
                "role": "model_feature" if info.is_model_feature else "link",
                "observed_in_train": counter is not None or key in stats.reference_values,
                "observations": (counter.events if counter else stats.reference_values.get(key, 0)),
                "clients": counter.clients if counter else 0,
                "encoder": (
                    config.numeric_encoders[key].method if key in config.numeric_encoders else None
                ),
            }
        )

    return rows


def build_contract(
    processed_dir: Path,
    raw_dir: Path,
    target: Path,
    config: TokenizerConfig,
    group: str = "train",
    allow_short_horizon: bool = False,
) -> ContractResult:
    """
    Собирает контракт входа и статистику разрешённого корпуса.
    """

    corpus = FitCorpus.open(processed_dir, raw_dir, group=group, allow_short_horizon=allow_short_horizon)

    schema = corpus.schema

    _check_versions(schema)
    _check_config(config, schema)

    stats = scan(
        corpus.iter_histories(),
        schema,
        sample_k=config.quantile_sample_k,
        distinct_cap=config.distinct_cap,
    )

    _check_scan(stats, corpus, schema)

    unit_open, accepted_limits = _unit_findings(schema, config)

    contradictions = _text_contradictions(stats, schema) + unit_open

    limitations = [
        *[f"{item['key']}: {item['decision']}" for item in accepted_limits],
        "справочник мерчантов не версионирован во времени: атрибуты точки берутся такими, "
        "какие они на момент выгрузки, и исключить будущее знание о точке нельзя",
        *[f"разделение: {item}" for item in corpus.split_limitations],
        *[f"история клиента: {item}" for item in sorted(stats.limitations)],
    ]

    unobserved = sorted(
        key
        for key, info in schema.keys.items()
        if info.is_model_feature and key not in stats.key_counts
    )

    report = {
        "stage": STAGE,
        "schema_version": SCHEMA_VERSION,
        "format_version": FORMAT_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "group": group,
        "fit_end": corpus.fit_end.isoformat(),
        "readiness": corpus.readiness.as_dict(),
        "corpus": {
            "clients": stats.clients,
            "events": stats.events,
            "values": stats.values,
            "profiles_as_of": stats.profile_versions,
            "clients_without_profile": stats.clients_without_profile,
            "day_precision_events": stats.day_precision_events,
            "event_types": dict(sorted(stats.event_types.items())),
        },
        "declared_by_split": {
            "events_rows": corpus.declared_rows,
            "source_profile_version_rows": corpus.declared_profile_rows,
            "clients_without_profile": corpus.declared_clients_without_profile,
            "content_sha256": corpus.content_sha256,
            "rule": (
                "source_profile_version_rows это все версии профиля, известные к fit_end; fit видит по одной "
                "действующей версии на клиента, а прежние значения приходят событиями изменения профиля"
            ),
        },
        "keys": {
            "declared": len(schema.keys),
            "model_feature": len(schema.model_feature_keys),
            "link": len(schema.link_keys),
            "by_kind": {
                "numeric": len(schema.numeric_keys),
                "categorical": len(schema.categorical_keys),
                "text": len(schema.text_keys),
            },
            "observed_in_train": len(stats.key_counts),
            "unobserved_in_train": unobserved,
            "rows": _key_rows(schema, stats, config),
        },
        "missing": {
            "rule": (
                "ключ, объявленный у этого типа события в реестре полей, но без значения, "
                "кодируется парой ключ/[MISSING]; расчётные и справочные ключи пары не получают, "
                "их отсутствие объясняет причина"
            ),
            "pairs": len(stats.missing),
            "events_with_declared_keys": sum(stats.event_types.values()),
            "top": [
                {"event_type": event_type, "key": key, "missing_events": count}
                for (event_type, key), count in sorted(
                    stats.missing.items(), key=lambda item: (-item[1], item[0])
                )[:20]
            ],
        },
        "absent_reasons": {key: dict(sorted(value.items())) for key, value in sorted(stats.absent_reasons.items())},
        "references": dict(sorted(stats.reference_values.items())),
        "text_empty": dict(sorted(stats.text_empty.items())),
        "contradictions": contradictions,
        "accepted_limits": accepted_limits,
        "fit_content_sha256": stats.content.value(),
        "input_files": corpus.inputs,
        "versions": {
            "keys": schema.keys_version,
            "semantic": schema.semantic_version,
            "projection": schema.projection_version,
            "stages": _stage_versions(processed_dir, group),
        },
        "config_sha256": config.sha256(),
        "limitations": limitations,
    }

    target = Path(target)

    outputs: list[Path] = []

    path = target / CONFIG_FILE
    write_json(path, config.as_dict())
    outputs.append(path)

    outputs.extend(_write_statistics(target, stats, schema))

    # Отпечатки того, на чём будут учиться следующие этапы.
    # Манифест пишется последним из этой группы: собственного
    # отпечатка в нём нет и быть не может.
    report["artifacts"] = {
        item.relative_to(target).as_posix(): sha256_file(item) for item in outputs
    }

    path = target / FIT_MANIFEST_FILE
    write_json(path, report)
    outputs.append(path)

    path = target / CONTRACT_REPORT_FILE
    write_text(path, render_contract_md(report))
    outputs.append(path)

    path = target / COMPATIBILITY_REPORT_FILE
    write_text(path, render_compatibility_md(report))
    outputs.append(path)

    return ContractResult(report=report, statistics=stats, outputs=outputs)


__all__ = [
    "CATEGORICAL_FILE",
    "COMPATIBILITY_REPORT_FILE",
    "CONFIG_FILE",
    "CONTRACT_REPORT_FILE",
    "FIT_MANIFEST_FILE",
    "MISSING_FILE",
    "NUMERIC_SAMPLES_FILE",
    "NUMERIC_SUMMARY_FILE",
    "STAGE",
    "STATISTICS_DIR",
    "TEXT_FILE",
    "ContractError",
    "ContractResult",
    "build_contract",
]
