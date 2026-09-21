from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa

from src.preprocessing.artifacts import write_json, write_table, write_text
from src.preprocessing.history import HistoryError
from src.preprocessing.semantic.chains import ChainsError
from src.preprocessing.semantic.keys import KeysError
from src.tokenization.layout import EMPTY, INVALID, MISSING, UNK

from .collate import collate
from .context import ContextError
from .dependencies import DEP_STATUSES, NOT_TRACKED, RELATION_FEATURES, DependencyError
from .encoding import EncodingError, causes_as_of, encode_history
from .inputs import DatasetInputs
from .report import render_golden_md, render_report_md
from .sample import Sample, SampleError, build_sample
from .settings import DatasetConfig
from .storage import (
    GOLDEN_JSON_FILE,
    GOLDEN_MD_FILE,
    INDEX_FILE,
    INDEX_SCHEMA,
    MASKER_COLUMNS,
    MODEL_COLUMNS,
    REPORT_JSON_FILE,
    REPORT_MD_FILE,
    SERVICE_COLUMNS,
    ShardWriter,
    StorageError,
    prepare_build,
    publish,
    write_manifest,
)
from .targets import COVERAGE_STATES
from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


# ============================================================
# ИДЕЯ
# ============================================================
#
# Сборка идёт потоком: группа, срез, клиент. В памяти живёт один
# пример, а не набор.
#
# Каждая пара «клиент, срез» проходит полный путь заново — от
# смысловой истории на эту дату до токенов. Ранний срез никогда
# не получается отсечением позднего: там уже лежат исправления,
# которых на раннюю дату не существовало.
#
# Отдельное правило у оценочных групп: потерять там возможную
# цель нельзя. Если отбор контекста выбрасывает событие периода
# целей, оценка начинает зависеть от политики усечения, и
# сравнивать две модели становится нечем. Сборка в этом случае
# отказывает, а не пишет тихо усечённый набор.
# ============================================================


class BuildError(ValueError):
    """
    Набор собрать нельзя.
    """


FAILURES = (
    ChainsError,
    ContextError,
    DependencyError,
    EncodingError,
    HistoryError,
    KeysError,
    SampleError,
    StorageError,
)


@dataclass
class Counters:
    samples: int = 0
    clients: set[str] = field(default_factory=set)
    events: int = 0
    tokens: int = 0
    values: int = 0
    profile_tokens: int = 0
    eligible_events: int = 0
    with_targets: int = 0
    empty_history: int = 0
    without_profile: int = 0
    truncated: int = 0
    excluded_events: int = 0
    excluded_tokens: int = 0
    excluded_eligible: int = 0
    excluded_milestones: int = 0
    excluded_by_type: dict[str, int] = field(default_factory=dict)
    specials: dict[str, int] = field(default_factory=dict)
    dependencies: dict[str, int] = field(default_factory=dict)
    profile_states: dict[str, int] = field(default_factory=dict)
    event_counts: list[int] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)
    visible_events: list[int] = field(default_factory=list)

    def add(self, sample: Sample, visible: int, specials: dict[str, int]) -> None:

        self.samples += 1
        self.clients.add(sample.client_id)
        self.events += sample.n_events
        self.tokens += sample.n_tokens
        self.values += sample.n_values
        self.profile_tokens += sample.profile_tokens
        self.eligible_events += sample.n_eligible_events
        self.with_targets += int(sample.has_targets)
        self.empty_history += int(sample.n_events == 0)
        self.without_profile += int(not sample.has_profile)
        self.truncated += int(sample.truncated)

        self.excluded_events += int(sample.selection.get("excluded_events", 0))
        self.excluded_tokens += int(sample.selection.get("excluded_tokens", 0))
        self.excluded_eligible += int(sample.selection.get("excluded_eligible", 0))
        self.excluded_milestones += int(sample.selection.get("excluded_milestones", 0))

        for name, count in (sample.selection.get("excluded_by_type") or {}).items():
            self.excluded_by_type[name] = self.excluded_by_type.get(name, 0) + count

        for name, count in specials.items():
            self.specials[name] = self.specials.get(name, 0) + count

        for name, count in sample.dependencies.counts().items():
            self.dependencies[name] = self.dependencies.get(name, 0) + count

        state = sample.profile_state or "—"
        self.profile_states[state] = self.profile_states.get(state, 0) + 1

        self.event_counts.append(sample.n_events)
        self.token_counts.append(sample.n_tokens)
        self.visible_events.append(visible)

    def as_dict(self) -> dict:
        return {
            "samples": self.samples,
            "clients": len(self.clients),
            "events": self.events,
            "tokens": self.tokens,
            "values": self.values,
            "profile_tokens": self.profile_tokens,
            "eligible_events": self.eligible_events,
            "samples_with_targets": self.with_targets,
            "samples_without_targets": self.samples - self.with_targets,
            "empty_history": self.empty_history,
            "without_profile": self.without_profile,
            "truncated": self.truncated,
            "excluded_events": self.excluded_events,
            "excluded_tokens": self.excluded_tokens,
            "excluded_eligible": self.excluded_eligible,
            "excluded_milestones": self.excluded_milestones,
            "excluded_by_type": dict(sorted(self.excluded_by_type.items())),
            "specials": dict(sorted(self.specials.items())),
            "dependencies": dict(sorted(self.dependencies.items())),
            "profile_states": dict(sorted(self.profile_states.items())),
            "lengths": {
                "events_per_sample": _lengths(self.event_counts),
                "tokens_per_sample": _lengths(self.token_counts),
                "visible_events_per_sample": _lengths(self.visible_events),
            },
        }


@dataclass
class BuildResult:
    directory: Path
    dataset_id: str
    manifest: dict
    report: dict


def _lengths(values: list[int]) -> dict:
    """
    Распределение длин: медиана и хвосты.

    Квантили берутся по ближайшему рангу целыми числами: длина
    истории это счёт, а не измерение, и дробная медиана здесь
    смысла не имеет.
    """

    if not values:
        return {"count": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0, "min": 0}

    ordered = sorted(values)

    def rank(share: float) -> int:
        position = max(0, min(len(ordered) - 1, int(share * len(ordered)) - (1 if share == 1.0 else 0)))
        return ordered[position]

    return {
        "count": len(ordered),
        "p50": rank(0.50),
        "p90": rank(0.90),
        "p95": rank(0.95),
        "p99": rank(0.99),
        "max": ordered[-1],
        "min": ordered[0],
    }


def _specials(artifacts, sample: Sample) -> dict[str, int]:

    names = {
        artifacts.special(MISSING): "missing",
        artifacts.special(UNK): "unknown",
        artifacts.special(INVALID): "invalid",
        artifacts.special(EMPTY): "empty",
    }

    out = {name: 0 for name in names.values()}

    for values in (sample.value_ids, sample.profile_value_ids):

        if not values.size:
            continue

        unique, counts = np.unique(values, return_counts=True)

        for value, count in zip(unique.tolist(), counts.tolist()):
            name = names.get(value)
            if name is not None:
                out[name] += count

    return out


def build_dataset(inputs: DatasetInputs, config: DatasetConfig, root: Path,
                  force: bool = False) -> BuildResult:
    """
    Собирает набор целиком и публикует его одним движением.
    """

    dataset_id = inputs.dataset_id()

    directory = prepare_build(root, dataset_id, force)

    artifacts = inputs.artifacts
    limit = inputs.tokenizer_config.max_pieces_per_value
    sources = inputs.sources()
    policy = config.context

    fit_group = artifacts.manifest["group"]

    counters: dict[str, Counters] = {}
    shards: dict[str, dict] = {}
    index: list[dict] = []
    outputs: list[Path] = []
    golden: list[dict] = []
    limitations: set[str] = set()

    for group in config.groups:

        entry = inputs.groups[group]

        writer = ShardWriter(
            directory=directory,
            group=group,
            shard_samples=config.shard_samples,
            row_group_samples=config.row_group_samples,
        )

        counter = Counters()

        pairs = [
            (client_id, cutoff)
            for client_id in entry.clients
            for cutoff in entry.cutoffs
        ]

        for client_id, cutoff in sorted(pairs, key=lambda item: (item[0], item[1])):

            sample, visible = _one(
                inputs=inputs,
                group=group,
                client_id=client_id,
                cutoff=cutoff,
                weight=entry.weight,
                sources=sources,
                policy=policy,
                limit=limit,
            )

            # Оценка обязана мериться одной линейкой. Потерянная
            # цель в validation или test делает её зависящей от
            # политики усечения, и молчать об этом нельзя.
            if group != fit_group and sample.selection.get("excluded_eligible", 0):
                raise BuildError(
                    f"группа {group}, клиент {client_id} на срезе {cutoff.date()}: отбор контекста "
                    f"выбросил {sample.selection['excluded_eligible']} событий периода целей. "
                    "В оценочной группе это запрещено: поднимите бюджет контекста или соберите "
                    "её политикой all"
                )

            writer.add(sample)

            counter.add(sample, visible, _specials(artifacts, sample))

            limitations.update(sample.limitations)

            golden = _remember_golden(golden, sample, config.golden_limit)

        writer.close()

        index.extend(writer.index)
        outputs.extend(writer.outputs)

        for name, rows in writer.shards.items():
            shards[name] = {"group": group, "rows": rows}

        counters[group] = counter

    # --- указатель ---

    path = directory / INDEX_FILE
    write_table(path, pa.Table.from_pylist(index, schema=INDEX_SCHEMA), INDEX_SCHEMA)
    outputs.append(path)

    # --- отчёт ---

    report = _report(inputs, config, dataset_id, counters, shards, sources, limitations)

    path = directory / REPORT_JSON_FILE
    write_json(path, report)
    outputs.append(path)

    path = directory / REPORT_MD_FILE
    write_text(path, render_report_md(report))
    outputs.append(path)

    path = directory / GOLDEN_JSON_FILE
    write_json(path, golden)
    outputs.append(path)

    path = directory / GOLDEN_MD_FILE
    write_text(path, render_golden_md(golden))
    outputs.append(path)

    # --- манифест и публикация ---

    manifest = _manifest(inputs, config, dataset_id, report, shards)

    write_manifest(directory, manifest, outputs)

    target = publish(root, dataset_id)

    # Словарь обязан быть тем же и ПОСЛЕ сборки: датасет ничего
    # не дообучает, и это проверяется, а не обещается.
    artifacts.verify()

    return BuildResult(
        directory=target,
        dataset_id=dataset_id,
        manifest=manifest,
        report=report,
    )


def _one(inputs: DatasetInputs, group: str, client_id: str, cutoff: datetime,
         weight: float, sources: tuple[str, ...], policy, limit: int) -> tuple[Sample, int]:
    """
    Один пример: история на дату, её причины, кодирование, отбор.
    """

    entry = inputs.groups[group]

    try:
        history = entry.history(client_id, cutoff)

        causes = causes_as_of(entry.corpus.store, client_id, cutoff)

        encoded = encode_history(inputs.artifacts, history, limit, cause_of=causes)

        sample = build_sample(
            artifacts=inputs.artifacts,
            encoded=encoded,
            group=group,
            window=entry.window,
            weight=weight,
            sources=sources,
            policy=policy,
        )

    except FAILURES as error:
        raise BuildError(
            f"группа {group}, клиент {client_id} на срезе {cutoff.isoformat()}: {error}"
        ) from error

    return sample, encoded.n_events


def _remember_golden(golden: list[dict], sample: Sample, limit: int) -> list[dict]:
    """
    Читаемые примеры по устойчивым признакам, а не по именам
    клиентов.

    Имя клиента живёт в фикстуре, а признак «пустая история» или
    «есть исправленная запись» есть в любом наборе.
    """

    if limit <= 0:
        return golden

    traits = _traits(sample)

    taken = {item["trait"] for item in golden}

    for trait in traits:

        if trait in taken:
            continue

        golden.append(_golden_entry(sample, trait))

        return golden[:limit]

    # Самая длинная история интереснее прежней самой длинной.
    for item in golden:
        if item["trait"] == "longest" and item["n_events"] < sample.n_events:
            item.update(_golden_entry(sample, "longest"))
            break

    return golden[:limit]


def _traits(sample: Sample) -> list[str]:

    out: list[str] = []

    if sample.n_events == 0:
        out.append("empty_history")

    if not sample.has_profile:
        out.append("no_profile")

    if not sample.has_targets:
        out.append("no_targets")

    if any(row["event_version"] > 1 for row in sample.events):
        out.append("corrected_event")

    if sample.truncated:
        out.append("truncated")

    if sample.n_events:
        out.append("longest")

    return out


def _golden_entry(sample: Sample, trait: str) -> dict:

    kept = [row for row in sample.events if row["kept"]][:8]

    return {
        "trait": trait,
        "sample_id": sample.sample_id,
        "group": sample.group,
        "client_id": sample.client_id,
        "cutoff": sample.cutoff.isoformat(),
        "n_events": sample.n_events,
        "n_tokens": sample.n_tokens,
        "n_values": sample.n_values,
        "weight": sample.weight,
        "has_targets": sample.has_targets,
        "n_eligible_events": sample.n_eligible_events,
        "profile_state": sample.profile_state,
        "profile_tokens": sample.profile_tokens,
        "history_age_days": sample.history_age_days,
        "history_age_reason": sample.history_age_reason,
        "selection": sample.selection,
        "events": [
            {
                "event_index": row["event_index"],
                "event_type": row["event_type"],
                "event_time": row["event_time"].isoformat(),
                "event_id": row["event_id"],
                "event_version": row["event_version"],
                "n_tokens": row["n_tokens"],
                "eligible": row["eligible"],
                "hour_known": row["hour_known"],
                "selection_reason": row["selection_reason"],
                "value_keys": row["value_keys"][:12],
            }
            for row in kept
        ],
        "excluded": [
            {
                "event_type": row["event_type"],
                "event_time": row["event_time"].isoformat(),
                "eligible": row["eligible"],
                "exclusion_reason": row["exclusion_reason"],
            }
            for row in sample.events
            if not row["kept"]
        ][:8],
        "batch": _batch_view(sample),
    }


def _batch_view(sample: Sample) -> dict:
    """
    Тот же пример в координатах batch из одного примера.
    """

    batch = collate([sample])

    return {
        "rows": batch.n_rows,
        "width": batch.width,
        "padded_tokens": batch.padded_tokens,
        "real_tokens": int(batch.event_token_mask.sum()),
        "target_candidates": int(batch.target_candidate_mask.sum()),
        "profile_tokens": int(batch.profile_token_mask.sum()),
        "history_slots": int(batch.history_mask.sum()),
    }


def _report(inputs: DatasetInputs, config: DatasetConfig, dataset_id: str,
            counters: dict[str, Counters], shards: dict[str, dict],
            sources: tuple[str, ...], limitations: set[str]) -> dict:

    by_group = {group: counter.as_dict() for group, counter in sorted(counters.items())}

    total = {
        "samples": sum(item["samples"] for item in by_group.values()),
        "clients": sum(item["clients"] for item in by_group.values()),
        "events": sum(item["events"] for item in by_group.values()),
        "tokens": sum(item["tokens"] for item in by_group.values()),
        "values": sum(item["values"] for item in by_group.values()),
        "eligible_events": sum(item["eligible_events"] for item in by_group.values()),
        "samples_without_targets": sum(item["samples_without_targets"] for item in by_group.values()),
        "empty_history": sum(item["empty_history"] for item in by_group.values()),
        "truncated": sum(item["truncated"] for item in by_group.values()),
    }

    declared = {
        group: inputs.groups[group].declared_eligible
        for group in config.groups
        if inputs.groups[group].declared_eligible is not None
    }

    agreement = {
        group: {
            "dataset": by_group[group]["eligible_events"],
            "split_manifest": value,
            "agree": by_group[group]["eligible_events"] == value,
            "comparable": (
                config.context.policy == "all"
                and len(inputs.groups[group].cutoffs) == 1
            ),
        }
        for group, value in declared.items()
    }

    return {
        "stage": "build",
        "schema_version": SCHEMA_VERSION,
        "format_version": FORMAT_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "dataset_id": dataset_id,
        "readiness": inputs.readiness,
        "inputs": inputs.as_dict(),
        "config": config.as_dict(),
        "counts": {**total, "by_group": by_group},
        "eligible_agreement": agreement,
        "sources": list(sources),
        "coverage_states": list(COVERAGE_STATES),
        "shards": shards,
        "limitations": sorted(limitations),
    }


def _manifest(inputs: DatasetInputs, config: DatasetConfig, dataset_id: str,
              report: dict, shards: dict[str, dict]) -> dict:

    return {
        "stage": "build",
        "implementation_version": IMPLEMENTATION_VERSION,
        "dataset_id": dataset_id,
        "identity": inputs.identity(),
        "config": config.as_dict(),
        "readiness": inputs.readiness,
        "inputs": inputs.as_dict(),
        "counts": {
            "samples": report["counts"]["samples"],
            "by_group": {
                group: item["samples"] for group, item in report["counts"]["by_group"].items()
            },
        },
        "shards": shards,
        "sources": report["sources"],
        "coverage_states": report["coverage_states"],
        "channels": {
            "model": list(MODEL_COLUMNS),
            "masker": list(MASKER_COLUMNS),
            "service": list(SERVICE_COLUMNS),
        },
        "contract": {
            "sample": "один пример это один клиент на один срез",
            "masks": "True это настоящее значение, а не выравнивание",
            "target_candidate_mask": (
                "область допустимых целей, а не выбранные цели: датасет не маскирует ничего, "
                "выбор делает Masker"
            ),
            "value_address": (
                "значение адресуется парой «событие, столбец внутри события»; адрес одинаков "
                "в одиночном примере и после сборки batch"
            ),
            "dependencies": (
                "in_context обещает адрес события И значения, cause_event только адрес события, "
                "остальные статусы адреса не дают"
            ),
            "not_tracked": list(NOT_TRACKED),
            "relation_features": list(RELATION_FEATURES),
            "dependency_statuses": list(DEP_STATUSES),
            "coverage": "состояние источника на срез, одно на пример",
            "weight": (
                "вес примера равен единице, делённой на число срезов клиента; применять его "
                "нужно после нормировки потерь внутри примера"
            ),
            "empty_sample": (
                "пример без событий и пример без целей сохраняются и помечены; шаг обучения "
                "без целей обязан быть пропущен, а не дать нулевую ошибку"
            ),
            "dataset_id": (
                "потребитель называет dataset_id явно: «последний собранный» набор не существует"
            ),
            "tokenized": (
                "данные собираются из смыслового слоя заново на каждый срез; "
                "data/tokenized не читается"
            ),
        },
    }


__all__ = [
    "BuildError",
    "BuildResult",
    "Counters",
    "build_dataset",
]
