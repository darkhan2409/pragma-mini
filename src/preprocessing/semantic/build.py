from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pyarrow as pa

from ..artifacts import _md_table, read_json, write_json, write_table, write_text
from ..canonical.build import REGISTRY_FILE as CANONICAL_REGISTRY_FILE
from ..canonical.build import STAGE as CANONICAL_STAGE
from ..canonical.registry import catalogue_from_registry
from ..history import CanonicalStore
from ..projection import PROJECTION_VERSION, projection_registry
from ..settings import CALENDAR_ENCODING, PreprocessingConfig
from . import activity as activity_module
from . import chains as chains_module
from .as_of import SEMANTIC_VERSION, SemanticHistory, semantic_as_of
from .keys import DERIVED_KEYS, KEYS_VERSION, RELATION_KEYS, TIMING_KEYS, keys_registry


# ============================================================
# ИДЕЯ
# ============================================================
#
# Этап 5 ничего не материализует как обучающий датасет: он
# объявляет смыслы и показывает, что они работают.
#
# Выход: реестр смыслов, читаемый отчёт, несколько понятных
# примеров и две диагностические таблицы с явным cutoff в имени.
# Таблица на позднем срезе входом раннего среза не является и
# никогда им не станет: имя файла называет свой момент.
# ============================================================


STAGE = "semantic"
STAGE_VERSION = "4.0.0"
SCHEMA_VERSION = 1

REGISTRY_FILE = "semantic_registry.json"
REPORT_MD_FILE = "semantic_report.md"

ACTIVITY_SCHEMA = pa.schema(
    [
        ("client_id", pa.string()),
        ("month", pa.string()),
        ("state", pa.string()),
        ("events", pa.int64()),
        ("client_actions", pa.int64()),
        ("partial", pa.bool_()),
    ]
)

CHAINS_SCHEMA = pa.schema(
    [
        ("client_id", pa.string()),
        ("kind", pa.string()),
        ("started_at", pa.timestamp("us")),
        ("last_step_at", pa.timestamp("us")),
        ("steps", pa.int64()),
        ("first_event_type", pa.string()),
        ("last_event_type", pa.string()),
        ("outcome", pa.string()),
    ]
)


@dataclass
class SemanticResult:
    report: dict
    outputs: list[Path] = field(default_factory=list)


def build_group(
    canonical_dir: Path,
    target: Path,
    config: PreprocessingConfig,
    group: str | None,
    cutoff: datetime,
    clients: list[str] | None = None,
    examples: int = 1,
) -> SemanticResult:
    """
    Собирает реестр смыслов и диагностику на одном срезе.
    """

    target = Path(target)

    store = CanonicalStore(Path(canonical_dir))

    # Каталог ключей читается из реестра полей canonical: это
    # ВХОД этапа. Импорт генератора описывал бы код, а не данные.
    catalogue = catalogue_from_registry(
        read_json(Path(canonical_dir) / CANONICAL_REGISTRY_FILE)
    )

    if not catalogue:
        raise SemanticError(
            f"в {CANONICAL_REGISTRY_FILE} нет полей payload: "
            "каталог ключей восстановить нельзя, соберите canonical заново"
        )

    payload_names = _payload_names(catalogue)

    registry = keys_registry(catalogue)

    wanted = clients or [row["client_id"] for row in store.clients[: config.semantic_sample_clients]]

    try:
        histories: list[SemanticHistory] = [
            semantic_as_of(store, client_id, cutoff) for client_id in wanted
        ]
    except chains_module.ChainsError as error:
        raise SemanticError(str(error)) from error

    activity_rows: list[dict] = []
    chain_rows: list[dict] = []
    used_keys: set[str] = set()
    values_total = 0
    derived_reasons: dict[str, int] = {}
    limitations: set[str] = set()

    for history in histories:

        for item in history.events:

            # Считаются все ключи, которые событие отдаёт модели:
            # значения, интервалы и удавшиеся расчёты. Проверка
            # только по values пропускала бы производные признаки.
            model_values = item.model_values()

            used_keys |= set(model_values)
            values_total += len(model_values)

            for derived in item.derived:
                if derived.reason is not None:
                    derived_reasons[derived.reason] = derived_reasons.get(derived.reason, 0) + 1

        activity_rows.extend({"client_id": history.client_id, **month.as_dict()} for month in history.activity)
        chain_rows.extend({"client_id": history.client_id, **chain.as_dict()} for chain in history.chains)

        limitations.update(history.limitations)

    moment = cutoff.date().isoformat()

    outputs: list[Path] = []

    activity_path = target / f"activity_months__{moment}.parquet"
    write_table(activity_path, pa.Table.from_pylist(activity_rows, schema=ACTIVITY_SCHEMA), ACTIVITY_SCHEMA)
    outputs.append(activity_path)

    chains_path = target / f"chains__{moment}.parquet"
    write_table(chains_path, pa.Table.from_pylist(chain_rows, schema=CHAINS_SCHEMA), CHAINS_SCHEMA)
    outputs.append(chains_path)

    declared = set(registry["keys"])

    report = {
        "stage": STAGE,
        "schema_version": SCHEMA_VERSION,
        "stage_version": STAGE_VERSION,
        "semantic_version": SEMANTIC_VERSION,
        "keys_version": KEYS_VERSION,
        "projection_version": PROJECTION_VERSION,
        "group": group,
        "cutoff": cutoff.isoformat(),
        "clients_checked": len(histories),
        "events": sum(history.n_events for history in histories),
        "values": values_total,
        "keys_declared": len(declared),
        "keys_used": len(used_keys),
        "keys_unused": sorted(declared - used_keys),
        "undeclared_keys": sorted(used_keys - declared),
        "activity": _activity_totals(activity_rows),
        "chains": chains_module.chain_summary(
            [chain for history in histories for chain in history.chains]
        ),
        "relations": sum(len(history.relations) for history in histories),
        "relation_intervals": _relation_totals(
            [item for history in histories for item in history.relations]
        ),
        "derived_reasons": dict(sorted(derived_reasons.items())),
        "computed_keys": {
            "timing": sorted(key.key for key in TIMING_KEYS.values()),
            "relation": sorted(key.key for key in RELATION_KEYS.values()),
            "formula": sorted(key.key for key in DERIVED_KEYS.values()),
        },
        "calendar": {
            "channel": CALENDAR_ENCODING["channel"],
            "features": list(CALENDAR_ENCODING["features"]),
            # Правило для следующего этапа.
            "hour_rule": (
                "время события точное, поэтому час суток наблюдался всегда: "
                "пара hour_sin/hour_cos измерена у каждой записи"
            ),
        },
        "timing": {
            "precision_rule": (
                "интервалы since_previous_hours, since_same_type_hours, "
                "since_last_income_hours, age_of_history_days и days_to_due считаются "
                "по точному времени событий: усекать и согласовывать точности нечего"
            ),
        },
        "profile_rule": (
            "анкета клиента одна — итоговая, на границу выгрузки. Расчётные признаки, "
            "которые делят сумму операции на доход (amount_to_declared_income и "
            "подобные), берут ИМЕННО ЕЁ, а не анкету на момент операции: прежних "
            "значений в данных больше нет. На конечном срезе группы это честно, на "
            "более раннем было бы знанием из будущего — поэтому ранние срезы запрещены "
            "построителем датасета"
        ),
        "registry": registry,
        "model_projection": projection_registry(
            (name for name in payload_names),
            timezone=config.timezone,
        ),
        "limitations": sorted(limitations),
    }

    if report["undeclared_keys"]:
        raise SemanticError(
            "в значениях встретились ключи вне реестра: " + ", ".join(report["undeclared_keys"])
        )

    registry_path = target / REGISTRY_FILE
    write_json(registry_path, report)
    outputs.append(registry_path)

    md_path = target / REPORT_MD_FILE
    write_text(md_path, render_semantic_md(report))
    outputs.append(md_path)

    for history in histories[:examples]:
        path = target / "examples" / f"{history.client_id}__{moment}.md"
        write_text(path, render_example_md(history))
        outputs.append(path)

    return SemanticResult(report=report, outputs=outputs)


class SemanticError(ValueError):
    """
    Смысловой слой построить нельзя: значение без объявленного
    ключа наружу не выпускается.
    """


def _payload_names(catalogue: dict) -> list[str]:

    names: set[str] = set()

    for info in catalogue.values():
        fields = info["fields"] if isinstance(info, dict) else info.fields
        for item in fields:
            names.add(item["name"] if isinstance(item, dict) else item.name)

    return sorted(names)


def _relation_totals(items: list) -> dict:
    """
    Сколько связей найдено.

    Длительность есть у каждой: время событий точное, и порядок
    причины со следствием известен всегда.
    """

    return {
        "with_interval": sum(1 for item in items if item.days_since_related_event is not None),
        "total": len(items),
        "rule": "время точное: причина всегда раньше следствия, иначе это поломка данных",
    }


def _activity_totals(rows: list[dict]) -> dict:

    counts: dict[str, int] = {state: 0 for state in activity_module.MONTH_STATES}

    for row in rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1

    return {"months": len(rows), "by_state": counts}


# ============================================================
# ОТЧЁТЫ
# ============================================================


def render_semantic_md(report: dict) -> str:

    out: list[str] = []

    registry = report["registry"]

    out.append(f"# Смысловой слой: группа {report.get('group') or '—'}\n")
    out.append(
        f"Срез {report['cutoff'][:10]}. Клиентов проверено: {report['clients_checked']}, "
        f"событий {report['events']}, значений {report['values']}.\n"
    )

    out.append("\n## Ключи\n")
    out.append(
        _md_table(
            [[kind, count] for kind, count in sorted(registry["counts"]["by_value_kind"].items())],
            ["вид значения", "ключей"],
        )
    )
    out.append(
        f"\nОбъявлено ключей: {report['keys_declared']}, встретилось в данных: {report['keys_used']}. "
        "Ключ без значений на этом срезе не ошибка: событий такого типа у выборки могло не быть.\n"
    )

    out.append("\n## Спорные объединения\n")
    out.extend(f"- {', '.join(item['keys'])}: {item['reason']}" for item in registry["ambiguous"])
    out.append("")

    out.append("\n## Разрешённые объединения\n")
    out.extend(
        f"- {item['key']} ← {', '.join(item['sources'])}: {item['reason']}"
        for item in registry["allowed_sharing"]
    )
    out.append("")

    out.append("\n## Активность по месяцам\n")
    out.append(
        _md_table(
            [[state, count] for state, count in report["activity"]["by_state"].items()],
            ["состояние месяца", "месяцев"],
        )
    )
    out.append(
        "\nПустой месяц сам по себе не означает неактивность: вывод «действий не было» требует, "
        "чтобы источники в этом месяце наблюдались.\n"
    )

    out.append("\n## Цепочки\n")
    out.append(
        _md_table(
            [[kind, count] for kind, count in report["chains"]["by_kind"].items()],
            ["вид цепочки", "штук"],
        )
    )
    out.append(
        f"\nНезавершённых: {report['chains']['unfinished']}. {report['chains']['rule']}.\n"
    )

    if report["derived_reasons"]:
        out.append("\n## Почему расчёт не получился\n")
        out.append(
            _md_table(
                [[reason, count] for reason, count in report["derived_reasons"].items()],
                ["причина", "случаев"],
            )
        )
        out.append("\nПричина вместо числа: неизвестный знаменатель не превращается в ноль.\n")

    out.append("\n## Календарь\n")
    out.append(
        f"{report['calendar']['channel']}: {', '.join(report['calendar']['features'])}. "
        "Считается из event_time и в смысловые значения не входит.\n"
    )
    out.append(f"\n{report['calendar']['hour_rule']}.\n")
    out.append(f"\n{report['profile_rule']}.\n")

    if report.get("timing"):
        out.append(f"\n{report['timing']['precision_rule']}.\n")

    if report["limitations"]:
        out.append("\n## Ограничения\n")
        out.extend(f"- {item}" for item in report["limitations"])
        out.append("")

    return "\n".join(out) + "\n"


def render_example_md(history: SemanticHistory) -> str:

    out: list[str] = []

    out.append(f"# Клиент {history.client_id} на {history.cutoff}\n")
    out.append(
        f"Финальная очищенная история до этого момента: {history.n_events} событий, "
        f"{len({key for item in history.events for key in item.values})} смысловых ключей.\n"
    )

    if history.profile:
        out.append("\n## Профиль на дату\n")
        out.append(_md_table([[key, value] for key, value in sorted(history.profile.items())], ["ключ", "значение"]))

    out.append("\n## Последние события\n")

    for item in history.events[-5:]:

        out.append(f"\n### {item.values.get('event_type')} — {item.event_time}\n")
        out.append(_md_table([[key, value] for key, value in sorted(item.values.items())], ["ключ", "значение"]))

        timing = {key: value for key, value in item.timing.as_dict().items() if value is not None}
        if timing:
            out.append("\nВремя: " + ", ".join(f"{key} = {value}" for key, value in timing.items()) + "\n")

        derived = [f"{one.key} = {one.value if one.value is not None else one.reason}" for one in item.derived]
        if derived:
            out.append("Расчёты: " + "; ".join(derived) + "\n")

    out.append("\n## Активность по месяцам\n")
    out.append(
        _md_table(
            [[item.month, item.state, item.events, item.client_actions, "да" if item.partial else "нет"]
             for item in history.activity],
            ["месяц", "состояние", "событий", "действий клиента", "неполный"],
        )
    )
    out.append(f"\nСводка: {history.activity_summary['by_state']}\n")

    if history.chains:
        out.append("\n## Цепочки\n")
        out.append(
            _md_table(
                [[item.kind, item.started_at, item.steps, item.first_event_type, item.outcome]
                 for item in history.chains],
                ["вид", "начало", "шагов", "первое событие", "исход"],
            )
        )

    if history.relations:
        out.append("\n## Связи событий\n")
        out.append(
            _md_table(
                [[item.related_event_type, item.relation_type,
                  item.reason if item.days_since_related_event is None
                  else round(item.days_since_related_event, 3),
                  "—" if item.same_merchant is None else ("да" if item.same_merchant else "нет")]
                 for item in history.relations[:10]],
                ["событие-причина", "вид связи", "дней прошло", "та же точка"],
            )
        )

    if history.limitations:
        out.append("\n## Ограничения\n")
        out.extend(f"- {item}" for item in history.limitations)
        out.append("")

    return "\n".join(out) + "\n"


__all__ = [
    "ACTIVITY_SCHEMA",
    "CANONICAL_STAGE",
    "CHAINS_SCHEMA",
    "REGISTRY_FILE",
    "REPORT_MD_FILE",
    "SCHEMA_VERSION",
    "STAGE",
    "STAGE_VERSION",
    "SemanticError",
    "SemanticResult",
    "build_group",
    "render_example_md",
    "render_semantic_md",
]
