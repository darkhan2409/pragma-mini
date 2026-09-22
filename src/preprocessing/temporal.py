from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Iterable, Sequence

import pyarrow as pa

from .artifacts import _md_table
from .history import (
    INTERNAL_COLUMNS,
    PAIR_OWN,
    PAIR_VISIBLE,
    CanonicalStore,
    ClientHistory,
    history_as_of,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Отчёт этапа 3 проверяет не данные, а ПРАВИЛА чтения.
#
# Внутри одного среза: ничего из будущего не видно, у события
# ровно одна версия, бизнес-порядок не ломается, служебные поля
# наружу не выходят.
#
# Между срезами: набор известных событий только растёт, версия
# события не понижается, а содержимое уже известного события не
# меняется от того, что срез стал позже. Последнее и есть главное
# обещание этапа: более поздний срез добавляет будущее, но не
# переписывает прошлое.
# ============================================================


CHECKS = {
    "no_future_rows": "во входе нет строк с event_time на границе cutoff либо позже",
    "one_row_per_event": "у каждого номера события клиента ровно одна строка",
    "business_order_non_decreasing": "бизнес-порядок не убывает по времени события",
    "internal_columns_hidden": "служебные поля наружу не выдаются",
    "one_profile_row": "у клиента не больше одной строки профиля",
    "known_events_only_grow": "между срезами набор известных событий только растёт",
    "past_is_immutable": "содержимое уже известного события не меняется от более позднего среза",
}


def month_starts(start: datetime, end: datetime) -> list[datetime]:
    """
    Начала месяцев в [start, end].
    """

    out: list[datetime] = []

    year, month = start.year, start.month

    while datetime(year, month, 1) <= end:
        moment = datetime(year, month, 1)
        if moment >= start:
            out.append(moment)
        month += 1
        if month == 13:
            month = 1
            year += 1

    return out


def choose_cutoffs(window_start: datetime, final_cutoff: datetime, period_end: datetime, count: int) -> list[datetime]:
    """
    Срезы для отчёта: конечный cutoff группы плюс равномерно
    разреженные более ранние начала месяцев.

    Расписание обучающих срезов здесь не выбирается: это только
    точки, на которых проверяются правила чтения.
    """

    limit = min(final_cutoff, period_end)

    candidates = [moment for moment in month_starts(window_start, limit) if moment <= limit]

    if not candidates:
        return [limit]

    if candidates[-1] != limit:
        candidates.append(limit)

    if len(candidates) <= count:
        return candidates

    step = (len(candidates) - 1) / (count - 1)

    picked = sorted({candidates[round(index * step)] for index in range(count)})

    if picked[-1] != limit:
        picked[-1] = limit

    return picked


# ============================================================
# ПРОВЕРКИ
# ============================================================


def _content_key(table: pa.Table, columns: Sequence[str]) -> dict[str, tuple]:

    # Тождество строки между срезами держит номер события внутри
    # клиента: идентификатора записи в выгрузке нет, а номер
    # canonical считает по времени, приоритету типа и месту в RAW
    # и от cutoff не зависит.
    rows = table.select(list(columns)).to_pylist()
    ids = table.column("stable_event_index").to_pylist()

    return {key: tuple(sorted(row.items(), key=lambda item: item[0])) for key, row in zip(ids, rows)}


def check_single(history: ClientHistory) -> list[str]:
    """
    Нарушения правил внутри одного среза.
    """

    problems: list[str] = []

    events = history.events
    cutoff = history.cutoff

    if events.num_rows:

        event_time = events.column("event_time").to_pylist()

        if any(value >= cutoff for value in event_time):
            problems.append("no_future_rows: событие не раньше cutoff")

        ids = events.column("stable_event_index").to_pylist()
        if len(ids) != len(set(ids)):
            problems.append("one_row_per_event: номер события встречается дважды")

        if any(later < earlier for earlier, later in zip(event_time, event_time[1:])):
            problems.append("business_order_non_decreasing: порядок убывает по времени события")

    leaked = [name for name in INTERNAL_COLUMNS if name in events.column_names]
    if leaked:
        problems.append(f"internal_columns_hidden: наружу вышли {leaked}")

    if (history.profile_meta or {}).get("rows", 0) > 1:
        problems.append("one_profile_row: у клиента больше одной строки профиля")

    return problems


def check_across(previous: ClientHistory, current: ClientHistory, columns: Sequence[str]) -> list[str]:
    """
    Нарушения правил между двумя срезами одного клиента.
    """

    problems: list[str] = []

    before = set(previous.events.column("stable_event_index").to_pylist())
    after = set(current.events.column("stable_event_index").to_pylist())

    lost = before - after
    if lost:
        problems.append(f"known_events_only_grow: пропало событий {len(lost)}")

    old = _content_key(previous.events, columns)
    new = _content_key(current.events, columns)

    changed = [key for key in old if key in new and old[key] != new[key]]

    if changed:
        problems.append(f"past_is_immutable: содержимое изменилось у {len(changed)} событий")

    return problems


# ============================================================
# ОТЧЁТ
# ============================================================


def temporal_report(
    store: CanonicalStore,
    clients: Iterable[str | int],
    cutoffs: Sequence[datetime],
    group: str | None = None,
) -> dict:

    cutoffs = sorted(cutoffs)

    counts: Counter = Counter()
    limitations: Counter = Counter()
    problems: list[dict] = []

    per_cutoff: dict[str, dict] = {}

    checked_clients = 0
    entity_states: Counter = Counter()
    transfers_visible = 0
    transfers_lonely = 0

    content_columns: list[str] | None = None

    for client in clients:

        checked_clients += 1

        previous: ClientHistory | None = None

        for cutoff in cutoffs:

            history = history_as_of(store, client, cutoff)

            if content_columns is None:
                content_columns = list(history.events.column_names)

            slot = per_cutoff.setdefault(
                cutoff.isoformat(),
                {"clients": 0, "events": 0, "with_profile": 0, "history_incomplete": 0},
            )

            slot["clients"] += 1
            slot["events"] += history.n_events
            slot["with_profile"] += int(history.profile is not None)
            slot["history_incomplete"] += int(history.relationship.history_incomplete)

            for key, value in history.counts.items():
                counts[key] += value

            for item in history.entities:
                entity_states[f"{item.kind}:{item.state}"] += 1

            for item in history.transfers:
                # Перевод себе наблюдается обеими ногами: он такой же
                # собранный, как пара двух клиентов.
                if item.pair_state in (PAIR_VISIBLE, PAIR_OWN):
                    transfers_visible += 1
                else:
                    transfers_lonely += 1

            for note in history.limitations:
                limitations[note.split(":")[0]] += 1

            found = check_single(history)

            if previous is not None:
                found += check_across(previous, history, content_columns or [])

            for item in found:
                problems.append(
                    {"client_id": history.client_id, "cutoff": cutoff.isoformat(), "problem": item}
                )

            previous = history

    return {
        "stage": "history",
        "group": group,
        "status": "ok" if not problems else "violations",
        "cutoffs": [moment.isoformat() for moment in cutoffs],
        "clients_checked": checked_clients,
        "checks": CHECKS,
        "problems": problems[:50],
        "problem_count": len(problems),
        "rows": dict(sorted(counts.items())),
        "per_cutoff": per_cutoff,
        "entity_states": dict(sorted(entity_states.items())),
        "transfers": {
            "counterpart_visible": transfers_visible,
            "counterpart_not_visible": transfers_lonely,
            "rule": (
                "встречная сторона видна только после своего события; иначе перевод односторонний. "
                "Перевод между своими счетами наблюдается обеими ногами у одного клиента и "
                "односторонним не считается"
            ),
        },
        "limitations": dict(sorted(limitations.items())),
        "rules": {
            "cutoff": "исключительная граница: событие обязано быть строго раньше",
            "rows": "запись приходит в выгрузку один раз и сразу окончательной",
            "order": "бизнес-порядок по времени события и приоритету типа",
            "profile": "одна итоговая строка на клиента; версий и границ действия нет",
            "transitions": (
                "переход состояния действует с момента, которым записан: отдельного "
                "времени вступления в силу у записи нет. Объявленное заранее изменение "
                "выразить нечем, и событие обязано рождаться тогда, когда изменение "
                "наступает"
            ),
            "internal": "служебные флаги слоя наружу не выдаются",
        },
    }


# ============================================================
# ЧИТАЕМАЯ ИСТОРИЯ
# ============================================================


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    return str(value)


def render_history_md(history: ClientHistory, tail: int = 15) -> str:

    out: list[str] = []

    out.append(f"# Клиент {history.client_id} на {history.cutoff.isoformat(sep=' ')}\n")
    out.append(
        "Показано то, что банк знал строго до этого момента: события, "
        "произошедшие раньше cutoff, в действующей версии.\n"
    )

    counts = history.counts

    out.append("## Что видно и что ещё нет\n")
    out.append(
        _md_table(
            [
                ["строк у клиента всего", counts["rows"]],
                ["видно событий", counts["visible"]],
                ["ещё не произошло", counts["event_not_happened"]],
            ],
            ["показатель", "значение"],
        )
    )

    meta = history.profile_meta

    out.append("\n## Профиль\n")

    if history.profile is None:
        out.append("Анкеты у клиента нет: банк её ещё не посчитал.\n")
    else:
        profile = history.profile
        out.append(
            _md_table(
                [
                    ["возраст клиента", _fmt(profile.get("age"))],
                    ["город", _fmt(profile.get("city"))],
                    ["доход заявленный", _fmt(profile.get("declared_income"))],
                    ["месяцев с банком по профилю", _fmt(profile.get("relationship_months"))],
                    ["договоров", _fmt(profile.get("contracts_count"))],
                ],
                ["поле", "значение"],
            )
        )

    relationship = history.relationship

    out.append("\n## Отношения с банком\n")
    out.append(
        _md_table(
            [
                ["наблюдение началось", _fmt(relationship.observed_start)],
                ["наблюдается дней", _fmt(relationship.observed_days)],
                ["история неполна", _fmt(relationship.history_incomplete)],
            ],
            ["показатель", "значение"],
        )
    )
    if history.entities:

        out.append("\n## Счета, карты, договоры, заявки\n")
        out.append(
            _md_table(
                [
                    [
                        item.kind,
                        item.entity_id,
                        _fmt(item.state),
                        _fmt(item.since),
                        _fmt(item.opening_observed),
                        _fmt(item.last_transition),
                    ]
                    for item in history.entities[:20]
                ],
                ["вид", "идентификатор", "состояние", "с", "открытие наблюдалось", "последний переход"],
            )
        )

    if history.transfers:

        out.append("\n## Переводы\n")
        out.append(
            _md_table(
                [
                    [
                        item.transfer_id,
                        _fmt(item.side),
                        _fmt(item.amount),
                        _fmt(item.event_time),
                        item.pair_state,
                        _fmt(item.counterpart_client_id),
                    ]
                    for item in history.transfers[-10:]
                ],
                ["перевод", "сторона", "сумма", "когда", "встречная сторона", "контрагент"],
            )
        )

    events = history.events

    if events.num_rows:

        out.append(f"\n## Последние {min(tail, events.num_rows)} событий в бизнес-порядке\n")

        rows = events.slice(max(0, events.num_rows - tail)).to_pylist()

        out.append(
            _md_table(
                [
                    [
                        _fmt(row["event_time"]),
                        row["type"],
                        row["source"],
                        _fmt(row.get("amount")),
                        _fmt(row.get("merchant_name") or row.get("counterparty") or row.get("template")),
                    ]
                    for row in rows
                ],
                ["время события", "тип", "источник", "сумма", "кому или что"],
            )
        )

    if history.limitations:
        out.append("\n## Ограничения восстановления\n")
        out.extend(f"- {item}" for item in history.limitations)
        out.append("")

    return "\n".join(out) + "\n"


def render_temporal_md(report: dict) -> str:

    out: list[str] = []

    out.append(f"# История на дату: группа {report.get('group') or '—'}\n")
    out.append(f"Статус: **{report['status']}**. Нарушений правил: {report['problem_count']}.\n")

    out.append("## Правила\n")
    out.append(_md_table([[name, text] for name, text in report["rules"].items()], ["правило", "смысл"]))

    out.append("\n## Проверки\n")
    out.append(_md_table([[name, text] for name, text in report["checks"].items()], ["проверка", "что утверждает"]))

    if report["problems"]:
        out.append("\n## Нарушения\n")
        out.extend(f"- {item['client_id']} на {item['cutoff']}: {item['problem']}" for item in report["problems"])
        out.append("")

    out.append(f"\nПроверено клиентов: {report['clients_checked']}, срезов: {len(report['cutoffs'])}.\n")

    out.append("\n## По срезам\n")
    out.append(
        _md_table(
            [
                [cutoff, item["clients"], item["events"], item["with_profile"], item["history_incomplete"]]
                for cutoff, item in sorted(report["per_cutoff"].items())
            ],
            ["cutoff", "клиентов", "видимых событий", "с профилем", "с неполной историей"],
        )
    )

    out.append("\n## Отброшено при чтении\n")
    out.append(_md_table([[name, value] for name, value in report["rows"].items()], ["причина", "строк"]))

    out.append("\n## Переводы\n")
    out.append(
        _md_table(
            [
                ["встречная сторона видна", report["transfers"]["counterpart_visible"]],
                ["встречная сторона не видна", report["transfers"]["counterpart_not_visible"]],
            ],
            ["показатель", "случаев"],
        )
    )
    out.append(f"\n{report['transfers']['rule']}.\n")

    if report["limitations"]:
        out.append("\n## Ограничения восстановления\n")
        out.append(_md_table([[name, value] for name, value in report["limitations"].items()], ["ограничение", "случаев"]))

    return "\n".join(out) + "\n"


__all__ = [
    "CHECKS",
    "check_across",
    "check_single",
    "choose_cutoffs",
    "month_starts",
    "render_history_md",
    "render_temporal_md",
    "temporal_report",
]
