from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

from ..config import (
    HISTORY_END,
    HISTORY_START,
    INITIATOR_CLIENT,
    INITIATOR_EXTERNAL,
    INITIATOR_SYSTEM,
    RAW_DIR,
    SOURCES,
)
from ..finance import invariants as invariants_module
from ..observe import leak_audit


# ============================================================
# ОТЧЁТ РЕАЛИЗМА
# ============================================================
#
# Отчёт обязан показать не только средние: распределения с
# нулевыми месяцами, длинные хвосты, переходы состояний,
# повторяемость мерчантов и контрагентов, доходы и задержки,
# стресс и восстановление, финансовые инварианты, продуктовые
# и мошеннические цепочки, дефекты источников, сравнение с
# калибровочными эталонами, список метрик без эталона,
# целостные истории клиентов и аудит proxy-утечек.
# ============================================================


CLIENT_INITIATORS = (INITIATOR_CLIENT,)


def _quantiles(values: list, points=(0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)) -> dict:

    if not values:
        return {f"p{int(point * 100)}": None for point in points}

    ordered = sorted(values)

    result = {}

    for point in points:
        index = min(len(ordered) - 1, int(point * len(ordered)))
        result[f"p{int(point * 100)}"] = ordered[index]

    result["mean"] = round(statistics.fmean(ordered), 2)
    result["max"] = ordered[-1]

    return result


def _round(value, digits: int = 4):
    return round(value, digits) if isinstance(value, float) else value


def _month(ts: datetime) -> str:
    return ts.strftime("%Y-%m")


def _months_between(start: datetime, end: datetime) -> list:

    months = []

    current = datetime(start.year, start.month, 1)

    while current < end:
        months.append(_month(current))
        current = (
            datetime(current.year + 1, 1, 1)
            if current.month == 12
            else datetime(current.year, current.month + 1, 1)
        )

    return months


def _load(raw_dir: Path) -> dict:

    data = {}

    for name, relative in (
        ("events", "events.parquet"),
        ("profile", "profile.parquet"),
        ("coverage", "source_coverage.parquet"),
        ("truth_clients", "truth/clients.parquet"),
        ("truth_events", "truth/events.parquet"),
        ("truth_relationships", "truth/relationships.parquet"),
        ("products", "catalog/products.parquet"),
        ("merchants", "catalog/merchants.parquet"),
    ):
        path = raw_dir / relative
        data[name] = pq.read_table(path).to_pylist() if path.exists() else []

    data["manifest"] = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))

    for row in data["events"]:
        row["payload"] = json.loads(row["payload"])

    return data


# ============================================================
# ОКНО НАБЛЮДЕНИЯ
# ============================================================
#
# source_coverage надо читать по смыслу, а не по именам колонок:
#
#   first_available_at  когда источник появился в БАНКЕ;
#                       к конкретному клиенту отношения не имеет
#   first_seen          max(доступность источника, начало клиента
#                       в этом источнике) — вот здесь клиент учтён
#   last_available_at   заполняется только при закрытии отношений;
#                       None означает «конец не задан», а не
#                       «наблюдения нет»
#   source_outage       сбои были, но КАЛЕНДАРЯ сбойных месяцев в
#                       строке нет: дни живут внутри генератора и
#                       наружу не выдаются
#   client_not_onboarded у источников приложения означает «нет
#                       установленного приложения», а вовсе не
#                       «нет отношений с банком»
#
# Отсюда три правила. Окно начинается с фактического начала
# отношений, а не с начала истории. Месяцы до регистрации и после
# закрытия в сетку не входят. Недостаток покрытия это НЕИЗВЕСТНО,
# а не ноль: такой месяц не превращается в пустой и не склеивает
# через себя паузу.
# ============================================================


# Источники, где вообще может появиться действие клиента.
# Каналы приложения входят: экран и операция в приложении это
# действия клиента, и без них молчание не доказано.
CLIENT_ACTION_SOURCES = (
    "transactions",
    "product_events",
    "applications",
    "loans",
    "support",
    "app_screens",
    "app_operations",
)

# Источники, определяющие начало отношений. Приложение и
# коммуникации сюда не входят: их отсутствие говорит о канале,
# а не о клиенте.
RELATIONSHIP_SOURCES = (
    "profile",
    "applications",
    "product_events",
    "loans",
    "transactions",
    "support",
    "antifraud",
)

# Причины, по которым источника у клиента нет ПО-НАСТОЯЩЕМУ:
# приложение не установлено, согласия не давал. Это свойство
# клиента, а не пробел наблюдения.
NOT_APPLICABLE_REASONS = ("client_not_onboarded", "no_consent")


def _known_from(rows: dict, required: tuple) -> str | None:
    """
    Месяц, начиная с которого известны ВСЕ нужные источники.

    None означает, что хотя бы один нужный источник не наблюдался
    никогда, и судить по нему нельзя вовсе.

    Покрытие здесь монотонно: источник начинается со своего
    first_seen и идёт до конца окна, а закрытие отношений окно и
    завершает. Поэтому достаточно самого позднего начала.
    """

    latest = ""

    for source in required:

        row = rows.get(source)

        if row is None:
            continue

        if row["first_seen"] is None:
            # Источника у клиента нет по-настоящему — это не
            # пробел наблюдения.
            if row["coverage_reason"] in NOT_APPLICABLE_REASONS:
                continue
            return None

        month = _month(row["first_seen"])

        if month > latest:
            latest = month

    return latest


def _next_month_start(month: str) -> datetime:

    year, number = int(month[:4]), int(month[5:])

    return datetime(year + 1, 1, 1) if number == 12 else datetime(year, number + 1, 1)


def _windows(data: dict) -> dict:
    """
    Окно наблюдения на каждого клиента и статус каждого месяца.

    Одно определение окна на весь отчёт: активность и раздел
    исчезновений обязаны смотреть на одну и ту же сетку.
    """

    by_client: dict[str, dict] = defaultdict(dict)

    for row in data["coverage"]:
        by_client[row["client_id"]][row["source"]] = row

    windows: dict[str, dict] = {}

    for client in data["truth_clients"]:

        client_id = client["client_id"]

        rows = by_client.get(client_id, {})

        # --- начало отношений ---

        starts = [
            rows[source]["first_seen"]
            for source in RELATIONSHIP_SOURCES
            if rows.get(source) and rows[source]["first_seen"] is not None
        ]

        if not starts:
            starts = [
                row["first_seen"] for row in rows.values() if row["first_seen"] is not None
            ]

        if not starts:
            windows[client_id] = {"start": None, "end": None, "months": [], "closed_at": None}
            continue

        start = max(HISTORY_START, min(starts))

        # --- конец: закрытие отношений либо открытый конец ---

        closings = [
            row["last_available_at"]
            for row in rows.values()
            if row["coverage_reason"] == "relationship_closed" and row["last_available_at"] is not None
        ]

        closed_at = min(closings) if closings else None

        end = min(HISTORY_END, closed_at) if closed_at is not None else HISTORY_END

        if end <= start:
            windows[client_id] = {"start": start, "end": end, "months": [], "closed_at": closed_at}
            continue

        known_action = _known_from(rows, CLIENT_ACTION_SOURCES)
        known_any = _known_from(rows, tuple(SOURCES))

        names = _months_between(start, end)

        months = []

        for index, name in enumerate(names):

            partial = (index == 0 and start.day != 1) or (
                index == len(names) - 1 and end < _next_month_start(name)
            )

            months.append(
                {
                    "month": name,
                    "partial": bool(partial),
                    "known_action": known_action is not None and name >= known_action,
                    "known_any": known_any is not None and name >= known_any,
                }
            )

        windows[client_id] = {
            "start": start,
            "end": end,
            "months": months,
            "closed_at": closed_at,
        }

    return windows


# ============================================================
# РАЗДЕЛЫ
# ============================================================


def _activity(data: dict) -> dict:
    """
    События на клиент-месяц по классам инициатора.

    Три РАЗНЫХ показателя месяца, и путать их нельзя:

      нет никаких записей          ни клиент, ни банк, ни система
                                   за месяц не записали ничего
      записи есть, клиент молчит   банк начислял проценты, слал
                                   выписки, клиент не делал ничего
      есть действия клиента        клиент что-то сделал сам

    Их доли дают ровно 100 %, а общий показатель «месяцев без
    действий клиента» равен сумме первых двух.

    Месяц, где нужных источников не хватает, не ноль, а
    НЕИЗВЕСТНО: он идёт своей строкой и в знаменатель не входит.
    """

    windows = data["windows"]

    per_client_month: dict[tuple, Counter] = defaultdict(Counter)

    for row in data["events"]:

        key = (row["client_id"], _month(row["event_time"]))

        per_client_month[key]["all"] += 1

        initiator = row["change_initiator"]

        if initiator == INITIATOR_CLIENT:
            per_client_month[key]["client"] += 1
        elif initiator == INITIATOR_SYSTEM:
            per_client_month[key]["system"] += 1
        elif initiator == INITIATOR_EXTERNAL:
            # Зарплата от работодателя, перевод от родни и чужая
            # рука мошенника — это не банк.
            per_client_month[key]["external"] += 1
        else:
            per_client_month[key]["bank"] += 1

    totals: list[int] = []
    client_only: list[int] = []
    bank_system: list[int] = []
    external_only: list[int] = []

    zero_months = 0
    bank_only_months = 0
    client_months = 0

    grid = 0
    classified = 0
    partial_months = 0
    unknown_months = 0

    # Активность клиента это условная ГРУППИРОВКА ОТЧЁТА, а не
    # свойство клиента: активным считаем того, кто действовал не
    # менее чем в половине своих месяцев. Раздел исчезновений
    # этой группировкой не пользуется.
    active_months: list[int] = []
    inactive_months: list[int] = []

    active_clients = 0
    inactive_clients = 0

    for client_id in sorted(windows):

        months = windows[client_id]["months"]

        counted: list[tuple] = []

        for item in months:

            if item["partial"]:
                partial_months += 1
                continue

            counts = per_client_month.get((client_id, item["month"]), Counter())

            counted.append((item, counts))

        if not counted:
            continue

        acted = sum(1 for _, counts in counted if counts["client"] > 0)

        is_active = acted * 2 >= len(counted)

        if is_active:
            active_clients += 1
        else:
            inactive_clients += 1

        for item, counts in counted:

            grid += 1

            totals.append(counts["all"])
            client_only.append(counts["client"])
            bank_system.append(counts["bank"] + counts["system"])
            external_only.append(counts["external"])

            (active_months if is_active else inactive_months).append(counts["all"])

            # Три вида месяца требуют, чтобы были известны ВСЕ
            # источники: иначе «нет никаких записей» отличить от
            # «часть записей не наблюдалась» невозможно, а доли
            # перестанут делить одно целое. Объёмные показатели
            # выше такой строгости не требуют и считаются по
            # всему окну.
            if not item["known_any"]:
                unknown_months += 1
                continue

            classified += 1

            if counts["client"] > 0:
                client_months += 1
            elif counts["all"] == 0:
                zero_months += 1
            else:
                bank_only_months += 1

    segments = Counter()

    for value in totals:
        if value <= 2:
            segments["silent_0_2"] += 1
        elif value <= 15:
            segments["sleepy_3_15"] += 1
        elif value <= 60:
            segments["moderate_16_60"] += 1
        elif value <= 130:
            segments["regular_61_130"] += 1
        elif value <= 250:
            segments["high_131_250"] += 1
        else:
            segments["extreme_250_plus"] += 1

    silent = zero_months + bank_only_months

    return {
        # Сетка объёма: все полные месяцы окна. По ней считаются
        # распределения и частоты вроде сессий на клиент-месяц.
        "client_months": grid,
        # Сетка классификации: месяцы, где известны все источники.
        # Только на ней можно утверждать «записей не было вовсе».
        "classified_months": classified,
        "partial_months": partial_months,
        "unknown_months": unknown_months,
        "coverage_note": (
            "покрытие проверено по доступности источников; "
            "отдельные дни сбоев в source_coverage не восстановимы и не учтены"
        ),
        # Определение НЕ менялось: это все месяцы без действий
        # клиента, включая полностью пустые.
        "no_client_action_month_share": round(silent / classified, 4) if classified else None,
        "zero_month_share": round(zero_months / classified, 4) if classified else None,
        "bank_only_month_share": round(bank_only_months / classified, 4) if classified else None,
        "client_action_month_share": round(client_months / classified, 4) if classified else None,
        "month_kinds_sum": (
            round((zero_months + bank_only_months + client_months) / classified, 4)
            if classified
            else None
        ),
        "active_clients": active_clients,
        "inactive_clients": inactive_clients,
        "active_definition": "действия клиента не менее чем в половине его полных месяцев",
        "all_events": _quantiles(totals),
        "client_events": _quantiles(client_only),
        "bank_system_events": _quantiles(bank_system),
        "external_events": _quantiles(external_only),
        "active_client_months": _quantiles(active_months) if active_months else {},
        "inactive_client_months": _quantiles(inactive_months) if inactive_months else {},
        "segments": {name: round(count / grid, 4) for name, count in segments.items()} if grid else {},
        "events_by_type": dict(Counter(row["event_type"] for row in data["events"]).most_common()),
        "events_by_source": dict(Counter(row["source"] for row in data["events"]).most_common()),
        "events_by_initiator": dict(Counter(row["change_initiator"] for row in data["events"])),
    }


def _absence(data: dict) -> dict:
    """
    Исчезновение и возвращение клиента по НАБЛЮДАЕМЫМ действиям.

    Пауза это серия подряд идущих месяцев окна без действий
    клиента. Три отрезка молчания различаются и не смешиваются:

      стартовое молчание   месяцы от начала окна до первого
                           действия. Клиент ещё не начал, это не
                           исчезновение, и ни в один знаменатель
                           оно не входит
      пауза                молчание ранее активного клиента
      неизвестный месяц    нужных источников не хватает, судить
                           нельзя

    У паузы четыре исхода. Три установленных — вернулся, молчит
    на конец наблюдения, отношения закрыты — образуют знаменатель
    доли возвращения. Четвёртый, «наблюдение прервано», в него не
    входит: про такую паузу неизвестно ничего, и отнести её к
    любому из трёх значило бы выдумать факт.

    Молчание на конце окна НЕ равно уходу из банка: будущее
    клиента за границей датасета неизвестно. Подтверждённое
    закрытие отношений считается отдельно и только по факту.
    """

    windows = data["windows"]

    acted: dict[tuple, bool] = {}
    used_product: dict[tuple, bool] = {}

    product_use = (
        "product_opened",
        "application_submitted",
        "installment_paid",
        "deposit_topup",
        "purchase",
    )

    for row in data["events"]:

        if row["change_initiator"] != INITIATOR_CLIENT:
            continue

        key = (row["client_id"], _month(row["event_time"]))

        acted[key] = True

        if row["event_type"] in product_use:
            used_product[key] = True

    # --- подтверждённое закрытие отношений ---
    #
    # Закрыты ПО ФАКТУ на конец наблюдения: состояние клиента
    # закрытое либо покрытие источников кончилось закрытием.
    # Клиент, который когда-то закрывался, а потом вернулся,
    # отношений не прекратил, и в это число не входит — для него
    # есть отдельная строка.

    closed: set[str] = {
        row["client_id"]
        for row in data["truth_clients"]
        if row["final_state"] == "closed_relationship"
    }

    for client_id, window in windows.items():
        if window["closed_at"] is not None:
            closed.add(client_id)

    closed_ever: set[str] = set(closed)

    for row in data["truth_events"]:
        if row["kind"] == "state_transition" and row["key"] == "closed_relationship":
            closed_ever.add(row["client_id"])

    buckets = ("1-2", "3-5", "6-11", "12+")

    episodes = {name: Counter() for name in buckets}
    clients_with = {name: set() for name in buckets}

    outcomes = Counter()

    leading_silence = []
    never_acted = 0
    returned_to_products: set[str] = set()
    clients_counted = 0

    for client_id in sorted(windows):

        months = windows[client_id]["months"]

        if not months:
            continue

        clients_counted += 1

        # Зафиксированное действие сильнее пробела в покрытии:
        # месяц с действием активен при любом покрытии. Обратное
        # неверно — молчание при нехватке источников это
        # неизвестность, а не бездействие.
        acted_in = [bool(acted.get((client_id, item["month"]))) for item in months]

        if not any(acted_in):
            never_acted += 1
            leading_silence.append(len(months))
            continue

        first = acted_in.index(True)

        if first:
            leading_silence.append(first)

        run = 0
        broken = False

        for index in range(first + 1, len(months) + 1):

            if index == len(months):

                if run:
                    if broken:
                        outcome = "observation_broken"
                    elif client_id in closed:
                        outcome = "closed"
                    else:
                        outcome = "ongoing"
                    _record(episodes, clients_with, client_id, run, outcome)
                    outcomes[outcome] += 1

                break

            if acted_in[index]:

                if run:
                    outcome = "observation_broken" if broken else "returned"
                    _record(episodes, clients_with, client_id, run, outcome)
                    outcomes[outcome] += 1

                    if not broken and any(
                        used_product.get((client_id, months[position]["month"]))
                        for position in range(index, len(months))
                    ):
                        returned_to_products.add(client_id)

                run = 0
                broken = False
                continue

            # Молчание: либо настоящее, либо неизвестность.
            # Неизвестный месяц паузу не склеивает, а помечает её
            # неопределимой.
            if not months[index]["known_action"]:
                broken = True

            run += 1

    established = outcomes["returned"] + outcomes["ongoing"] + outcomes["closed"]

    return {
        "clients": clients_counted,
        "leading_silence_months": _quantiles(leading_silence) if leading_silence else {},
        "clients_never_acted": never_acted,
        "buckets": {
            name: {
                "episodes": sum(episodes[name].values()),
                "clients": len(clients_with[name]),
                "client_share": (
                    round(len(clients_with[name]) / clients_counted, 4) if clients_counted else None
                ),
                "returned": episodes[name]["returned"],
                "ongoing": episodes[name]["ongoing"],
                "closed": episodes[name]["closed"],
                "observation_broken": episodes[name]["observation_broken"],
            }
            for name in buckets
        },
        "outcomes": dict(outcomes),
        # Наблюдаемый результат НА ДАТУ КОНЦА ДАТАСЕТА, а не
        # вероятность возвращения: сроки наблюдения у клиентов
        # разные, и продолжающаяся пауза ещё может кончиться
        # возвратом.
        "returned_by_window_end_share": (
            round(outcomes["returned"] / established, 4) if established else None
        ),
        "pause_ongoing_at_window_end_share": (
            round(outcomes["ongoing"] / established, 4) if established else None
        ),
        "confirmed_closure_clients": len(closed & set(windows)),
        "confirmed_closure_share": (
            round(len(closed & set(windows)) / clients_counted, 4) if clients_counted else None
        ),
        # Закрывались, но вернулись: отношения не прекращены.
        "closed_once_but_returned": len((closed_ever - closed) & set(windows)),
        "returned_and_used_products": len(returned_to_products),
        # Диагностика самого генератора. Порог там ДРУГОЙ: запись
        # ставится после перерыва не менее 45 дней между
        # действиями клиента, а раздел выше считает полные
        # календарные месяцы. Совпадать счётчики не обязаны, и
        # расхождение ошибкой не является.
        "generator_pause_notes": {
            "pause_start": sum(1 for row in data["truth_events"] if row["kind"] == "pause_start"),
            "pause_end": sum(1 for row in data["truth_events"] if row["kind"] == "pause_end"),
            "threshold_days": 45,
        },
    }


def _record(episodes: dict, clients_with: dict, client_id: str, length: int, outcome: str) -> None:

    name = "1-2" if length <= 2 else "3-5" if length <= 5 else "6-11" if length <= 11 else "12+"

    episodes[name][outcome] += 1
    clients_with[name].add(client_id)


def _long_tails(data: dict) -> dict:

    def tail(counter: Counter) -> dict:
        total = sum(counter.values())
        if not total:
            return {"distinct": 0}
        ordered = counter.most_common()
        top10 = sum(count for _, count in ordered[:10]) / total
        singles = sum(1 for _, count in ordered if count == 1)
        return {
            "distinct": len(ordered),
            "top10_share": round(top10, 4),
            "singleton_share": round(singles / len(ordered), 4),
            "top5": [name for name, _ in ordered[:5]],
        }

    mcc = Counter()
    merchants = Counter()
    outlets = Counter()
    templates = Counter()
    cities = Counter()
    counterparties = Counter()

    for row in data["events"]:

        payload = row["payload"]

        if payload.get("mcc"):
            mcc[payload["mcc"]] += 1
        if payload.get("merchant_id"):
            merchants[payload["merchant_id"]] += 1
        if payload.get("outlet_id"):
            outlets[payload["outlet_id"]] += 1
        if payload.get("template"):
            templates[payload["template"]] += 1
        if payload.get("merchant_city"):
            cities[payload["merchant_city"]] += 1
        if payload.get("counterparty"):
            counterparties[payload["counterparty"]] += 1

    return {
        "mcc": tail(mcc),
        "merchants": tail(merchants),
        "outlets": tail(outlets),
        "templates": tail(templates),
        "cities": tail(cities),
        "counterparties": tail(counterparties),
    }


def _repeatability(data: dict) -> dict:
    """
    Повторяемость мерчантов и контрагентов: доля покупок в
    любимых точках и доля переводов знакомым.
    """

    by_client_outlet: dict[str, Counter] = defaultdict(Counter)
    by_client_counterparty: dict[str, Counter] = defaultdict(Counter)

    for row in data["events"]:

        payload = row["payload"]

        if row["event_type"] == "purchase" and payload.get("outlet_id"):
            by_client_outlet[row["client_id"]][payload["outlet_id"]] += 1

        if row["event_type"] in ("p2p_out", "transfer_out") and payload.get("counterparty"):
            by_client_counterparty[row["client_id"]][payload["counterparty"]] += 1

    favourite_shares = []

    for counter in by_client_outlet.values():
        total = sum(counter.values())
        if total < 10:
            continue
        top3 = sum(count for _, count in counter.most_common(3))
        favourite_shares.append(top3 / total)

    known_shares = []

    for counter in by_client_counterparty.values():
        total = sum(counter.values())
        if total < 5:
            continue
        repeated = sum(count for _, count in counter.items() if count > 1)
        known_shares.append(repeated / total)

    return {
        "top3_outlet_share": _quantiles(favourite_shares) if favourite_shares else {},
        "repeat_counterparty_share": _quantiles(known_shares) if known_shares else {},
        "clients_with_purchases": len(by_client_outlet),
    }


def _lifecycle(data: dict) -> dict:

    transitions = Counter()
    pauses = []
    planned_returns = 0

    for row in data["truth_events"]:

        if row["kind"] == "state_transition":
            transitions[row["key"]] += 1

        if row["kind"] == "pause_start":
            value = json.loads(row["value"])
            start = row["ts"]
            end = datetime.fromisoformat(value["actual_end"])
            pauses.append((end - start).days)
            # Это ПЛАН, а не факт: намерение вернуться, записанное
            # при планировании паузы. Наблюдаемые возвращения
            # считает раздел «Исчезновение и возвращение».
            if value.get("return_trigger") not in (None, "none"):
                planned_returns += 1

    states = Counter(row["final_state"] for row in data["truth_clients"])

    return {
        "state_transitions": dict(transitions.most_common()),
        "final_states": dict(states.most_common()),
        "pauses": {
            "count": len(pauses),
            "length_days": _quantiles(pauses) if pauses else {},
            "with_planned_return": planned_returns,
        },
    }


def _income(data: dict) -> dict:

    outcomes = Counter()
    delays = []

    for row in data["truth_events"]:
        if row["kind"] == "income_event":
            outcomes[row["key"]] += 1

    salaries = defaultdict(list)

    for row in data["events"]:
        if row["event_type"] in ("salary_credit", "pension_credit"):
            salaries[row["client_id"]].append(row["event_time"])

    for moments in salaries.values():
        moments.sort()
        for left, right in zip(moments, moments[1:]):
            delays.append((right - left).days)

    return {
        "payout_outcomes": dict(outcomes.most_common()),
        "gap_between_credits_days": _quantiles(delays) if delays else {},
        "clients_with_salary": len(salaries),
    }


def _stress(data: dict) -> dict:

    starts = Counter()
    resolutions = Counter()
    lengths = []

    for row in data["truth_events"]:

        if row["kind"] == "stress_start":
            starts[row["key"]] += 1
            value = json.loads(row["value"])
            lengths.append((datetime.fromisoformat(value["end"]) - row["ts"]).days)

        if row["kind"] == "stress_end":
            resolutions[row["key"]] += 1

    dpd = Counter()
    cleared = 0

    for row in data["events"]:
        if row["event_type"] == "delinquency_registered":
            dpd[row["payload"].get("days_past_due")] += 1
        if row["event_type"] == "arrears_cleared":
            cleared += 1

    return {
        "episodes": dict(starts.most_common()),
        "resolutions": dict(resolutions.most_common()),
        "length_days": _quantiles(lengths) if lengths else {},
        "delinquency_milestones": {str(key): value for key, value in sorted(dpd.items(), key=lambda item: (item[0] or 0))},
        "arrears_cleared": cleared,
        "restructured": sum(1 for row in data["events"] if row["event_type"] == "loan_restructured"),
    }


def _products(data: dict) -> dict:

    submitted = [row for row in data["events"] if row["event_type"] == "application_submitted"]
    decided = [row for row in data["events"] if row["event_type"] == "application_decision"]
    opened = [row for row in data["events"] if row["event_type"] == "product_opened"]
    migrated = [row for row in data["events"] if row["event_type"] == "product_migrated"]
    renewed = [row for row in data["events"] if row["event_type"] == "product_renewed"]
    repriced = [row for row in data["events"] if row["event_type"] == "product_repriced"]
    changed = [row for row in data["events"] if row["event_type"] == "contract_terms_changed"]

    approvals = Counter(row["payload"].get("decision") for row in decided)
    rejects = Counter(
        row["payload"].get("reject_reason") for row in decided if row["payload"].get("decision") == "rejected"
    )

    with_offer = sum(1 for row in submitted if row["payload"].get("offer_id"))

    versions = Counter()
    archived_versions = 0

    status_by_code_version = {}

    for row in data["products"]:
        status_by_code_version.setdefault(row["product_code"], []).append(row)

    for row in opened:
        payload = row["payload"]
        versions[f"{payload.get('product_code')} v{payload.get('product_version')}.{payload.get('tariff_version')}"] += 1

    for row in data["events"]:
        if row["event_type"] not in ("installment_due", "loan_payment", "interest_credit"):
            continue

    return {
        "applications": len(submitted),
        "applications_from_offer_share": round(with_offer / len(submitted), 4) if submitted else None,
        "decisions": dict(approvals),
        "approval_rate": round(approvals.get("approved", 0) / len(decided), 4) if decided else None,
        "reject_reasons": dict(rejects.most_common()),
        "opened_by_family": dict(Counter(row["payload"].get("product_family") for row in opened).most_common()),
        "opened_by_version": dict(versions.most_common(12)),
        "migrations": len(migrated),
        "renewals": len(renewed),
        "repriced": len(repriced),
        "terms_changed": len(changed),
        "catalog_rows": len(data["products"]),
        "synthetic_products": len({row["product_code"] for row in data["products"] if row["is_synthetic"]}),
        "unresolved_sources": [
            {"product_code": row["product_code"], "confidence": row["confidence"], "note": row["note"]}
            for row in data["products"]
            if row["unresolved_source"]
        ],
    }


def _fraud(data: dict) -> dict:

    alerts = [row for row in data["events"] if row["event_type"] == "fraud_alert"]
    decisions = [row for row in data["events"] if row["event_type"] == "fraud_decision"]
    blocks = [row for row in data["events"] if row["event_type"] == "card_blocked"]
    unblocks = [row for row in data["events"] if row["event_type"] == "card_unblocked"]
    reissues = [row for row in data["events"] if row["event_type"] == "card_reissued"]
    chargebacks = [row for row in data["events"] if row["event_type"] == "chargeback"]

    episodes = Counter(row["key"] for row in data["truth_events"] if row["kind"] == "fraud_episode")

    chains = 0

    by_cause = defaultdict(list)

    for row in data["events"]:
        cause = row["payload"].get("cause_event_id") or row.get("correlation_id")
        if cause:
            by_cause[cause].append(row["event_type"])

    for types in by_cause.values():
        if "fraud_decision" in types or "card_blocked" in types:
            chains += 1

    return {
        "episodes_planned": dict(episodes.most_common()),
        "alerts": len(alerts),
        "score_bands": dict(Counter(row["payload"].get("score_band") for row in alerts)),
        "decisions": dict(Counter(row["payload"].get("decision") for row in decisions)),
        "resolutions": dict(Counter(row["payload"].get("resolution") for row in decisions)),
        "card_blocked": len(blocks),
        "card_unblocked": len(unblocks),
        "card_reissued": len(reissues),
        "chargebacks": len(chargebacks),
        "linked_chains": chains,
        "block_reasons": dict(Counter(row["payload"].get("reason") for row in blocks)),
    }


def _defects(data: dict) -> dict:

    duplicates = 0
    corrections = 0
    missing = Counter()

    seen: dict[str, int] = Counter()

    for row in data["events"]:

        seen[row["event_id"]] += 1

        if row["event_version"] > 1:
            corrections += 1

        payload = row["payload"]

        for name, value in payload.items():
            if value is None:
                missing[f"{row['event_type']}.{name}"] += 1

    duplicates = sum(count - 1 for count in seen.values() if count > 1) - corrections

    coverage = Counter(row["coverage_status"] for row in data["coverage"])
    reasons = Counter(row["coverage_reason"] for row in data["coverage"] if row["coverage_reason"])

    return {
        "duplicates": max(0, duplicates),
        "corrections": corrections,
        "precision": dict(Counter(row["time_precision"] for row in data["events"])),
        "coverage_status": dict(coverage),
        "coverage_reason": dict(reasons),
        "test_accounts": sum(1 for row in data["truth_clients"] if row["is_test_account"]),
        "top_missing_fields": dict(missing.most_common(12)),
    }


def _finance(data: dict) -> dict:

    by_client = defaultdict(list)

    for row in data["events"]:
        by_client[row["client_id"]].append(row)

    # Строки, потерянные сбоем источника, в RAW отсутствуют, но
    # деньги по ним двигались. Без них разрыв наблюдаемой цепочки
    # выглядел бы сломанной арифметикой.
    truth_by_client = defaultdict(list)

    for row in data["truth_events"]:
        truth_by_client[row["client_id"]].append(row)

    unobserved = {
        client_id: invariants_module.unobserved_rows(rows)
        for client_id, rows in truth_by_client.items()
    }

    problems = invariants_module.check_all(by_client, unobserved)

    transfers = defaultdict(set)

    for row in data["events"]:
        if row["event_type"] in ("p2p_out", "p2p_in") and row["correlation_id"]:
            transfers[row["correlation_id"]].add(row["event_type"])

    paired = sum(1 for sides in transfers.values() if len(sides) == 2)

    corrected = sum(
        1
        for rows in by_client.values()
        for versions in invariants_module.versions_by_event(rows).values()
        if len({row["event_version"] for row in versions}) > 1
    )

    return {
        "checked_version": "last",
        "violations": len(problems),
        "violations_by_check": dict(Counter(item.check for item in problems)),
        "examples": [str(item) for item in problems[:5]],
        "unobserved_rows": sum(len(rows) for rows in unobserved.values()),
        "unobserved_rule": (
            "строка, потерянная сбоем источника, в RAW отсутствует, а остаток её учёл: "
            "разрыв наблюдаемой цепочки объясняется ею и нарушением не считается"
        ),
        "internal_transfers": len(transfers),
        "internal_transfers_paired": paired,
        "corrected_events": corrected,
        "declined_operations": sum(
            1 for row in data["events"] if row["payload"].get("status") == "declined"
        ),
    }


CREDIT_FAMILIES = ("cash_loan", "credit_card", "refinance", "installment")


def _correlation(pairs: list) -> float | None:
    """
    Связь скрытой черты с поведением, которым она управляет.
    """

    if len(pairs) < 10:
        return None

    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]

    n = len(pairs)

    mx, my = sum(xs) / n, sum(ys) / n

    sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys) / n) ** 0.5

    if sx == 0 or sy == 0:
        return None

    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n / sx / sy, 3)


def _behaviour(data: dict) -> dict:
    """
    Проверка, что черта действительно управляет своим
    поведением, а не украшает скрытую истину.
    """

    truth = {row["client_id"]: row for row in data["truth_clients"]}

    counts: dict[str, Counter] = defaultdict(Counter)
    outlets: dict[str, Counter] = defaultdict(Counter)
    sessions: dict[str, set] = defaultdict(set)

    for row in data["events"]:

        client = row["client_id"]
        kind = row["event_type"]
        payload = row["payload"]

        counts[client][kind] += 1

        if kind == "purchase" and payload.get("outlet_id") and payload.get("status") == "approved":
            outlets[client][payload["outlet_id"]] += 1

        if kind == "app_screen" and row["correlation_id"]:
            sessions[client].add(row["correlation_id"])

    def pairs(trait: str, value) -> list:
        return [
            (row[trait], value(row["client_id"]))
            for row in truth.values()
            if row.get(trait) is not None
        ]

    repeat: list = []

    for client, counter in outlets.items():
        total = sum(counter.values())
        if total < 20:
            continue
        again = total - len(counter)
        row = truth.get(client)
        if row is not None:
            repeat.append((row["trait_merchant_loyalty"], again / total))

    checks = {
        "impulsivity_purchases": _correlation(
            pairs("trait_spending_impulsivity", lambda c: counts[c]["purchase"])
        ),
        "digital_sessions": _correlation(
            pairs("trait_digital_affinity", lambda c: len(sessions[c]))
        ),
        # Дисциплина измеряется ДОЛЕЙ пропущенных взносов, а не
        # их числом: число зависит ещё и от того, сколько у
        # клиента кредитов, и это размывает связь.
        "discipline_missed": _correlation(
            [
                (
                    row["trait_financial_discipline"],
                    counts[row["client_id"]]["installment_missed"]
                    / counts[row["client_id"]]["installment_due"],
                )
                for row in truth.values()
                if counts[row["client_id"]]["installment_due"] >= 6
            ]
        ),
        "savings_deposit_topups": _correlation(
            pairs("trait_savings_propensity", lambda c: counts[c]["deposit_topup"])
        ),
        "credit_appetite_applications": _correlation(
            pairs("trait_credit_appetite", lambda c: counts[c]["application_submitted"])
        ),
        "sociality_transfers": _correlation(
            pairs("trait_sociality", lambda c: counts[c]["p2p_out"] + counts[c]["transfer_out"])
        ),
        "loyalty_repeat_outlets": _correlation(repeat),
    }

    strength = [abs(value) for value in checks.values() if value is not None]

    return {
        "correlations": checks,
        "weakest": round(min(strength), 3) if strength else None,
    }


def _holdings(data: dict) -> dict:
    """
    Сколько продуктов держит клиент и каких.
    """

    clients = {row["client_id"] for row in data["truth_clients"]}

    per_client: Counter = Counter()
    by_family: Counter = Counter()
    families_per_client: dict = defaultdict(Counter)

    for row in data["events"]:

        if row["event_type"] not in ("product_opened", "product_migrated"):
            continue

        family = row["payload"].get("product_family")

        per_client[row["client_id"]] += 1
        by_family[family] += 1
        families_per_client[row["client_id"]][family] += 1

    counts = sorted(per_client[client] for client in clients)

    holders: dict = {}

    for family in by_family:
        distribution = Counter(
            min(4, families_per_client[client][family])
            for client in clients
            if families_per_client[client][family] > 0
        )
        holders[family] = {str(key): value for key, value in sorted(distribution.items())}

    return {
        "contracts_per_client": _quantiles(counts),
        "by_family": dict(by_family.most_common()),
        "holders_by_count": holders,
    }


def _fraud_profile(data: dict) -> dict:
    """
    Частота и форма мошеннических эпизодов.
    """

    clients = {row["client_id"] for row in data["truth_clients"]}

    episodes = Counter(
        row["key"] for row in data["truth_events"] if row["kind"] == "fraud_episode"
    )

    months = max(1, len(_months_between(HISTORY_START, HISTORY_END)))

    alerts = [row for row in data["events"] if row["event_type"] == "fraud_alert"]

    return {
        "episodes": dict(episodes.most_common()),
        "episodes_per_client_year": round(
            sum(episodes.values()) / max(1, len(clients)) / (months / 12.0), 4
        ),
        "alert_subjects": dict(Counter(row["payload"].get("subject") for row in alerts)),
        "score_bands": dict(Counter(row["payload"].get("score_band") for row in alerts)),
        "materialised_as": dict(
            Counter(
                row["event_type"]
                for row in data["events"]
                if row["link_type"] == "fraud_episode"
            )
        ),
    }


def _hours(data: dict) -> dict:
    """
    Время суток и соблюдение часов работы точек.
    """

    outlets = {row["outlet_id"]: row for row in data["merchants"]}

    purchases = [
        row
        for row in data["events"]
        if row["event_type"] == "purchase" and row["payload"].get("status") == "approved"
    ]

    if not purchases:
        return {"night_share": None, "out_of_hours_share": None}

    night = sum(1 for row in purchases if row["event_time"].hour < 6)

    offline = 0
    outside = 0

    for row in purchases:

        outlet = outlets.get(row["payload"].get("outlet_id"))

        if outlet is None or outlet.get("is_online"):
            continue

        offline += 1

        hour = row["event_time"].hour

        if not (outlet["opening_hour"] <= hour < outlet["closing_hour"]):
            outside += 1

    return {
        "night_share": round(night / len(purchases), 4),
        "out_of_hours_share": round(outside / offline, 4) if offline else None,
        "offline_purchases": offline,
        "by_hour": dict(Counter(row["event_time"].hour for row in purchases)),
    }


def _credit(data: dict) -> dict:
    """
    Кредитный портфель: переходы просрочки, дисциплина платежей,
    одобрение и размер выдачи.

    Это первое, на что смотрит банковский аналитик, и первое,
    что выдаёт нереалистичную синтетику.
    """

    events = data["events"]

    clients = {row["client_id"] for row in data["truth_clients"]}

    income = {row["client_id"]: row.get("true_income") or 0 for row in data["truth_clients"]}

    # У заявки нет колонки семейства: она выводится из каталога
    # продуктов по product_id.
    family_by_product = {
        row["product_id"]: row["product_family"] for row in data["products"]
    }

    worst: dict[str, int] = defaultdict(int)
    borrowers: set = set()

    due = paid = missed = partial = 0
    autopay = manual = 0

    approved: Counter = Counter()
    decided: Counter = Counter()
    reject_reasons: Counter = Counter()

    ratios: list = []

    closed: Counter = Counter()

    topups = inbound = declined_payments = 0

    for row in events:

        kind = row["event_type"]
        payload = row["payload"]

        if kind == "schedule_created":
            borrowers.add(row["client_id"])

        elif kind == "delinquency_registered":
            value = payload.get("days_past_due") or 0
            worst[row["client_id"]] = max(worst[row["client_id"]], value)

        elif kind == "installment_due":
            due += 1

        elif kind == "installment_paid":
            paid += 1
            if (payload.get("amount_paid") or 0) < (payload.get("amount_due") or 0):
                partial += 1

        elif kind == "installment_missed":
            missed += 1

        elif kind == "loan_payment":
            if payload.get("status") == "declined":
                declined_payments += 1
            elif payload.get("channel") == "system":
                autopay += 1
            else:
                manual += 1

        elif kind == "application_decision":
            family = family_by_product.get(payload.get("product_id"))
            decided[family] += 1
            if payload.get("decision") == "approved":
                approved[family] += 1
            else:
                reject_reasons[payload.get("reject_reason")] += 1

        elif kind == "loan_disbursement" and payload.get("status") == "approved":
            base = income.get(row["client_id"]) or 0
            if base > 0:
                ratios.append((payload.get("amount") or 0) / base)

        elif kind == "loan_closed":
            closed[payload.get("reason")] += 1

        elif kind == "transfer_in":
            if payload.get("reason") == "topup_before_installment":
                topups += 1
            elif payload.get("reason") == "inbound":
                inbound += 1

    total = len(clients) or 1

    credit_decided = sum(decided[name] for name in CREDIT_FAMILIES)
    credit_approved = sum(approved[name] for name in CREDIT_FAMILIES)

    months = max(1, len(_months_between(HISTORY_START, HISTORY_END)))

    return {
        "borrowers": len(borrowers),
        "borrower_share": round(len(borrowers) / total, 4),
        "dpd_client_share": {
            f"dpd{level}": round(sum(1 for v in worst.values() if v >= level) / total, 4)
            for level in (1, 30, 60, 90)
        },
        "installments_due": due,
        "installments_paid": paid,
        "installments_partial": partial,
        "installment_missed_share": round(missed / due, 4) if due else None,
        "autopay_payments": autopay,
        "manual_payments": manual,
        "declined_payments": declined_payments,
        "approval_rate_total": round(sum(approved.values()) / sum(decided.values()), 4)
        if decided
        else None,
        "approval_rate_credit": round(credit_approved / credit_decided, 4)
        if credit_decided
        else None,
        "reject_reasons": dict(reject_reasons.most_common()),
        "loan_amount_to_income": _quantiles(ratios),
        "loans_closed": dict(closed.most_common()),
        "topups_before_installment": topups,
        "inbound_transfers_per_client_month": round(inbound / total / months, 4),
    }


def _calibration(data: dict, report: dict) -> dict:

    targets = data["manifest"].get("calibration_targets", [])

    measured = {
        "communications_per_client_month": _rate(report, "communication_sent"),
        "app_sessions_per_client_month": _sessions(data, report),
        "banner_ctr": _ctr(data),
        "events_per_client_month_mean": report["activity"]["all_events"].get("mean"),
        "events_per_client_month_p10": report["activity"]["all_events"].get("p10"),
        "events_per_client_month_p25": report["activity"]["all_events"].get("p25"),
        "events_per_client_month_median": report["activity"]["all_events"].get("p50"),
        "events_per_client_month_p75": report["activity"]["all_events"].get("p75"),
        "events_per_client_month_p90": report["activity"]["all_events"].get("p90"),
        "events_per_client_month_p95": report["activity"]["all_events"].get("p95"),
        "events_per_client_month_p99": report["activity"]["all_events"].get("p99"),
        "contracts_per_client_median": report["holdings"]["contracts_per_client"].get("p50"),
        "fraud_episodes_per_client_year": report["fraud_profile"]["episodes_per_client_year"],
        "night_purchase_share": report["hours"]["night_share"],
        "out_of_hours_pos_share": report["hours"]["out_of_hours_share"],
        "support_chat_share": _support_chat_share(data),
        "trait_behaviour_min_correlation": report["behaviour"]["weakest"],
        "dpd90_client_share": report["credit"]["dpd_client_share"]["dpd90"],
        "installment_missed_share": report["credit"]["installment_missed_share"],
        "approval_rate_credit": report["credit"]["approval_rate_credit"],
        "loan_amount_to_income_median": report["credit"]["loan_amount_to_income"].get("p50"),
        "inbound_transfers_per_client_month": report["credit"]["inbound_transfers_per_client_month"],
        "zero_month_share": report["activity"]["zero_month_share"],
        "no_client_action_month_share": report["activity"]["no_client_action_month_share"],
        "returned_by_window_end_share": report["absence"]["returned_by_window_end_share"],
        "pause_ongoing_at_window_end_share": report["absence"]["pause_ongoing_at_window_end_share"],
        "confirmed_closure_share": report["absence"]["confirmed_closure_share"],
        "segment_share_silent": report["activity"]["segments"].get("silent_0_2"),
        "segment_share_sleepy": report["activity"]["segments"].get("sleepy_3_15"),
        "segment_share_moderate": report["activity"]["segments"].get("moderate_16_60"),
        "segment_share_regular": report["activity"]["segments"].get("regular_61_130"),
        "segment_share_high": report["activity"]["segments"].get("high_131_250"),
        "segment_share_extreme": report["activity"]["segments"].get("extreme_250_plus"),
    }

    for channel in ("call", "sms", "push", "email"):
        measured[f"delivery_rate_{channel}"] = _delivery(data, channel)

    for domain, share in _app_domains(data).items():
        measured[f"app_domain_share_{domain}"] = share

    rows = []
    without_reference = []

    for target in targets:

        metric = target["metric"]

        value = measured.get(metric)

        if target["status"] == "no_reference":
            without_reference.append(metric)
            continue

        if value is None:
            rows.append(
                {
                    "metric": metric,
                    "status": "not_measured",
                    "kind": target["status"],
                    "source": target["source"],
                }
            )
            continue

        if target.get("value") is not None:
            reference = target["value"]
            tolerance = target.get("tolerance", 0.25)
            span = abs(reference) * tolerance
            inside = abs(value - reference) <= span
            rows.append(
                {
                    "metric": metric,
                    "measured": round(value, 4),
                    "reference": reference,
                    "tolerance": tolerance,
                    # Границы допуска печатаются готовыми, чтобы
                    # «в допуске» можно было проверить глазами.
                    "allowed": [round(reference - span, 4), round(reference + span, 4)],
                    "inside": inside,
                    "source": target["source"],
                    "confidence": target["confidence"],
                    "kind": target["status"],
                }
            )
        else:
            low, high = target.get("low"), target.get("high")
            inside = low is not None and high is not None and low <= value <= high
            rows.append(
                {
                    "metric": metric,
                    "measured": round(value, 4),
                    "band": [low, high],
                    "inside": inside,
                    "source": target["source"],
                    "kind": target["status"],
                }
            )

    # Эталон и гипотеза — РАЗНЫЕ вещи, и складывать их в один
    # счётчик нельзя. Эталон это число из отчёта банка, и выход
    # за допуск означает ошибку генератора. Гипотеза это полоса
    # из плана, ничем не подтверждённая: расхождение с ней
    # ошибкой не является и подгонки не требует.
    references = [row for row in rows if row.get("kind") == "reference"]
    hypotheses = [row for row in rows if row.get("kind") == "hypothesis"]

    return {
        "checks": rows,
        "references": {
            "total": len(references),
            "inside": sum(1 for row in references if row.get("inside")),
            "outside": sum(1 for row in references if row.get("inside") is False),
            "not_measured": sum(1 for row in references if row.get("status") == "not_measured"),
        },
        "hypotheses": {
            "total": len(hypotheses),
            "inside": sum(1 for row in hypotheses if row.get("inside")),
            "outside": sum(1 for row in hypotheses if row.get("inside") is False),
            "not_measured": sum(1 for row in hypotheses if row.get("status") == "not_measured"),
        },
        "metrics_without_reference": sorted(without_reference),
    }


def _app_domains(data: dict) -> dict:
    """
    Доля пользователей приложения, заходивших в домен.

    Отчёт банка считает именно охват домена среди тех, кто
    приложением вообще пользуется, а не долю операций.
    """

    # Знаменатель это ВСЕ клиенты, а не только те, кто открывал
    # приложение. Так считает отчёт банка: доля раздела auth
    # там 84.9 %, что совпадает с долей установивших приложение,
    # а не с долей внутри них.
    total = len({row["client_id"] for row in data["truth_clients"]})

    by_domain: dict[str, set] = defaultdict(set)

    for row in data["events"]:

        if row["event_type"] not in ("app_operation", "app_screen"):
            continue

        domain = row["payload"].get("domain")

        if domain:
            by_domain[domain].add(row["client_id"])

    if not total:
        return {}

    return {
        domain: round(len(clients) / total, 4)
        for domain, clients in by_domain.items()
    }


def _support_chat_share(data: dict) -> float | None:

    cases = [row for row in data["events"] if row["event_type"] == "case_opened"]

    if not cases:
        return None

    chat = sum(1 for row in cases if row["payload"].get("channel") == "chat")

    return round(chat / len(cases), 4)


def _rate(report: dict, event_type: str) -> float | None:

    months = report["activity"]["client_months"]

    if not months:
        return None

    return report["activity"]["events_by_type"].get(event_type, 0) / months


def _sessions(data: dict, report: dict) -> float | None:

    months = report["activity"]["client_months"]

    if not months:
        return None

    sessions = {
        row["correlation_id"]
        for row in data["events"]
        if row["event_type"] == "app_screen" and row["correlation_id"]
    }

    return len(sessions) / months


def _ctr(data: dict) -> float | None:

    shown = sum(1 for row in data["events"] if row["event_type"] == "banner_shown")
    clicked = sum(1 for row in data["events"] if row["event_type"] == "banner_clicked")

    return clicked / shown if shown else None


def _delivery(data: dict, channel: str) -> float | None:

    sent = [
        row
        for row in data["events"]
        if row["event_type"] == "communication_sent" and row["payload"].get("channel") == channel
    ]

    if not sent:
        return None

    delivered = sum(1 for row in sent if row["payload"].get("delivered"))

    return delivered / len(sent)


# Сколько записей просматривает поиск значений скрытых черт
# в payload. Имена ключей проверяются по всей ленте, значения —
# по выборке: разбор JSON каждой строки на большом наборе стоит
# дороже, чем даёт.
VALUE_SCAN_LIMIT = 50_000

# Счётчики, по которым ищутся proxy-утечки. Список объявлен
# явно: отчёт обязан сказать, ЧТО именно проверено.
PROXY_FEATURES = (
    "app_operations",
    "purchases",
    "transfers",
    "applications",
    "deposits",
    "delinquencies",
    "cash",
)

PROXY_TRUTH_FIELDS = frozenset({"activity_mode", "hcb_role", "life_stage"})


def _proxy(data: dict) -> dict:

    truth = {row["client_id"]: row for row in data["truth_clients"]}

    columns = list(data["events"][0].keys()) if data["events"] else []

    # Имена ключей дёшево собрать по ВСЕЙ ленте: пропустить
    # редкий тип события здесь означало бы пропустить утечку.
    payload_keys = {name for row in data["events"] for name in row["payload"]}

    names = leak_audit.forbidden_names(columns, payload_keys)

    value_sample = min(len(data["events"]), VALUE_SCAN_LIMIT)

    values = leak_audit.forbidden_values(data["events"], truth, sample=VALUE_SCAN_LIMIT)

    features: dict[str, dict] = defaultdict(dict)

    counts: dict[str, Counter] = defaultdict(Counter)

    for row in data["events"]:
        counts[row["client_id"]][row["event_type"]] += 1

    for client_id, counter in counts.items():
        features[client_id] = {
            "app_operations": counter.get("app_operation", 0),
            "purchases": counter.get("purchase", 0),
            "transfers": counter.get("transfer_out", 0) + counter.get("p2p_out", 0),
            "applications": counter.get("application_submitted", 0),
            "deposits": counter.get("deposit_topup", 0),
            "delinquencies": counter.get("delinquency_registered", 0),
            "cash": counter.get("cash_withdrawal", 0),
        }

    proxies = leak_audit.proxy_report(features, truth)

    return {
        "forbidden_names": names,
        "forbidden_values": values,
        "proxy_candidates": proxies[:15],
        "proxy_count": len(proxies),
        "scope": {
            "events_total": len(data["events"]),
            "names_scanned": len(data["events"]),
            "values_scanned": value_sample,
            "payload_keys_seen": len(payload_keys),
            "features_checked": sorted(PROXY_FEATURES),
            "truth_fields_checked": sorted(
                {
                    name
                    for row in truth.values()
                    for name in row
                    if name.startswith("trait_") or name in PROXY_TRUTH_FIELDS
                }
            ),
        },
    }


# ============================================================
# СБОРКА
# ============================================================


def build_report(raw_dir: Path, stories: int = 6) -> dict:

    from .stories import build_stories

    data = _load(raw_dir)

    report = {
        "dataset": {
            "path": str(raw_dir),
            "clients": len(data["truth_clients"]),
            "events": len(data["events"]),
            "profile_versions": len(data["profile"]),
            "history_start": data["manifest"]["history_start"],
            "history_end": data["manifest"]["history_end"],
            "seed": data["manifest"]["seed"],
            "community_size": data["manifest"]["community_size"],
            "product_timeline_sha256": data["manifest"]["product_timeline_sha256"],
        }
    }

    data["windows"] = _windows(data)

    report["activity"] = _activity(data)
    report["absence"] = _absence(data)
    report["long_tails"] = _long_tails(data)
    report["repeatability"] = _repeatability(data)
    report["lifecycle"] = _lifecycle(data)
    report["income"] = _income(data)
    report["stress"] = _stress(data)
    report["products"] = _products(data)
    report["fraud"] = _fraud(data)
    report["defects"] = _defects(data)
    report["credit"] = _credit(data)
    report["holdings"] = _holdings(data)
    report["fraud_profile"] = _fraud_profile(data)
    report["hours"] = _hours(data)
    report["behaviour"] = _behaviour(data)
    report["finance"] = _finance(data)
    report["calibration"] = _calibration(data, report)
    report["leaks"] = _proxy(data)
    report["stories"] = build_stories(data, limit=stories)

    return report


def _table(rows: list, headers: list) -> str:

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    for row in rows:
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")

    return "\n".join(lines)


def render_markdown(report: dict) -> str:

    out: list[str] = []

    dataset = report["dataset"]

    out.append("# Отчёт реализма синтетического датасета")
    out.append("")
    out.append(
        _table(
            [
                ["клиентов", dataset["clients"]],
                ["событий", dataset["events"]],
                ["версий профиля", dataset["profile_versions"]],
                ["окно", f"{dataset['history_start']} .. {dataset['history_end']}"],
                ["seed", dataset["seed"]],
                ["размер сообщества", dataset["community_size"]],
                ["sha256 хронологии продуктов", dataset["product_timeline_sha256"][:16]],
            ],
            ["показатель", "значение"],
        )
    )

    activity = report["activity"]

    out.append("")
    out.append("## Активность на клиент-месяц")
    out.append("")
    out.append(
        f"Сетка клиент × месяц включает пустые месяцы: {activity['client_months']} полных "
        f"месяцев окна. Неполных месяцев на краях {activity['partial_months']}, они в "
        "распределения не входят: делить на половину месяца как на целый нельзя."
    )
    out.append("")
    out.append(f"Ограничение: {activity['coverage_note']}.")
    out.append("")
    out.append("### Три вида месяца")
    out.append("")
    out.append(
        f"Знаменатель здесь строже: {activity['classified_months']} месяцев, где известны "
        "ВСЕ источники. Иначе «записей не было вовсе» не отличить от «часть записей не "
        f"наблюдалась». Ещё {activity['unknown_months']} месяцев остались НЕИЗВЕСТНЫМИ — "
        "это не ноль, и в знаменатель они не входят."
    )
    out.append("")
    out.append("| вид месяца | доля |")
    out.append("|---|---|")
    out.append(f"| нет никаких записей | {activity['zero_month_share']} |")
    out.append(f"| записи есть, действий клиента нет | {activity['bank_only_month_share']} |")
    out.append(f"| есть действия клиента | {activity['client_action_month_share']} |")
    out.append(f"| **сумма трёх долей** | **{activity['month_kinds_sum']}** |")
    out.append("")
    out.append(
        "Виды не пересекаются и делят одно целое. Первый говорит, что о клиенте "
        "не написал никто, включая начисления и рассылки. Второй говорит, что банк "
        "работал, а клиент молчал."
    )
    out.append("")
    out.append(
        "Общий показатель месяцев БЕЗ ДЕЙСТВИЙ КЛИЕНТА равен сумме первых двух: "
        f"{activity['no_client_action_month_share']} = {activity['zero_month_share']} + "
        f"{activity['bank_only_month_share']}."
    )
    out.append("")
    out.append("### События на клиент-месяц")
    out.append("")
    out.append(
        _table(
            [
                [name] + [stats.get(key) for key in ("mean", "p50", "p90", "p95", "p99", "max")]
                for name, stats in (
                    ("все события", activity["all_events"]),
                    ("действия клиента", activity["client_events"]),
                    ("банк и система", activity["bank_system_events"]),
                    ("внешние", activity["external_events"]),
                    ("активные клиенты", activity["active_client_months"]),
                    ("неактивные клиенты", activity["inactive_client_months"]),
                )
            ],
            ["класс", "среднее", "P50", "P90", "P95", "P99", "максимум"],
        )
    )
    out.append("")
    out.append(
        f"Активных клиентов {activity['active_clients']}, неактивных "
        f"{activity['inactive_clients']}. Активный это {activity['active_definition']}. "
        "Это условная группировка отчёта, а не свойство клиента: раздел исчезновений "
        "ею не пользуется и смотрит фактические действия по месяцам."
    )
    out.append("")
    out.append(
        "«Внешние» это не банк: зарплата от работодателя, перевод от родни, чужая "
        "рука мошенника. Раньше они молча складывались с банковскими."
    )

    out.append("")
    out.append("### Сегменты клиент-месяцев")
    out.append("")
    out.append(_table([[name, share] for name, share in activity["segments"].items()], ["сегмент", "доля"]))

    out.append("")
    out.append("### События по источникам")
    out.append("")
    out.append(_table([[name, count] for name, count in activity["events_by_source"].items()],
                      ["источник", "событий"]))

    tails = report["long_tails"]

    out.append("")
    out.append("## Длинные хвосты")
    out.append("")
    out.append(
        _table(
            [
                [name, stats.get("distinct"), stats.get("top10_share"), stats.get("singleton_share")]
                for name, stats in tails.items()
            ],
            ["измерение", "уникальных", "доля топ-10", "доля единичных"],
        )
    )

    repeat = report["repeatability"]

    out.append("")
    out.append("## Повторяемость мерчантов и контрагентов")
    out.append("")
    out.append(f"Доля покупок в трёх любимых точках: медиана {_round(repeat['top3_outlet_share'].get('p50'))}.")
    out.append(f"Доля переводов повторному контрагенту: медиана {_round(repeat['repeat_counterparty_share'].get('p50'))}.")

    life = report["lifecycle"]

    out.append("")
    out.append("## Жизненный цикл")
    out.append("")
    out.append(_table([[name, count] for name, count in life["final_states"].items()],
                      ["состояние на конец окна", "клиентов"]))
    out.append("")
    out.append(
        f"Запланированных пауз: {life['pauses']['count']}, из них с намерением вернуться "
        f"{life['pauses']['with_planned_return']}. Длина паузы: медиана "
        f"{life['pauses']['length_days'].get('p50')} дней. Это ПЛАН генератора, "
        "а не наблюдаемое поведение; факт считает раздел ниже."
    )

    absence = report["absence"]

    out.append("")
    out.append("## Исчезновение и возвращение")
    out.append("")
    out.append(
        f"Считается по наблюдаемым действиям клиента, по {absence['clients']} клиентам. "
        "Пауза это серия подряд идущих месяцев окна без действий клиента."
    )
    out.append("")
    out.append(
        "Стартовое молчание до первого действия исчезновением не считается: клиент ещё "
        f"не начал. Медиана такого молчания {absence['leading_silence_months'].get('p50')} "
        f"месяцев, клиентов без единого действия за всё окно {absence['clients_never_acted']}."
    )
    out.append("")
    out.append(
        _table(
            [
                [
                    name,
                    item["episodes"],
                    item["clients"],
                    item["client_share"],
                    item["returned"],
                    item["ongoing"],
                    item["closed"],
                    item["observation_broken"],
                ]
                for name, item in absence["buckets"].items()
            ],
            [
                "длина паузы, мес",
                "эпизодов",
                "клиентов",
                "доля клиентов",
                "вернулся",
                "длится на конец",
                "закрытие",
                "наблюдение прервано",
            ],
        )
    )
    out.append("")
    out.append(
        "Исход «наблюдение прервано» означает, что пауза упёрлась в месяц с недостаточным "
        "покрытием. Про такую паузу неизвестно ничего: клиент мог вернуться в невидимый "
        "месяц, мог закрыть отношения, мог продолжать молчать. В знаменатель доли "
        "возвращения она не входит, и склеивать через неизвестный месяц две паузы нельзя."
    )
    out.append("")
    out.append(
        "**Доля пауз, завершившихся возвращением К КОНЦУ НАБЛЮДЕНИЯ: "
        f"{absence['returned_by_window_end_share']}.** Это наблюдаемый результат на дату "
        "конца датасета, а не вероятность возвращения: сроки наблюдения у клиентов разные, "
        "а продолжающаяся пауза ещё может закончиться возвратом. Ещё "
        f"{absence['pause_ongoing_at_window_end_share']} пауз на эту дату продолжаются."
    )
    out.append("")
    out.append(
        f"Подтверждённо прекратили отношения с банком {absence['confirmed_closure_clients']} "
        f"клиентов ({absence['confirmed_closure_share']}). Считается только по факту: "
        "состояние closed_relationship на конец окна или закрытие в покрытии источников. "
        "Молчание на конце окна сюда не попадает никогда. Ещё "
        f"{absence['closed_once_but_returned']} клиентов когда-то закрывали отношения, но "
        "вернулись: отношений они не прекратили, и в это число не входят."
    )
    out.append("")
    out.append(
        f"После возвращения снова пользовались продуктами {absence['returned_and_used_products']} "
        "клиентов: открыли договор, подали заявку, заплатили взнос, пополнили вклад "
        "или расплатились картой."
    )
    out.append("")
    notes = absence["generator_pause_notes"]
    out.append(
        f"Диагностика генератора: записей pause_start {notes['pause_start']}, "
        f"pause_end {notes['pause_end']}. Порог там ДРУГОЙ — перерыв не менее "
        f"{notes['threshold_days']} дней между действиями клиента, тогда как раздел выше "
        "считает полные календарные месяцы. Совпадать эти числа не обязаны, и расхождение "
        "ошибкой не является."
    )

    income = report["income"]

    out.append("")
    out.append("## Доходы")
    out.append("")
    out.append(_table([[name, count] for name, count in income["payout_outcomes"].items()],
                      ["исход выплаты", "случаев"]))

    stress = report["stress"]

    out.append("")
    out.append("## Стресс и восстановление")
    out.append("")
    out.append(_table([[name, count] for name, count in stress["episodes"].items()],
                      ["причина эпизода", "случаев"]))
    out.append("")
    out.append(_table([[name, count] for name, count in stress["resolutions"].items()],
                      ["исход эпизода (план)", "случаев"]))
    out.append("")
    out.append(_table([[name, count] for name, count in stress["delinquency_milestones"].items()],
                      ["веха DPD", "случаев"]))
    out.append("")
    out.append(f"Просрочка погашена: {stress['arrears_cleared']}, реструктуризаций: {stress['restructured']}.")

    products = report["products"]

    out.append("")
    out.append("## Продуктовые цепочки")
    out.append("")
    out.append(
        _table(
            [
                ["заявок", products["applications"]],
                ["из них по предложению", products["applications_from_offer_share"]],
                ["одобрение", products["approval_rate"]],
                ["миграций", products["migrations"]],
                ["пролонгаций", products["renewals"]],
                ["смен тарифа действующим", products["repriced"]],
                ["смен условий действующим", products["terms_changed"]],
                ["строк каталога продуктов", products["catalog_rows"]],
                ["вымышленных продуктов", products["synthetic_products"]],
            ],
            ["показатель", "значение"],
        )
    )
    out.append("")
    out.append(_table([[name, count] for name, count in products["opened_by_family"].items()],
                      ["семейство", "договоров"]))

    if products["unresolved_sources"]:
        out.append("")
        out.append("### Записи хронологии, требующие подтверждения")
        out.append("")
        out.append(
            _table(
                [[row["product_code"], row["confidence"], row["note"]] for row in products["unresolved_sources"]],
                ["продукт", "уверенность", "примечание"],
            )
        )

    fraud = report["fraud"]

    out.append("")
    out.append("## Мошенничество и антифрод")
    out.append("")
    out.append(
        _table(
            [
                ["срабатываний", fraud["alerts"]],
                ["блокировок карты", fraud["card_blocked"]],
                ["разблокировок", fraud["card_unblocked"]],
                ["перевыпусков", fraud["card_reissued"]],
                ["возвратов по оспариванию", fraud["chargebacks"]],
                ["связанных цепочек", fraud["linked_chains"]],
            ],
            ["показатель", "значение"],
        )
    )

    defects = report["defects"]

    out.append("")
    out.append("## Дефекты источников")
    out.append("")
    out.append(f"Дублей: {defects['duplicates']}, исправлений: {defects['corrections']}, "
               f"тестовых аккаунтов: {defects['test_accounts']}.")
    out.append("")
    out.append(_table([[name, count] for name, count in defects["coverage_status"].items()],
                      ["статус покрытия", "строк"]))

    finance = report["finance"]

    out.append("")
    credit = report["credit"]

    holdings = report["holdings"]

    out.append("## Продукты на клиента")
    out.append("")
    out.append(
        _table(
            [[name, value] for name, value in holdings["contracts_per_client"].items()],
            ["квантиль", "договоров"],
        )
    )
    out.append("")
    out.append(_table(list(holdings["by_family"].items()), ["семейство", "договоров"]))
    out.append("")

    fraud = report["fraud_profile"]

    out.append("## Мошенничество: частота и форма")
    out.append("")
    out.append(
        f"Эпизодов на клиента в год: {fraud['episodes_per_client_year']}."
    )
    out.append("")
    out.append(_table(list(fraud["episodes"].items()), ["вид эпизода", "случаев"]))
    out.append("")
    out.append(
        _table(
            [
                ["предмет срабатывания", fraud["alert_subjects"]],
                ["полосы тревожности", fraud["score_bands"]],
                ["чем материализуется", fraud["materialised_as"]],
            ],
            ["показатель", "значение"],
        )
    )
    out.append("")

    hours = report["hours"]

    out.append("## Время покупок")
    out.append("")
    out.append(
        _table(
            [
                ["ночных покупок", hours["night_share"]],
                ["покупок вне часов работы точки", hours["out_of_hours_share"]],
                ["офлайновых покупок", hours["offline_purchases"]],
            ],
            ["показатель", "значение"],
        )
    )
    out.append("")

    behaviour = report["behaviour"]

    out.append("## Черты и поведение")
    out.append("")
    out.append(
        "Черта обязана управлять тем поведением, ради которого она "
        "существует. Значение около нуля означает, что черта украшает "
        "скрытую истину и ничего не решает."
    )
    out.append("")
    out.append(
        _table(
            list(behaviour["correlations"].items()),
            ["черта и поведение", "корреляция"],
        )
    )
    out.append("")

    out.append("## Кредитный портфель")
    out.append("")
    out.append(
        f"Заёмщиков {credit['borrowers']} ({credit['borrower_share']} от всех клиентов). "
        f"Платежей к сроку {credit['installments_due']}, оплачено {credit['installments_paid']}, "
        f"из них частично {credit['installments_partial']}."
    )
    out.append("")
    out.append(
        _table(
            [[f"DPD {level}+", credit["dpd_client_share"][f"dpd{level}"]] for level in (1, 30, 60, 90)],
            ["веха просрочки", "доля клиентов"],
        )
    )
    out.append("")
    out.append(
        _table(
            [
                ["доля пропущенных платежей", credit["installment_missed_share"]],
                ["платежей автосписанием", credit["autopay_payments"]],
                ["платежей вручную", credit["manual_payments"]],
                ["неудачных автосписаний", credit["declined_payments"]],
                ["пополнений перед платежом", credit["topups_before_installment"]],
                ["одобрение по кредитным продуктам", credit["approval_rate_credit"]],
                ["одобрение по всем продуктам", credit["approval_rate_total"]],
                ["выдача к доходу, медиана", credit["loan_amount_to_income"].get("p50")],
                ["входящих переводов на клиент-месяц", credit["inbound_transfers_per_client_month"]],
            ],
            ["показатель", "значение"],
        )
    )
    out.append("")
    out.append(_table(list(credit["loans_closed"].items()), ["закрытие кредита", "случаев"]))
    out.append("")
    out.append(_table(list(credit["reject_reasons"].items()), ["причина отказа", "случаев"]))
    out.append("")

    out.append("## Финансовые инварианты")
    out.append("")
    out.append(f"Нарушений: {finance['violations']}.")
    out.append("")
    out.append(
        "Проверяется ПОСЛЕДНЯЯ версия каждой записи, на месте первой. "
        "Именно её читает потребитель данных. Первая версия исправленной "
        "записи намеренно расходится с проводками: это и есть та ошибка "
        "витрины, ради которой появилось исправление."
    )
    out.append("")
    out.append(f"{finance['unobserved_rule'].capitalize()}.")
    out.append("")
    out.append(
        _table(
            [
                ["внутренних переводов", finance["internal_transfers"]],
                ["из них парных", finance["internal_transfers_paired"]],
                ["исправленных записей", finance["corrected_events"]],
                ["отклонённых операций", finance["declined_operations"]],
                ["строк потеряно наблюдением", finance["unobserved_rows"]],
            ],
            ["показатель", "значение"],
        )
    )

    if finance["examples"]:
        out.append("")
        for example in finance["examples"]:
            out.append(f"- {example}")

    calibration = report["calibration"]

    references = [row for row in calibration["checks"] if row.get("kind") == "reference"]
    hypotheses = [row for row in calibration["checks"] if row.get("kind") == "hypothesis"]

    out.append("")
    out.append("## Сверка с внешними ориентирами")
    out.append("")
    out.append(
        "Ориентиры бывают двух РАЗНЫХ видов, и складывать их в один счётчик нельзя. "
        "Эталон это число из отчёта банка: выход за допуск означает ошибку генератора. "
        "Гипотеза это полоса из плана, ничем не подтверждённая: расхождение с ней "
        "ошибкой не является и подгонки не требует."
    )

    counts = calibration["references"]

    out.append("")
    out.append("### Реальные эталоны банка")
    out.append("")
    out.append(
        f"Всего {counts['total']}, в допуске {counts['inside']}, вне допуска {counts['outside']}."
    )
    out.append("")
    out.append(
        _table(
            [
                [
                    row["metric"],
                    row.get("measured"),
                    row.get("reference"),
                    f"±{int(round(row['tolerance'] * 100))}%" if row.get("tolerance") else None,
                    f"{row['allowed'][0]} … {row['allowed'][1]}" if row.get("allowed") else None,
                    "да" if row.get("inside") else "нет",
                    row.get("confidence"),
                ]
                for row in references
            ],
            ["метрика", "измерено", "эталон", "допуск", "интервал допуска", "в допуске", "уверенность"],
        )
    )

    counts = calibration["hypotheses"]

    out.append("")
    out.append("### Гипотезы (НЕ подтверждены данными банка)")
    out.append("")
    out.append(
        f"Всего {counts['total']}, внутри полосы {counts['inside']}, вне полосы "
        f"{counts['outside']}. Это ориентиры из «плана для генератора», раздел 17.2. "
        "Реального эталона у них нет, поэтому выход за полосу сам по себе ошибкой "
        "не является и подгонять под него генератор не следует."
    )
    out.append("")
    out.append(
        _table(
            [
                [
                    row["metric"],
                    row.get("measured"),
                    f"{row['band'][0]} … {row['band'][1]}" if row.get("band") else None,
                    "да" if row.get("inside") else "нет",
                    row.get("source"),
                ]
                for row in hypotheses
            ],
            ["метрика", "измерено", "полоса гипотезы", "внутри полосы", "откуда полоса"],
        )
    )

    out.append("")
    out.append("### Метрики без реального эталона")
    out.append("")
    out.append(", ".join(calibration["metrics_without_reference"]))

    leaks = report["leaks"]

    out.append("")
    out.append("## Аудит утечек")
    out.append("")
    scope = leaks.get("scope", {})

    out.append(
        "В ВЫПОЛНЕННЫХ ПРОВЕРКАХ утечек не обнаружено."
        if not (leaks["forbidden_names"] or leaks["forbidden_values"] or leaks["proxy_count"])
        else "Проверки нашли следующее."
    )
    out.append("")
    out.append(
        "Формулировка осторожная намеренно: проверено ровно то, что "
        "перечислено ниже, и полноту она не доказывает."
    )
    out.append("")
    out.append("| проверка | охват | найдено |")
    out.append("|---|---|---|")
    out.append(
        f"| запрещённые имена колонок и ключей payload | вся лента, "
        f"{scope.get('events_total', 0)} записей, {scope.get('payload_keys_seen', 0)} ключей "
        f"| {len(leaks['forbidden_names'])} |"
    )
    out.append(
        f"| значения скрытых черт внутри payload | выборка "
        f"{scope.get('values_scanned', 0)} записей из {scope.get('events_total', 0)} "
        f"| {len(leaks['forbidden_values'])} |"
    )
    out.append(
        f"| proxy-утечка: взаимная информация | {len(scope.get('features_checked', []))} признаков × "
        f"{len(scope.get('truth_fields_checked', []))} скрытых полей "
        f"| {leaks['proxy_count']} |"
    )
    out.append("")
    out.append(
        "Проверенные признаки: " + ", ".join(scope.get("features_checked", [])) + "."
    )
    out.append(
        "Непроверенное: сочетания признаков, тексты шаблонов и коды кампаний, "
        "признаки уровня события, а также записи за пределами выборки значений."
    )

    if leaks["proxy_candidates"]:
        out.append("")
        out.append(
            _table(
                [
                    [row["trait"], row["feature"], row["mutual_information_bits"], row["share_of_entropy"]]
                    for row in leaks["proxy_candidates"]
                ],
                ["скрытая характеристика", "наблюдаемый признак", "взаимная информация, бит", "доля энтропии"],
            )
        )

    out.append("")
    out.append("## Истории клиентов")

    for story in report["stories"]:
        out.append("")
        out.append(f"### {story['title']}")
        out.append("")
        out.append(story["summary"])
        out.append("")
        for line in story["timeline"]:
            out.append(f"- {line}")

    out.append("")

    return "\n".join(out)


def main() -> None:

    parser = argparse.ArgumentParser(description="Отчёт реализма RAW-датасета")

    parser.add_argument("--raw", type=Path, default=RAW_DIR / "smoke")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--stories", type=int, default=6)

    args = parser.parse_args()

    report = build_report(args.raw, stories=args.stories)

    out = args.out or args.raw / "realism_report.md"

    out.write_text(render_markdown(report), encoding="utf-8")

    (out.with_suffix(".json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    activity = report["activity"]

    print(f"клиентов {report['dataset']['clients']}, событий {report['dataset']['events']}")
    print(f"на клиент-месяц: среднее {activity['all_events']['mean']}, "
          f"медиана {activity['all_events']['p50']}, "
          f"месяцев без записей {activity['zero_month_share']}, "
          f"без действий клиента {activity['no_client_action_month_share']} "
          f"(из них с записями банка {activity['bank_only_month_share']})")
    print(f"нарушений инвариантов: {report['finance']['violations']}")

    # Эталон и гипотеза считаются РАЗДЕЛЬНО. Один счётчик на обе
    # группы читался как противоречие: реальные эталоны все в
    # допуске, а общее число говорило обратное.
    references = report["calibration"]["references"]
    hypotheses = report["calibration"]["hypotheses"]

    print(f"эталонов банка: {references['total']}, в допуске {references['inside']}, "
          f"вне {references['outside']}")
    print(f"гипотез плана: {hypotheses['total']}, внутри полосы {hypotheses['inside']}, "
          f"вне {hypotheses['outside']} — расхождение с гипотезой ошибкой не является")

    absence = report["absence"]

    print(f"паузы: вернулись к концу наблюдения {absence['returned_by_window_end_share']}, "
          f"ещё длятся {absence['pause_ongoing_at_window_end_share']}, "
          f"подтверждённых закрытий {absence['confirmed_closure_clients']}")
    print(f"отчёт: {out}")


if __name__ == "__main__":
    main()
