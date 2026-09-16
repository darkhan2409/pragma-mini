from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq

from ..config import HISTORY_END, HISTORY_START, INITIATOR_CLIENT, INITIATOR_SYSTEM, RAW_DIR
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
# РАЗДЕЛЫ
# ============================================================


def _activity(data: dict) -> dict:
    """
    События на клиент-месяц с обязательными нулевыми месяцами.
    """

    events = data["events"]

    first_seen: dict[str, datetime] = {}

    for row in data["coverage"]:
        if row["source"] != "transactions" or row["first_seen"] is None:
            continue
        first_seen[row["client_id"]] = max(HISTORY_START, row["first_seen"])

    per_client_month: dict[tuple, Counter] = defaultdict(Counter)

    for row in events:
        key = (row["client_id"], _month(row["event_time"]))
        per_client_month[key]["all"] += 1
        if row["change_initiator"] == INITIATOR_CLIENT:
            per_client_month[key]["client"] += 1
        elif row["change_initiator"] == INITIATOR_SYSTEM:
            per_client_month[key]["system"] += 1
        else:
            per_client_month[key]["bank"] += 1

    totals: list[int] = []
    client_only: list[int] = []
    bank_only: list[int] = []
    system_only: list[int] = []

    zero_months = 0
    silent_client_months = 0
    grid = 0

    clients = {row["client_id"] for row in data["truth_clients"]}

    for client_id in sorted(clients):

        start = first_seen.get(client_id)

        if start is None:
            continue

        for month in _months_between(start, HISTORY_END):

            grid += 1

            counts = per_client_month.get((client_id, month), Counter())

            totals.append(counts["all"])
            client_only.append(counts["client"])
            bank_only.append(counts["bank"])
            system_only.append(counts["system"])

            # Два РАЗНЫХ показателя, и путать их нельзя.
            # «Пустой месяц» это месяц вообще без записей,
            # включая начисления и рассылки банка. «Месяц без
            # действий клиента» это месяц, где банк что-то
            # записал, а клиент не сделал ничего.
            if counts["all"] == 0:
                zero_months += 1

            if counts["client"] == 0:
                silent_client_months += 1

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

    return {
        "client_months": grid,
        "zero_month_share": round(zero_months / grid, 4) if grid else None,
        "no_client_action_month_share": (
            round(silent_client_months / grid, 4) if grid else None
        ),
        "all_events": _quantiles(totals),
        "client_events": _quantiles(client_only),
        "bank_events": _quantiles(bank_only),
        "system_events": _quantiles(system_only),
        "segments": {name: round(count / grid, 4) for name, count in segments.items()} if grid else {},
        "events_by_type": dict(Counter(row["event_type"] for row in data["events"]).most_common()),
        "events_by_source": dict(Counter(row["source"] for row in data["events"]).most_common()),
        "events_by_initiator": dict(Counter(row["change_initiator"] for row in data["events"])),
    }


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
    returns = 0

    for row in data["truth_events"]:

        if row["kind"] == "state_transition":
            transitions[row["key"]] += 1

        if row["kind"] == "pause_start":
            value = json.loads(row["value"])
            start = row["ts"]
            end = datetime.fromisoformat(value["actual_end"])
            pauses.append((end - start).days)
            if value.get("return_trigger") not in (None, "none"):
                returns += 1

    states = Counter(row["final_state"] for row in data["truth_clients"])

    return {
        "state_transitions": dict(transitions.most_common()),
        "final_states": dict(states.most_common()),
        "pauses": {
            "count": len(pauses),
            "length_days": _quantiles(pauses) if pauses else {},
            "with_return": returns,
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

    delays = defaultdict(list)
    duplicates = 0
    corrections = 0
    missing = Counter()

    seen: dict[str, int] = Counter()

    for row in data["events"]:

        if row["record_time"] is not None:
            delays[row["source"]].append(
                round((row["record_time"] - row["event_time"]).total_seconds() / 60.0, 1)
            )

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
        "record_delay_minutes": {
            source: _quantiles(values, points=(0.5, 0.9, 0.99))
            for source, values in sorted(delays.items())
        },
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

    problems = invariants_module.check_all(by_client)

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
        "internal_transfers": len(transfers),
        "internal_transfers_paired": paired,
        "corrected_events": corrected,
        "declined_operations": sum(
            1 for row in data["events"] if row["payload"].get("status") == "declined"
        ),
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
        "zero_month_share": report["activity"]["zero_month_share"],
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
            rows.append({"metric": metric, "status": "not_measured", "target": target})
            continue

        if target.get("value") is not None:
            reference = target["value"]
            tolerance = target.get("tolerance", 0.25)
            inside = abs(value - reference) <= abs(reference) * tolerance
            rows.append(
                {
                    "metric": metric,
                    "measured": round(value, 4),
                    "reference": reference,
                    "tolerance": tolerance,
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

    return {
        "checks": rows,
        "inside": sum(1 for row in rows if row.get("inside")),
        "outside": sum(1 for row in rows if row.get("inside") is False),
        "metrics_without_reference": sorted(without_reference),
    }


def _app_domains(data: dict) -> dict:
    """
    Доля пользователей приложения, заходивших в домен.

    Отчёт банка считает именно охват домена среди тех, кто
    приложением вообще пользуется, а не долю операций.
    """

    users: set = set()
    by_domain: dict[str, set] = defaultdict(set)

    for row in data["events"]:

        if row["event_type"] not in ("app_operation", "app_screen"):
            continue

        users.add(row["client_id"])

        domain = row["payload"].get("domain")

        if domain:
            by_domain[domain].add(row["client_id"])

    if not users:
        return {}

    return {
        domain: round(len(clients) / len(users), 4)
        for domain, clients in by_domain.items()
    }


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

    report["activity"] = _activity(data)
    report["long_tails"] = _long_tails(data)
    report["repeatability"] = _repeatability(data)
    report["lifecycle"] = _lifecycle(data)
    report["income"] = _income(data)
    report["stress"] = _stress(data)
    report["products"] = _products(data)
    report["fraud"] = _fraud(data)
    report["defects"] = _defects(data)
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
        f"Сетка клиент × месяц включает пустые месяцы: {activity['client_months']} строк."
    )
    out.append("")
    out.append("| показатель | доля |")
    out.append("|---|---|")
    out.append(
        f"| месяцев вообще без записей | {activity['zero_month_share']} |"
    )
    out.append(
        "| месяцев без действий клиента (записи банка есть) | "
        f"{activity['no_client_action_month_share']} |"
    )
    out.append("")
    out.append(
        "Это разные величины. Первая говорит, что о клиенте не написал "
        "никто, включая начисления и рассылки. Вторая говорит, что клиент "
        "сам ничего не делал, а банк продолжал работать."
    )
    out.append("")
    out.append(
        _table(
            [
                [name] + [stats.get(key) for key in ("p10", "p25", "p50", "mean", "p75", "p90", "p95", "p99", "max")]
                for name, stats in (
                    ("все события", activity["all_events"]),
                    ("клиентские", activity["client_events"]),
                    ("банковские", activity["bank_events"]),
                    ("системные", activity["system_events"]),
                )
            ],
            ["класс", "P10", "P25", "медиана", "среднее", "P75", "P90", "P95", "P99", "максимум"],
        )
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
    out.append(f"Пауз: {life['pauses']['count']}, из них с возвращением {life['pauses']['with_return']}. "
               f"Длина паузы: медиана {life['pauses']['length_days'].get('p50')} дней.")

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
                      ["исход эпизода", "случаев"]))
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
    out.append("## Дефекты и задержки источников")
    out.append("")
    out.append(
        _table(
            [[source, stats.get("p50"), stats.get("p90"), stats.get("p99")]
             for source, stats in defects["record_delay_minutes"].items()],
            ["источник", "медиана, мин", "P90", "P99"],
        )
    )
    out.append("")
    out.append(f"Дублей: {defects['duplicates']}, исправлений: {defects['corrections']}, "
               f"тестовых аккаунтов: {defects['test_accounts']}.")
    out.append("")
    out.append(_table([[name, count] for name, count in defects["coverage_status"].items()],
                      ["статус покрытия", "строк"]))

    finance = report["finance"]

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
    out.append(
        _table(
            [
                ["внутренних переводов", finance["internal_transfers"]],
                ["из них парных", finance["internal_transfers_paired"]],
                ["исправленных записей", finance["corrected_events"]],
                ["отклонённых операций", finance["declined_operations"]],
            ],
            ["показатель", "значение"],
        )
    )

    if finance["examples"]:
        out.append("")
        for example in finance["examples"]:
            out.append(f"- {example}")

    calibration = report["calibration"]

    out.append("")
    out.append("## Сравнение с калибровочными эталонами")
    out.append("")
    out.append(
        _table(
            [
                [
                    row["metric"],
                    row.get("measured"),
                    row.get("reference") if "reference" in row else row.get("band"),
                    "да" if row.get("inside") else "нет",
                    row.get("kind"),
                ]
                for row in calibration["checks"]
            ],
            ["метрика", "измерено", "эталон", "в допуске", "тип"],
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
          f"без действий клиента {activity['no_client_action_month_share']}")
    print(f"нарушений инвариантов: {report['finance']['violations']}")
    print(f"эталонов в допуске: {report['calibration']['inside']}, вне: {report['calibration']['outside']}")
    print(f"отчёт: {out}")


if __name__ == "__main__":
    main()
