from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .config import BASE_DIR, FEATURE_END, HISTORY_START, RAW_DIR, RUNS_DIR
from .v2.config import BILL_MCC_GROUP
from .version import V1, V2_1, manifest_version
from .world import DOMAIN_OPERATIONS


# ============================================================
# ИДЕЯ
# ============================================================
#
# Сравнение ДАННЫХ двух версий генератора на одних и тех же
# клиентах. Это не сравнение моделей: вклад в качество модели
# здесь не измеряется и не утверждается.
#
#     python -m src.generator.compare_versions \
#         --v1 data/raw/smoke --v2 data/raw/v2/smoke --clients 100
#
# Для набора на 1000 клиентов версия v1 берётся из готового
# data/raw/dev с фильтром client_id < 1000: клиенты независимы,
# поэтому это тот же самый v1-набор.
# ============================================================


EVENT_TABLES = (
    "transactions",
    "product_events",
    "communications",
    "app_screens",
    "app_operations",
    "banners",
)

MONTHS = (FEATURE_END.year - HISTORY_START.year) * 12 + FEATURE_END.month - HISTORY_START.month

OPERATION_DOMAIN = {
    operation: domain
    for domain, operations in DOMAIN_OPERATIONS.items()
    for operation in operations
}

FUNNEL_PREFIX = "s_a"


# ============================================================
# ЧТЕНИЕ
# ============================================================


def read_tables(raw: Path, clients: int | None) -> dict[str, pd.DataFrame]:

    tables: dict[str, pd.DataFrame] = {}

    for name in EVENT_TABLES:

        frame = pd.read_parquet(raw / f"{name}.parquet")

        if clients is not None:
            frame = frame[frame.client_id < clients]

        tables[name] = frame.reset_index(drop=True)

    return tables


def dataset_version(raw: Path) -> str:
    return manifest_version(json.loads((raw / "manifest.json").read_text(encoding="utf-8")))


# ============================================================
# МЕТРИКИ
# ============================================================


def volumes(tables: dict[str, pd.DataFrame], clients: int) -> dict[str, float]:
    """
    Строк на клиента в месяц: масштаб потоков.
    """

    return {
        name: round(len(frame) / clients / MONTHS, 3)
        for name, frame in tables.items()
    }


def sessions(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:

    screens = tables["app_screens"]

    browse = screens[~screens.firebase_screen.str.startswith(FUNNEL_PREFIX)]

    if browse.empty:
        return {"sessions": 0}

    per_session = browse.groupby(["client_id", "session_id"]).size()

    order = (
        browse.sort_values(["client_id", "session_id", "ts"])
        .groupby(["client_id", "session_id"])
        .firebase_screen.apply(tuple)
    )

    return {
        "sessions": int(len(per_session)),
        "screens_per_session": round(float(per_session.mean()), 2),
        "distinct_sequences": int(order.nunique()),
        "distinct_share": round(float(order.nunique() / len(order)), 3),
        "duration_p50_sec": int(
            browse.groupby(["client_id", "session_id"]).ts.apply(
                lambda values: (values.max() - values.min()).total_seconds()
            ).median()
        ),
    }


def domain_mix(tables: dict[str, pd.DataFrame]) -> dict[str, float]:
    """
    Состав действий: доля операций по доменам.
    """

    operations = tables["app_operations"]

    if operations.empty:
        return {}

    counts = operations.domain.value_counts(normalize=True)

    return {name: round(float(value), 4) for name, value in counts.items()}


def outcomes(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """
    Исходы операций: в целом и по типам, где это важно.
    """

    operations = tables["app_operations"]

    if operations.empty:
        return {}

    overall = operations.status.value_counts(normalize=True, dropna=False)

    families = {
        "read_only": ("card_view", "loan_view", "deposit_view", "market_browse"),
        "auth": ("login", "biometry_login"),
        "money": ("transfer_phone", "transfer_card", "pay_utility", "pay_mobile"),
        "abroad": ("transfer_abroad",),
    }

    by_family: dict[str, dict[str, float]] = {}

    for family, names in families.items():

        subset = operations[operations.operation.isin(names)]

        if subset.empty:
            continue

        share = subset.status.value_counts(normalize=True, dropna=False)

        by_family[family] = {
            str(key): round(float(value), 4) for key, value in share.items()
        }

    return {
        "overall": {str(k): round(float(v), 4) for k, v in overall.items()},
        "by_family": by_family,
        "distinct_operations": int(operations.operation.nunique()),
    }


def repeats(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """
    Повтор операции после неудачи в пределах 15 минут.

    Наблюдаемая оценка: намерения в RAW нет, поэтому считается
    та же операция того же клиента вскоре после сбоя.
    """

    operations = tables["app_operations"].sort_values(["client_id", "ts"])

    if operations.empty:
        return {}

    failed = 0
    retried = 0
    support_after = 0

    for _, group in operations.groupby("client_id", sort=False):

        rows = list(zip(group.ts, group.operation, group.status))

        for index, (ts, operation, status) in enumerate(rows):

            if status != "failed":
                continue

            failed += 1

            window = [
                (other_ts, other_op)
                for other_ts, other_op, _ in rows[index + 1 : index + 12]
                if other_ts - ts <= timedelta(minutes=15)
            ]

            if any(other_op == operation for _, other_op in window):
                retried += 1

            if any(
                OPERATION_DOMAIN.get(other_op) == "support" for _, other_op in window
            ):
                support_after += 1

    return {
        "failed": failed,
        "retry_share": round(retried / failed, 3) if failed else 0.0,
        "support_share": round(support_after / failed, 3) if failed else 0.0,
    }


def funnel(tables: dict[str, pd.DataFrame], clients: int) -> dict[str, Any]:

    screens = tables["app_screens"]

    stages = screens[screens.funnel_stage.notna()]

    contracts = tables["product_events"]

    applications = int((stages.funnel_stage == "application").sum())

    return {
        "applications_per_client": round(applications / clients, 3),
        "clients_with_funnel": int(stages.client_id.nunique()),
        "approved": int((stages.funnel_stage == "approved").sum()),
        "rejected": int((stages.funnel_stage == "rejected").sum()),
        "contracts_per_client": round(len(contracts) / clients, 3),
        "offline_share": round(
            1.0 - (applications / len(contracts)) if len(contracts) else 0.0, 3
        ),
    }


def habits(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """
    Привычные и новые покупки.
    """

    transactions = tables["transactions"]

    purchases = transactions[
        (transactions.direction == "debit") & (~transactions.is_subscription)
    ]

    if purchases.empty:
        return {}

    top_shares = []
    repeat_shares = []
    once_shares = []
    amount_repeats = []

    for _, group in purchases.groupby("client_id", sort=False):

        combos = Counter(
            zip(group.mcc, group.merchant_city.fillna("—"), group.is_online)
        )

        total = len(group)

        top_shares.append(sum(count for _, count in combos.most_common(10)) / total)
        repeat_shares.append(
            sum(count for count in combos.values() if count >= 3) / total
        )
        once_shares.append(sum(1 for count in combos.values() if count == 1) / total)
        amount_repeats.append(max(Counter(group.amount).values()))

    return {
        "top10_combo_share": round(float(pd.Series(top_shares).mean()), 3),
        "repeated_combo_share": round(float(pd.Series(repeat_shares).mean()), 3),
        "first_seen_combo_share": round(float(pd.Series(once_shares).mean()), 3),
        "max_amount_repeat_p50": int(pd.Series(amount_repeats).median()),
    }


def bills(tables: dict[str, pd.DataFrame], clients: int) -> dict[str, Any]:
    """
    Число и суммы платежей по счетам: связанные платежи должны
    ЗАМЕНЯТЬ независимые, а не добавляться к ним.
    """

    transactions = tables["transactions"]

    billed = transactions[
        transactions.mcc.isin(BILL_MCC_GROUP) & (~transactions.is_subscription)
    ]

    operations = tables["app_operations"]

    paid_in_app = 0

    if not operations.empty:
        paid_in_app = int(
            (
                operations.operation.isin(
                    ["pay_utility", "pay_mobile", "pay_internet", "pay_fine", "pay_tax"]
                )
                & (operations.status == "success")
            ).sum()
        )

    by_mcc = billed.mcc.value_counts()

    return {
        "per_client_month": round(len(billed) / clients / MONTHS, 3),
        "amount_p50": int(billed.amount.median()) if len(billed) else 0,
        "amount_total_per_client": int(billed.amount.sum() / clients) if len(billed) else 0,
        "successful_pay_operations": paid_in_app,
        "by_mcc": {str(k): int(v) for k, v in by_mcc.items()},
    }


def subscriptions(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:

    transactions = tables["transactions"]

    subs = transactions[transactions.is_subscription]

    if subs.empty:
        return {}

    started = stopped = changed = 0
    total = 0

    horizon_start = subs.ts.min()
    horizon_end = subs.ts.max()

    for _, group in subs.groupby(["client_id", "mcc"], sort=False):

        total += 1

        if group.ts.min() > horizon_start + pd.Timedelta(days=60):
            started += 1

        if group.ts.max() < horizon_end - pd.Timedelta(days=60):
            stopped += 1

        if group.amount.nunique() > 1:
            changed += 1

    return {
        "series": total,
        "started_inside": started,
        "stopped_inside": stopped,
        "amount_changed": changed,
        "cities_per_series_max": int(
            subs.groupby(["client_id", "mcc"]).merchant_city.nunique().max()
        ),
    }


def taxonomy(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """
    Поля, которые остаются производными, почти константными,
    случайными или редко заполненными.
    """

    screens = tables["app_screens"]
    operations = tables["app_operations"]

    def lookup(frame: pd.DataFrame, source: str, target: str) -> float:

        subset = frame[frame[source].notna() & frame[target].notna()]

        if subset.empty:
            return 0.0

        best = subset.groupby(source)[target].agg(lambda values: values.value_counts().iloc[0])

        return round(float(best.sum() / len(subset)), 4)

    result: dict[str, Any] = {
        "screen_to_funnel_stage": lookup(screens, "firebase_screen", "funnel_stage"),
        "operation_to_domain": lookup(operations, "operation", "domain"),
        "screen_to_product": lookup(screens, "firebase_screen", "product"),
    }

    if not operations.empty:
        share = operations.status.value_counts(normalize=True, dropna=False)
        result["status_top1_share"] = round(float(share.iloc[0]), 4)

    result["null_shares"] = {
        "app_screen.funnel_stage": round(float(screens.funnel_stage.isna().mean()), 4),
        "app_screen.reject_reason": round(float(screens.reject_reason.isna().mean()), 4),
        "app_operation.status": round(float(operations.status.isna().mean()), 4)
        if not operations.empty
        else 0.0,
        "transaction.merchant_city": round(
            float(tables["transactions"].merchant_city.isna().mean()), 4
        ),
    }

    return result


# ============================================================
# ОТЧЁТ
# ============================================================


def compare(v1: Path, v2: Path, clients: int) -> dict[str, Any]:

    left = read_tables(v1, clients)
    right = read_tables(v2, clients)

    def block(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
        return {
            "volumes": volumes(tables, clients),
            "sessions": sessions(tables),
            "domain_mix": domain_mix(tables),
            "outcomes": outcomes(tables),
            "repeats": repeats(tables),
            "funnel": funnel(tables, clients),
            "habits": habits(tables),
            "bills": bills(tables, clients),
            "subscriptions": subscriptions(tables),
            "taxonomy": taxonomy(tables),
        }

    return {
        "clients": clients,
        "months": MONTHS,
        "sources": {"v1": str(v1), "v2": str(v2)},
        "versions": {"v1": dataset_version(v1), "v2.1": dataset_version(v2)},
        V1: block(left),
        V2_1: block(right),
    }


def sequences(v1: Path, v2: Path, client_id: int, limit: int = 24) -> str:
    """
    Одна и та же сессия одного клиента до и после изменения правил.

    Сессии выровнены по session_id: расписание сессий в v2 то же,
    что в v1, меняется только их наполнение.
    """

    out: list[str] = []

    left = pd.read_parquet(v1 / "app_screens.parquet")
    right = pd.read_parquet(v2 / "app_screens.parquet")

    left = left[left.client_id == client_id]
    right = right[right.client_id == client_id]

    shared = sorted(set(left.session_id) & set(right.session_id))

    out.append(f"### Клиент {client_id}")
    out.append("")

    if not shared:
        out.append("Общих сессий нет: клиент не пользуется приложением.")
        return "\n".join(out)

    session_id = max(
        shared, key=lambda value: (right.session_id == value).sum()
    )

    for name, frame in (("v1", left), ("v2.1", right)):

        rows = frame[frame.session_id == session_id].sort_values("ts")

        out.append(f"**{name}**, сессия `{session_id}`, {len(rows)} экранов")
        out.append("")
        out.append("| время | экран | продукт |")
        out.append("|---|---|---|")

        for _, row in rows.head(limit).iterrows():
            product = row["product"] if isinstance(row["product"], str) else "—"
            out.append(f"| {row.ts:%H:%M:%S} | `{row.firebase_screen}` | {product} |")

        out.append("")

    return "\n".join(out)


def transaction_week(v1: Path, v2: Path, client_id: int, limit: int = 14) -> str:

    out = [f"### Клиент {client_id}: неделя транзакций", ""]

    for name, raw in (("v1", v1), ("v2.1", v2)):

        frame = pd.read_parquet(raw / "transactions.parquet")
        frame = frame[(frame.client_id == client_id) & (frame.direction == "debit")]
        frame = frame.sort_values("ts")

        if frame.empty:
            continue

        anchor = frame.ts.iloc[len(frame) // 2]

        window = frame[(frame.ts >= anchor) & (frame.ts < anchor + pd.Timedelta(days=7))]

        out.append(f"**{name}**, с {anchor:%Y-%m-%d}")
        out.append("")
        out.append("| время | MCC | сумма | город | онлайн |")
        out.append("|---|---|---|---|---|")

        for _, row in window.head(limit).iterrows():
            city = row.merchant_city if isinstance(row.merchant_city, str) else "—"
            out.append(
                f"| {row.ts:%m-%d %H:%M} | {row.mcc} | {row.amount:,} | {city} | "
                f"{'да' if row.is_online else 'нет'} |".replace(",", " ")
            )

        out.append("")

    return "\n".join(out)


def render(report: dict[str, Any]) -> str:

    left = report[V1]
    right = report[V2_1]

    out: list[str] = []

    out.append("# Генератор V1 и V2.1 на одних клиентах")
    out.append("")
    out.append(
        f"Клиентов: {report['clients']}, месяцев окна признаков: {report['months']}. "
        f"Источники: `{report['sources']['v1']}` и `{report['sources']['v2']}`."
    )
    out.append("")
    out.append(
        "Это сравнение ДАННЫХ. Вклад изменений в качество модели здесь "
        "не измерялся и не утверждается."
    )
    out.append("")

    out.append("## Объём потоков, строк на клиента в месяц")
    out.append("")
    out.append("| Поток | v1 | v2.1 |")
    out.append("|---|---:|---:|")

    for name in EVENT_TABLES:
        out.append(f"| {name} | {left['volumes'][name]} | {right['volumes'][name]} |")

    out.append("")

    out.append("## Состав действий")
    out.append("")
    out.append("| Показатель | v1 | v2.1 |")
    out.append("|---|---:|---:|")

    for label, key in (
        ("экранов на сессию", "screens_per_session"),
        ("различных последовательностей", "distinct_sequences"),
        ("их доля от числа сессий", "distinct_share"),
        ("медиана длительности сессии, с", "duration_p50_sec"),
    ):
        out.append(f"| {label} | {left['sessions'].get(key)} | {right['sessions'].get(key)} |")

    out.append(f"| различных операций | {left['outcomes'].get('distinct_operations')} "
               f"| {right['outcomes'].get('distinct_operations')} |")
    out.append("")

    domains = sorted(set(left["domain_mix"]) | set(right["domain_mix"]))

    out.append("| Домен операций | v1 | v2.1 |")
    out.append("|---|---:|---:|")

    for domain in domains:
        out.append(
            f"| {domain} | {left['domain_mix'].get(domain, 0.0)} "
            f"| {right['domain_mix'].get(domain, 0.0)} |"
        )

    out.append("")

    out.append("## Исходы операций")
    out.append("")
    out.append("| Группа | статус | v1 | v2.1 |")
    out.append("|---|---|---:|---:|")

    for status in ("success", "failed", "cancelled"):
        out.append(
            f"| все | {status} | {left['outcomes']['overall'].get(status, 0.0)} "
            f"| {right['outcomes']['overall'].get(status, 0.0)} |"
        )

    for family in ("read_only", "auth", "money", "abroad"):
        for status in ("success", "failed", "cancelled"):
            a = left["outcomes"]["by_family"].get(family, {}).get(status, 0.0)
            b = right["outcomes"]["by_family"].get(family, {}).get(status, 0.0)
            out.append(f"| {family} | {status} | {a} | {b} |")

    out.append("")

    out.append("| Реакция на сбой | v1 | v2.1 |")
    out.append("|---|---:|---:|")
    out.append(f"| сбоев | {left['repeats'].get('failed')} | {right['repeats'].get('failed')} |")
    out.append(
        f"| повтор той же операции за 15 мин | {left['repeats'].get('retry_share')} "
        f"| {right['repeats'].get('retry_share')} |"
    )
    out.append(
        f"| обращение в поддержку за 15 мин | {left['repeats'].get('support_share')} "
        f"| {right['repeats'].get('support_share')} |"
    )
    out.append("")

    out.append("## Привычки в транзакциях")
    out.append("")
    out.append("| Показатель | v1 | v2.1 |")
    out.append("|---|---:|---:|")

    for label, key in (
        ("доля 10 самых частых сочетаний (mcc, город, онлайн)", "top10_combo_share"),
        ("доля покупок в сочетаниях с 3+ повторами", "repeated_combo_share"),
        ("доля сочетаний, встреченных один раз", "first_seen_combo_share"),
        ("медиана максимального повтора точной суммы", "max_amount_repeat_p50"),
    ):
        out.append(f"| {label} | {left['habits'].get(key)} | {right['habits'].get(key)} |")

    out.append("")

    out.append("## Платежи по счетам")
    out.append("")
    out.append("| Показатель | v1 | v2.1 |")
    out.append("|---|---:|---:|")
    out.append(
        f"| платежей на клиента в месяц | {left['bills']['per_client_month']} "
        f"| {right['bills']['per_client_month']} |"
    )
    out.append(
        f"| медиана суммы | {left['bills']['amount_p50']} | {right['bills']['amount_p50']} |"
    )
    out.append(
        f"| сумма на клиента за окно | {left['bills']['amount_total_per_client']} "
        f"| {right['bills']['amount_total_per_client']} |"
    )
    out.append(
        f"| успешных pay_* операций | {left['bills']['successful_pay_operations']} "
        f"| {right['bills']['successful_pay_operations']} |"
    )
    out.append("")

    out.append("## Подписки")
    out.append("")
    out.append("| Показатель | v1 | v2.1 |")
    out.append("|---|---:|---:|")

    for label, key in (
        ("рядов (клиент, mcc)", "series"),
        ("начались внутри окна", "started_inside"),
        ("прекратились внутри окна", "stopped_inside"),
        ("сменили сумму", "amount_changed"),
        ("максимум городов на ряд", "cities_per_series_max"),
    ):
        out.append(
            f"| {label} | {left['subscriptions'].get(key)} | {right['subscriptions'].get(key)} |"
        )

    out.append("")

    out.append("## Воронка и договоры")
    out.append("")
    out.append("| Показатель | v1 | v2.1 |")
    out.append("|---|---:|---:|")

    for label, key in (
        ("заявок на клиента", "applications_per_client"),
        ("клиентов с воронкой", "clients_with_funnel"),
        ("одобрено", "approved"),
        ("отказано", "rejected"),
        ("договоров на клиента", "contracts_per_client"),
        ("доля офлайн-оформления", "offline_share"),
    ):
        out.append(f"| {label} | {left['funnel'].get(key)} | {right['funnel'].get(key)} |")

    out.append("")

    out.append("## Что осталось производным, почти константным и редким")
    out.append("")
    out.append("| Зависимость | v1 | v2.1 |")
    out.append("|---|---:|---:|")

    for label, key in (
        ("экран → стадия воронки", "screen_to_funnel_stage"),
        ("операция → домен", "operation_to_domain"),
        ("экран → продукт", "screen_to_product"),
        ("доля самого частого статуса", "status_top1_share"),
    ):
        out.append(f"| {label} | {left['taxonomy'].get(key)} | {right['taxonomy'].get(key)} |")

    out.append("")
    out.append("| Доля пропусков | v1 | v2.1 |")
    out.append("|---|---:|---:|")

    for key in sorted(left["taxonomy"]["null_shares"]):
        out.append(
            f"| {key} | {left['taxonomy']['null_shares'][key]} "
            f"| {right['taxonomy']['null_shares'][key]} |"
        )

    out.append("")

    return "\n".join(out)


# ============================================================
# CLI
# ============================================================


def main() -> None:

    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Сравнение генераторов v1 и v2")

    parser.add_argument("--v1", type=Path, default=RAW_DIR / "smoke")
    parser.add_argument("--v2", type=Path, default=RAW_DIR / "v2.1" / "smoke")
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--out", type=Path, default=RUNS_DIR / "v2_generator")
    parser.add_argument("--sequence-clients", type=int, nargs="*", default=[3, 11, 23])

    args = parser.parse_args()

    report = compare(args.v1, args.v2, args.clients)

    args.out.mkdir(parents=True, exist_ok=True)

    (args.out / "compare_v1_v2.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )

    (args.out / "compare_v1_v2.md").write_text(
        render(report), encoding="utf-8", newline="\n"
    )

    parts = ["# Последовательности одного клиента до и после", ""]

    for client_id in args.sequence_clients:
        parts.append(sequences(args.v1, args.v2, client_id))
        parts.append("")
        parts.append(transaction_week(args.v1, args.v2, client_id))
        parts.append("")

    (args.out / "sequences.md").write_text(
        "\n".join(parts), encoding="utf-8", newline="\n"
    )

    print(render(report))
    print(f"записано: {args.out}")


if __name__ == "__main__":
    main()
