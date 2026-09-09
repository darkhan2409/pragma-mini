from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import (
    MAX_EVENTS_PER_HISTORY,
    PROFILE_FIELDS,
    RAW_DIR,
    SOURCES,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Сводка по готовому датасету: объёмы потоков, покрытие
# источников, доставка, воронка, длина ленты и, главное,
# ДЕФЕКТЫ. Если доля пропусков или совпадающих ts упала в ноль,
# синтетика перестала быть похожей на реальные данные.
# ============================================================


EVENT_TABLES = (
    "transactions",
    "product_events",
    "communications",
    "app_screens",
    "app_operations",
    "banners",
)


def read_manifest(raw_dir: Path) -> dict:
    return json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))


def read(raw_dir: Path, name: str, columns: list[str] | None = None):
    return pq.read_table(raw_dir / f"{name}.parquet", columns=columns)


def rows(raw_dir: Path, name: str) -> int:
    return int(pq.read_metadata(raw_dir / f"{name}.parquet").num_rows)


def shares(table, column: str) -> dict[str, float]:

    values = table.column(column)

    total = len(values)

    if total == 0:
        return {}

    counts = pc.value_counts(values)

    return {
        str(item["values"]): item["counts"] / total
        for item in sorted(
            counts.to_pylist(), key=lambda item: -item["counts"]
        )
    }


def null_share(table, column: str) -> float:

    total = len(table)

    if total == 0:
        return 0.0

    return float(table.column(column).null_count) / total


def print_shares(title: str, values: dict[str, float], limit: int = 12) -> None:

    print()
    print(title)

    for index, (name, value) in enumerate(values.items()):
        if index >= limit:
            print(f"  ... ещё {len(values) - limit} значений")
            break
        print(f"  {name:26s}{value:.4f}")


# ============================================================
# ОТЧЁТ
# ============================================================


def main() -> None:

    parser = argparse.ArgumentParser(description="Сводка по RAW-датасету")
    parser.add_argument("--raw", type=Path, default=RAW_DIR / "smoke")
    args = parser.parse_args()

    raw_dir: Path = args.raw

    manifest = read_manifest(raw_dir)

    total_clients = int(manifest["total_clients"])

    months = 24.0

    print("=" * 64)
    print(f"CALIBRATION  {raw_dir}")
    print("=" * 64)
    print()
    print(f"clients: {total_clients:,}")
    print()

    # --------------------------------------------------------
    # ОБЪЁМЫ
    # --------------------------------------------------------

    for name in EVENT_TABLES + ("profile",):
        count = rows(raw_dir, name)
        print(f"  {name:18s}{count:>12,}   {count / total_clients / months:7.2f} / client / month")

    # --------------------------------------------------------
    # ЛЕНТА
    # --------------------------------------------------------

    timeline = read(raw_dir, "timeline", ["client_id", "ts", "seq", "event_type"])

    lengths = np.bincount(
        timeline.column("client_id").to_numpy(zero_copy_only=False),
        minlength=total_clients,
    )

    print()
    print("timeline events per client")
    for percentile in (10, 50, 90, 99):
        print(f"  p{percentile:<3d}{np.percentile(lengths, percentile):10.0f}")
    print(f"  max {lengths.max():9d}   (лимит {MAX_EVENTS_PER_HISTORY})")
    print(f"  over limit: {int((lengths > MAX_EVENTS_PER_HISTORY).sum())} клиентов")

    print_shares("timeline by event_type", shares(timeline, "event_type"))

    # --------------------------------------------------------
    # СОВПАДАЮЩИЕ TS
    # --------------------------------------------------------
    #
    # Доля событий, у которых внутри клиента есть хотя бы одно
    # другое событие с тем же ts. Именно её разрешает tie-break.
    #
    # Ноль означает, что tie-break не проверяется ничем.
    # Слишком высокая доля означает, что совпадения созданы
    # искусственно и в реальных данных их столько не будет.
    # --------------------------------------------------------

    frame = timeline.select(["client_id", "ts", "event_type"]).to_pandas()

    duplicated = frame.duplicated(subset=["client_id", "ts"], keep=False)

    per_client = frame.groupby("client_id").apply(
        lambda group: group.duplicated(subset=["ts"], keep=False).mean(),
        include_groups=False,
    )

    print()
    print("совпадающие ts (их разрешает seq)")
    print(f"  доля событий           {duplicated.mean():.4f}")
    print(f"  по клиентам: медиана   {per_client.median():.4f}")
    print(f"               p90       {per_client.quantile(0.90):.4f}")
    print(f"               максимум  {per_client.max():.4f}")

    groups = frame[duplicated].groupby(["client_id", "ts"]).event_type.agg(
        lambda values: " + ".join(sorted(set(values)))
    )

    if len(groups):
        print()
        print("  из каких типов складываются")
        for combination, count in groups.value_counts().head(6).items():
            print(f"    {combination:42s}{count / len(groups):.3f}")

    # --------------------------------------------------------
    # ПОКРЫТИЕ
    # --------------------------------------------------------

    coverage = read(raw_dir, "source_coverage")

    print()
    print("source coverage")

    source_column = coverage.column("source").to_pylist()
    first_seen = coverage.column("first_seen").to_pylist()

    for source in SOURCES:
        seen = [
            value
            for value, name in zip(first_seen, source_column)
            if name == source
        ]
        covered = sum(1 for value in seen if value is not None)
        print(f"  {source:18s}covered {covered / max(1, len(seen)):.3f}")

    # --------------------------------------------------------
    # ТРАНЗАКЦИИ
    # --------------------------------------------------------

    transactions = read(raw_dir, "transactions")

    print_shares("transaction direction", shares(transactions, "direction"))
    print_shares("top mcc", shares(transactions, "mcc"), limit=8)

    print()
    print(f"is_online:        {pc.mean(transactions.column('is_online').cast('int8')).as_py():.4f}")
    print(f"is_subscription:  {pc.mean(transactions.column('is_subscription').cast('int8')).as_py():.4f}")
    print(f"merchant_city null: {null_share(transactions, 'merchant_city'):.4f}")
    print(f"foreign country:  {1 - shares(transactions, 'merchant_country').get('KZ', 0.0):.4f}")

    # --------------------------------------------------------
    # ДОГОВОРЫ
    # --------------------------------------------------------

    products = read(raw_dir, "product_events")

    print_shares("product_type", shares(products, "product_type"))
    print_shares("timestamp_quality", shares(products, "timestamp_quality"))

    print()
    print(f"term null (карты):  {null_share(products, 'term'):.4f}")

    # --------------------------------------------------------
    # КОММУНИКАЦИИ
    # --------------------------------------------------------

    communications = read(raw_dir, "communications")

    print_shares("communication channel", shares(communications, "channel"))

    print()
    print("delivered по каналам")

    channels = communications.column("channel").to_pylist()
    delivered = communications.column("delivered").to_pylist()

    for channel in sorted(set(channels)):
        pairs = [d for c, d in zip(channels, delivered) if c == channel]
        print(f"  {channel:18s}{sum(pairs) / max(1, len(pairs)):.4f}   n={len(pairs)}")

    print_shares("top template", shares(communications, "template"), limit=8)

    # --------------------------------------------------------
    # ПРИЛОЖЕНИЕ
    # --------------------------------------------------------

    screens = read(raw_dir, "app_screens")

    funnel = shares(screens, "funnel_stage")

    print_shares("funnel stage (доля экранов)", funnel)

    stages = screens.column("funnel_stage").to_pylist()

    applications = sum(1 for s in stages if s == "application")
    approved = sum(1 for s in stages if s == "approved")
    rejected = sum(1 for s in stages if s == "rejected")

    print()
    print("воронка заявок")

    # Клиенты, у которых воронка вообще была, и клиенты,
    # у которых есть приложение: без него заявка идёт офлайн.
    with_funnel = len(
        set(
            screens.filter(pc.is_valid(screens.column("funnel_stage")))
            .column("client_id")
            .to_pylist()
        )
    )

    app_users = sum(
        1
        for value, source in zip(first_seen, source_column)
        if source == "app_screens" and value is not None
    )

    print(f"  клиентов с воронкой    {with_funnel} из {total_clients} ({with_funnel / total_clients:.3f})")
    print(f"  из пользователей app   {with_funnel / max(1, app_users):.3f}")
    print(f"  заявок                 {applications} ({applications / total_clients:.2f} на клиента за 24 мес)")
    print(f"  одобрено / отказов     {approved} / {rejected}")

    if approved + rejected:
        print(f"  доля одобрений         {approved / (approved + rejected):.3f}")

    print_shares("reject_reason", shares(screens, "reject_reason"), limit=8)

    print()
    print(f"firebase_screen '(not set)': {shares(screens, 'firebase_screen').get('(not set)', 0.0):.4f}")

    operations = read(raw_dir, "app_operations")

    print_shares("operation domain", shares(operations, "domain"))
    print_shares("operation status", shares(operations, "status"))

    banners = read(raw_dir, "banners")

    banner_actions = shares(banners, "action")

    print_shares("banner action", banner_actions)

    if banner_actions.get("shown"):
        print()
        print(f"CTR: {banner_actions.get('clicked', 0.0) / banner_actions['shown']:.4f}")

    # --------------------------------------------------------
    # ПРОФИЛЬ
    # --------------------------------------------------------

    profile = read(raw_dir, "profile")

    print()
    print("profile null share по полям")

    for field in PROFILE_FIELDS:
        print(f"  {field:22s}{null_share(profile, field):.4f}")

    months_column = profile.column("snapshot_month").to_pylist()
    income = profile.column("declared_income").to_pylist()

    by_month: dict[str, list[int]] = {}

    for month, value in zip(months_column, income):
        by_month.setdefault(str(month)[:7], []).append(value is None)

    variation = {
        month: sum(values) / len(values) for month, values in sorted(by_month.items())
    }

    print()
    print("declared_income null share по месяцам (первые 12)")
    for month, value in list(variation.items())[:12]:
        print(f"  {month}  {value:.3f}")

    print()
    print(f"разброс между месяцами: {min(variation.values()):.3f} .. {max(variation.values()):.3f}")

    # --------------------------------------------------------
    # МЕТКА
    # --------------------------------------------------------

    labels = read(raw_dir, "labels")

    print()
    print(
        "product_open_90d: "
        f"{pc.mean(labels.column('product_open_90d').cast('int8')).as_py():.4f}"
    )


if __name__ == "__main__":
    main()
