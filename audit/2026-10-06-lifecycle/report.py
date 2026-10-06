"""
Диагностика стадий CAPP на выгрузке генератора. Только читает RAW и
печатает таблицы; ничего не пишет.

    cd churn_baseline
    .venv/bin/python ../audit/2026-10-06-lifecycle/report.py --raw ../data/01_raw/val \
        --until 2026-05-01 --cutoff 2026-03-01
    # сверка горизонтов: та же группа, выгруженная до двух концов окна
    .venv/bin/python ../audit/2026-10-06-lifecycle/report.py --raw <short> --until <short_end> \
        --horizon <long>
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "churn_baseline"))

from churn import lifecycle as lc  # noqa: E402
from churn.activity import is_client_action  # noqa: E402
from churn.products import DEBIT_CARD, SERVICE, families  # noqa: E402
from churn.raw import client_blocks, read_profile  # noqa: E402
from churn.target import labels  # noqa: E402

LOCAL = timezone(timedelta(hours=5))


def local(text: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.fromisoformat(text).replace(tzinfo=LOCAL)).tz_convert("UTC")


def own(registered: dict, members: set[str]) -> dict:
    """
    Регистрации клиентов одного блока выгрузки.
    """
    return {client: moment for client, moment in registered.items() if client in members}


def build(raw: Path, until: pd.Timestamp, cutoff: pd.Timestamp | None):
    profile = read_profile(raw / "profile.parquet")
    registered = lc.registrations(profile)
    parts, label_parts, products = [], [], []
    catalog = families()
    seen_clients: set[str] = set()
    empty = None
    for block in client_blocks(raw / "events.parquet"):
        members = set(block["client_id"])
        seen_clients |= members
        empty = block.iloc[0:0]
        parts.append(lc.history(block, own(registered, members), until))
        if cutoff is not None:
            label_parts.append(labels(block, cutoff.to_pydatetime(), is_client_action(block)))
            seen = block[(block["t"] < cutoff) & block["contract_id"].notna() & block["product_id"].notna()]
            fam = seen.drop_duplicates(["client_id", "contract_id"])["product_id"].map(catalog)
            fam = fam[fam.notna() & (fam != SERVICE)]
            grouped = seen.loc[fam.index].assign(family=fam).groupby("client_id")["family"]
            products.append(pd.DataFrame({"contracts": grouped.size(), "debit": grouped.agg(lambda f: (f == DEBIT_CARD).any())}))
    # Зарегистрированные в приложении без единого события в выгрузке.
    silent = {client: moment for client, moment in registered.items() if client not in seen_clients}
    if silent and empty is not None:
        parts.append(lc.history(empty, silent, until))
    history = pd.concat(parts, ignore_index=True).sort_values(["client_id", "started_at"], kind="stable", ignore_index=True)
    target = pd.concat(label_parts) if label_parts else None
    owned = pd.concat(products) if products else None
    return profile, registered, history, target, owned


def spells(history: pd.DataFrame, until: pd.Timestamp) -> pd.DataFrame:
    """
    Отрезки стадий: начало, конец (следующий переход или until — тогда
    отрезок цензурирован) и следующая стадия.
    """
    frame = history.sort_values(["client_id", "started_at"], kind="stable").copy()
    nxt = frame.groupby("client_id")["started_at"].shift(-1)
    frame["next_stage"] = frame.groupby("client_id")["stage"].shift(-1)
    frame["censored"] = nxt.isna()
    frame["ended_at"] = nxt.fillna(until)
    frame["days"] = (frame["ended_at"] - frame["started_at"]) / pd.Timedelta(days=1)
    return frame


def quantiles(values: pd.Series) -> str:
    if values.empty:
        return "—"
    q = values.quantile([0.25, 0.5, 0.75, 0.9]).round(1).tolist()
    return f"p25 {q[0]}  p50 {q[1]}  p75 {q[2]}  p90 {q[3]}"


def report(raw: Path, until: pd.Timestamp, cutoff: pd.Timestamp | None) -> pd.DataFrame:

    profile, registered, history, target, owned = build(raw, until, cutoff)
    print(f"клиентов в анкете {len(profile)}, зарегистрированы в приложении {len(registered)}, "
          f"переходов {len(history)}")

    # --- распределение по месяцам ---
    print("\n## Распределение стадий на 1-е число, %")
    months = pd.date_range("2024-02-01", until.tz_convert(LOCAL).tz_localize(None), freq="MS")
    rows = {}
    for month in months[::3]:
        moment = local(month.strftime("%Y-%m-%d"))
        counts = lc.stage_at(history, moment)["stage"].value_counts()
        rows[month.strftime("%Y-%m")] = (100 * counts / counts.sum()).round(1)
    print(pd.DataFrame(rows).reindex(list(lc.STAGES)).fillna(0).to_string())

    # --- матрица переходов ---
    print("\n## Переходы: из (строки) в (столбцы)")
    moves = history[history["previous_stage"].notna()]
    print(pd.crosstab(moves["previous_stage"], moves["stage"]).reindex(index=list(lc.STAGES), columns=list(lc.STAGES)).fillna(0).astype(int).to_string())

    # --- длительности ---
    table = spells(history, until)
    print("\n## Дней в стадии (только завершённые отрезки)")
    for name in lc.STAGES:
        done = table[(table["stage"] == name) & ~table["censored"]]["days"]
        print(f"  {name:10s} n={len(done):5d}  {quantiles(done)}")

    # --- At Risk ---
    print("\n## At Risk: причина входа, прежняя стадия, исход")
    risk = table[table["stage"] == lc.AT_RISK].copy()
    risk["outcome"] = np.where(risk["censored"], "ещё в At Risk", np.where(risk["next_stage"] == lc.CHURN, "churn", "recovered"))
    print(pd.crosstab([risk["reason"], risk["previous_stage"]], risk["outcome"]).to_string())
    for outcome in ("recovered", "churn"):
        print(f"  дней до {outcome}: {quantiles(risk[risk['outcome'] == outcome]['days'])}")
    finished = risk[~risk["censored"]]
    print("  доля восстановившихся среди завершённых At Risk:",
          f"{(finished['outcome'] == 'recovered').mean():.2f}" if len(finished) else "—")

    # --- Churn ---
    print("\n## Вход в Churn: прежняя стадия; для прежней At Risk — причина At Risk")
    churn_rows = table[table["stage"] == lc.CHURN]
    print(churn_rows["previous_stage"].value_counts().to_string())
    before = table.assign(prev_reason=table.groupby("client_id")["reason"].shift(1))
    print(before[(before["stage"] == lc.CHURN) & (before["previous_stage"] == lc.AT_RISK)]["prev_reason"].value_counts().to_string())

    # --- стадия на T и метка churn_active90 ---
    if target is not None:
        print(f"\n## Стадия на T {cutoff.tz_convert(LOCAL).date()} и метка churn_active90 (маска G4)")
        rows = target[target["has_action_before"] & target["active90"]]
        at = lc.stage_at(history, cutoff)["stage"].reindex(rows.index).fillna("нет приложения")
        print(rows.groupby(at)["churn"].agg(["size", "sum", "mean"]).round(3).to_string())
        print(f"  всего: {len(rows)} строк, ушедших {int(rows['churn'].sum())} ({rows['churn'].mean():.4f})")

        print("\n## Продукты по событиям до T (недосчёт договоров до окна)")
        known = owned.reindex(list(registered)).fillna({"contracts": 0, "debit": False})
        print(f"  зарегистрированных: {len(known)}; видна дебетовая карта: {int(known['debit'].sum())}; "
              f"2+ договоров с дебетовой: {int(((known['contracts'] >= 2) & known['debit']).sum())}")

    return history


def leakage(raw: Path, history: pd.DataFrame, moments: list[pd.Timestamp]) -> None:
    """
    Та же история по ленте, обрезанной на M: переходы до M те же.
    Клиенты блока — по всей ленте: обрезка убирает события, а не клиентов.
    """
    registered = lc.registrations(read_profile(raw / "profile.parquet"))
    for moment in moments:
        parts, seen, empty = [], set(), None
        for block in client_blocks(raw / "events.parquet"):
            members = set(block["client_id"])
            seen |= members
            empty = block.iloc[0:0]
            parts.append(lc.history(block[block["t"] < moment], own(registered, members), moment))
        silent = {client: when for client, when in registered.items() if client not in seen}
        if silent and empty is not None:
            parts.append(lc.history(empty, silent, moment))
        short = pd.concat(parts, ignore_index=True).sort_values(["client_id", "started_at"], kind="stable", ignore_index=True)
        known = history[history["started_at"] <= moment].reset_index(drop=True)
        print(f"  обрезка на {moment.tz_convert(LOCAL)}: переходов {len(known)}, совпадают: {short.equals(known)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--until", required=True, help="конец выгрузки, местная дата")
    parser.add_argument("--cutoff", default=None, help="T для метки churn, местная дата")
    parser.add_argument("--leakage", nargs="*", default=[], help="моменты обрезки, местные даты")
    parser.add_argument("--horizon", type=Path, default=None, help="та же группа, выгруженная дальше")
    args = parser.parse_args()

    until = local(args.until)
    cutoff = local(args.cutoff) if args.cutoff else None
    history = report(args.raw, until, cutoff)

    if args.leakage:
        print("\n## Без будущего: обрезка ленты")
        leakage(args.raw, history, [local(text) for text in args.leakage])

    if args.horizon is not None:
        _, _, longer, _, _ = build(args.horizon, until, None)
        same = history.reset_index(drop=True).equals(longer.reset_index(drop=True))
        print(f"\n## Горизонт: переходы до {args.until} по длинной выгрузке совпадают: {same}")


if __name__ == "__main__":
    main()
