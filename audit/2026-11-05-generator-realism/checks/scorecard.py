"""Scorecard реализма генератора: одна таблица метрик против целевых полос аудита 2026-11-05.

    python scorecard.py --raw <мир> --cache <кэш вне data/> [--save out.json] [--before before.json]

Мир — каталог выгрузки (events.parquet, profile.parquet, truth/, manifest.json): scratch-мир
`diagnose.py generate` или группа data/01_raw. Рассчитан на миры до ~2000 клиентов: транзакции
читаются целиком. Кэш строится flatten.py, если его нет.

Метрики рождений восстанавливают план жизни чистыми функциями генератора (draw_persona,
plan_events) — они верны только для мира, сгенерированного ТЕКУЩИМ кодом.
Номера находок (C1…, M…) — README аудита.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from flatten import flatten  # noqa: E402

DEBITS = ("purchase", "transfer_out", "p2p_out", "cash_withdrawal", "bill_payment")
INCOME = ("salary_credit", "pension_credit", "other_income_credit")


class World:

    def __init__(self, raw: Path, cache: Path):
        self.raw, self.cache = raw, cache
        if not (cache / "transactions").exists():
            flatten(raw, cache)
        self.manifest = json.loads((raw / "manifest.json").read_text())
        self.start = pd.Timestamp(self.manifest["period_start"][:19])
        self.end = pd.Timestamp(self.manifest["period_end"][:19])
        self.profile = pd.read_parquet(raw / "profile.parquet")
        asof = pd.Timestamp(self.end)
        self.profile["age"] = (asof - pd.to_datetime(self.profile.birth_date)).dt.days / 365.25

    def load(self, source: str, columns: list[str]) -> pd.DataFrame:
        tables = []
        for f in sorted((self.cache / source).glob("part-*.parquet")):
            schema = pq.read_schema(f)
            cols = [c for c in columns if c in schema.names]
            strings = [c for c in cols if pa.types.is_string(schema.field(c).type)]
            tables.append(pq.read_table(f, columns=cols, read_dictionary=strings))
        if not tables:
            return pd.DataFrame(columns=columns)
        df = pa.concat_tables(tables, promote_options="permissive").unify_dictionaries().to_pandas()
        for c in columns:
            if c not in df:
                df[c] = None
        return df

    def truth(self, name: str) -> pd.DataFrame:
        t = pd.read_parquet(self.raw / "truth" / f"{name}.parquet")
        # Лента — в фиксированном UTC+5 (config.TIMEZONE), не в tzdata Asia/Almaty.
        t["time"] = t.time.dt.tz_convert("UTC").dt.tz_localize(None) + pd.Timedelta(hours=5)
        return t


def metric(rows: list, key: str, value, target: str, ok: bool | None, note: str = "") -> None:
    rows.append({"key": key, "value": value, "target": target, "ok": ok, "note": note})


def compute(w: World) -> list[dict]:

    rows: list[dict] = []
    tx = w.load("transactions", ["client_id", "ts", "seq", "type", "status", "amount", "merchant_category",
                                 "merchant_name", "merchant_country", "original_currency", "original_amount",
                                 "counterparty", "account_id", "transfer_id", "is_subscription", "balance_after"])
    tx["day"] = tx.ts.dt.normalize()
    ok_tx = tx[tx.status == "approved"]
    p = ok_tx[ok_tx.type == "purchase"]

    # --- C1: идентичность ---
    f = p[p.merchant_country.notna() & (p.merchant_country != "KZ")].sort_values(["client_id", "ts"])
    if len(f):
        gap = f.groupby("client_id", observed=True).ts.diff().dt.total_seconds() / 86400
        trip = (gap.isna() | (gap > 3)).cumsum()
        single = f.groupby(trip).merchant_country.nunique().eq(1).mean()
        metric(rows, "C1 поездок в одной стране", round(single, 3), "≥ 0.95", single >= 0.95)

    tel = ok_tx[(ok_tx.type == "bill_payment") & (ok_tx.merchant_category == "telecom")]
    n = tel.groupby("client_id", observed=True).size()
    one = tel[tel.client_id.isin(n[n >= 6].index)].groupby("client_id", observed=True).merchant_name.nunique().eq(1).mean()
    metric(rows, "C1 один оператор связи (≥6 платежей)", round(float(one), 3), "≥ 0.85", one >= 0.85)

    # Цены сервиса в одном месяце: за два года тариф законно меняется
    # (повышение датой), а внутри месяца цен столько, сколько тарифов.
    subs = ok_tx[ok_tx.is_subscription == True].assign(ym=lambda x: x.ts.dt.to_period("M"))  # noqa: E712
    busy = subs.groupby("merchant_name", observed=True).size()
    subs = subs[subs.merchant_name.isin(busy[busy >= 20].index)]
    per = subs.groupby(["merchant_name", "ym"], observed=True).amount.nunique().groupby(level=0, observed=True).max()
    med = float(per.median()) if len(per) else float("nan")
    metric(rows, "C1 цен у сервиса подписки в месяце (медиана)", med, "≤ 4 (тарифы)", med <= 4)

    sal = ok_tx[ok_tx.type == "salary_credit"]
    named = float((sal.counterparty.astype(str) != "Employer").mean()) if len(sal) else float("nan")
    metric(rows, "C1 зарплата с названием работодателя", round(named, 3), "1.0", named == 1.0)

    # --- C2, C4, M10, M11: баги ---
    pe = w.load("product_events", ["client_id", "ts", "type", "product_id", "contract_id", "account_id", "reason", "rate"])
    with_acc = set(tx.loc[tx.account_id.notna(), "client_id"].astype(str)) | set(pe.loc[pe.account_id.notna(), "client_id"].astype(str))
    no_acc = len(set(w.profile.client_id) - with_acc)
    metric(rows, "C2 клиентов без счёта", no_acc, "0", no_acc == 0)

    to = tx[tx.type == "transfer_out"]
    bad_ok = int(((to.status == "approved") & to.transfer_id.isna()).sum())
    metric(rows, "C4 одобренных transfer_out без transfer_id", bad_ok, "0", bad_ok == 0)
    dec = to[to.status == "declined"]
    share = float(dec.account_id.isna().mean()) if len(dec) else float("nan")
    metric(rows, "C4 отказов transfer_out без счёта", round(share, 3), "0", share == 0)

    loans = w.load("loans", ["client_id", "ts", "type", "contract_id", "principal_outstanding", "reason"])
    opened = pe[pe.type == "product_opened"]
    product_of = dict(zip(opened.contract_id.astype(str), opened.product_id.astype(str)))
    scheduled = set(loans.loc[loans.type == "schedule_created", "contract_id"].astype(str))
    loan_products = {product_of[c] for c in scheduled if c in product_of}
    lo = opened[opened.product_id.astype(str).isin(loan_products)]
    closed = set(pe.loc[pe.type == "product_closed", "contract_id"].astype(str))
    early = lo.ts < w.end - pd.Timedelta(days=2)
    ghosts = int((early & ~lo.contract_id.astype(str).isin(scheduled | closed)).sum())
    metric(rows, "M10 кредитов без выдачи и закрытия", ghosts, "0", ghosts == 0)
    rate = float(lo.rate.notna().mean()) if len(lo) else float("nan")
    metric(rows, "M12 ставка в открытии кредита", round(rate, 3), "1.0", rate == 1.0)

    # Порядок внутри одной секунды — по seq: вывод остатка стоит в ту
    # же секунду, что и последнее начисление.
    bal = tx[tx.balance_after.notna() & tx.account_id.notna()][["account_id", "ts", "seq", "balance_after"]]
    bal = bal.sort_values(["ts", "seq"]).drop_duplicates(["account_id", "ts"], keep="last").drop(columns="seq")
    cl = pe[(pe.type == "product_closed") & pe.account_id.notna()][["account_id", "ts"]].sort_values("ts")
    bal["account_id"] = bal.account_id.astype(str)
    cl["account_id"] = cl.account_id.astype(str)
    m = pd.merge_asof(cl, bal, on="ts", by="account_id", direction="backward")
    lost = m[m.balance_after > 0]
    metric(rows, "M11 счетов закрыто с остатком", len(lost), "0", len(lost) == 0, f"сумма {lost.balance_after.sum():,.0f}")

    li = loans.sort_values("ts").groupby("contract_id", observed=True).last()
    done = set(loans.loc[loans.type == "loan_closed", "contract_id"].astype(str)) | closed
    # Между вехами просрочки (180 → 270) законно до 90 дней без событий.
    silent = li[(li.ts < w.end - pd.Timedelta(days=100)) & (li.principal_outstanding > 0) & ~li.index.astype(str).isin(done)]
    metric(rows, "M13 кредитов молчат >100 дн при долге", len(silent), "0", len(silent) == 0)

    # --- C3: цена и payday ---
    s100 = float((p.amount == 100).mean())
    metric(rows, "C3 покупок ровно 100 ₸", round(s100, 4), "< 0.01", s100 < 0.01)

    days = (w.end - w.start).days
    cid = {c: i for i, c in enumerate(sorted(p.client_id.astype(str).unique()))}
    grid = np.zeros((len(cid), days + 1))
    di = (p.day - w.start).dt.days.to_numpy()
    ci = p.client_id.astype(str).map(cid).to_numpy()
    np.add.at(grid, (ci, di), p.amount.to_numpy())
    cum = np.concatenate([np.zeros((len(cid), 1)), grid.cumsum(axis=1)], axis=1)
    sd = sal.drop_duplicates(["client_id", "day"])
    sd = sd[sd.client_id.astype(str).isin(cid)]
    k = (sd.day - w.start).dt.days.to_numpy()
    r = sd.client_id.astype(str).map(cid).to_numpy()
    keep = (k >= 5) & (k + 6 <= days)
    k, r = k[keep], r[keep]
    after = (cum[r, k + 6] - cum[r, k + 1]).sum()
    before = (cum[r, k] - cum[r, k - 5]).sum()
    ratio = float(after / before) if before else float("nan")
    metric(rows, "C3 payday: траты +1…+5 к −5…−1", round(ratio, 2), "1.1–1.6", 1.1 <= ratio <= 1.6)

    # --- M1: распорядок ---
    def per_day(cat: str) -> pd.Series:
        return p[p.merchant_category == cat].groupby(["client_id", "day"], observed=True).size()
    t3 = float(per_day("transit").ge(3).mean())
    metric(rows, "M1 транспорт ≥3 в день (доля дней)", round(t3, 3), "< 0.02", t3 < 0.02)
    c2 = float(per_day("cinema").ge(2).mean())
    metric(rows, "M1 кино ≥2 в день (доля дней)", round(c2, 3), "< 0.01", c2 < 0.01)

    # --- C5: 1 января ---
    cnt = p.groupby("day").size()
    first = cnt.get(w.start, 0)
    base = cnt[(cnt.index > w.start) & (cnt.index < w.start + pd.Timedelta(days=30))].mean()
    jan = float(first / base) if base else float("nan")
    metric(rows, "C5 первый день окна к среднему", round(jan, 2), "≤ 1.5", jan <= 1.5)

    # --- C6: персона ---
    g = p.merge(w.profile[["client_id", "gender", "age"]], on="client_id")
    def share(mask_rows, cats):
        x = g[mask_rows]
        return x.merchant_category.isin(cats).mean()
    beauty = share(g.gender == "F", ["beauty", "cosmetics"]) / max(1e-9, share(g.gender == "M", ["beauty", "cosmetics"]))
    metric(rows, "C6 красота Ж/М", round(float(beauty), 2), "≥ 2", beauty >= 2)
    tob = share(g.gender == "M", ["tobacco"]) / max(1e-9, share(g.gender == "F", ["tobacco"]))
    metric(rows, "C6 табак М/Ж", round(float(tob), 2), "≥ 3", tob >= 3)
    old = g[g.age >= 65]
    gam = float(old.groupby("client_id").merchant_category.apply(lambda s: (s == "gambling").any()).mean()) if len(old) else float("nan")
    metric(rows, "C6 ставки у клиентов 65+", round(gam, 3), "≤ 0.05", gam <= 0.05)

    # --- M7: цены и курс ---
    p2 = p.assign(ym=p.ts.dt.to_period("M"))
    yoy = []
    for cat in ("grocery", "coffee", "fastfood", "pharmacy", "restaurant", "taxi"):
        med = p2[p2.merchant_category == cat].groupby("ym").amount.median()
        for ym, v in med.items():
            nxt = ym + 12
            if nxt in med.index:
                yoy.append(med[nxt] / v - 1)
    infl = float(np.median(yoy)) if yoy else float("nan")
    metric(rows, "M7 рост медианного чека за год", round(infl, 3), "0.05–0.15 (ИПЦ 8.6–12.3%)", 0.05 <= infl <= 0.15)
    # Только суммы от 100 единиц валюты: у мелкой покупки целый
    # original_amount даёт ошибку округления больше месячного хода курса.
    usd = p[(p.original_currency == "USD") & (p.original_amount >= 100)]
    if len(usd):
        rate = (usd.amount / usd.original_amount).groupby(usd.ts.dt.to_period("M")).median()
        cv = float(rate.std() / rate.mean())
        metric(rows, "M7 курс USD меняется по месяцам (CV)", round(cv, 3), "≥ 0.02 (НБК 2024–25 ≈ 0.05)", cv >= 0.02,
               f"медиана {rate.median():.1f}, мин {rate.min():.1f}, макс {rate.max():.1f}")

    # --- метки времени ---
    acts = tx[tx.type.isin(["purchase", "transfer_out", "p2p_out", "cash_withdrawal"])]
    ops = w.load("app_operations", ["ts"])
    sec0 = float(pd.concat([acts.ts, ops.ts]).dt.second.eq(0).mean())
    metric(rows, "M21 секунда :00 у действий клиента", round(sec0, 3), "≤ 0.05", sec0 <= 0.05)
    sup = w.load("support", ["ts", "type"])
    co = sup[sup.type == "case_opened"]
    s1320 = float(((co.ts.dt.hour == 13) & (co.ts.dt.minute == 20) & (co.ts.dt.second == 0)).mean()) if len(co) else float("nan")
    metric(rows, "M20 обращений ровно в 13:20:00", round(s1320, 3), "< 0.05", s1320 < 0.05)
    mid = tx[tx.type.isin(["cash_deposit", "transfer_in"]) & (tx.ts == tx.day)]
    metric(rows, "M22 пополнений ровно в 00:00:00", len(mid), "0", len(mid) == 0)
    swk = float(sal.ts.dt.weekday.ge(5).mean()) if len(sal) else float("nan")
    metric(rows, "M8 зарплат в субботу и воскресенье", round(swk, 3), "< 0.01", swk < 0.01)

    # --- M17: коллекшн ---
    comm = w.load("communications", ["client_id", "ts", "purpose"])
    if len(comm):
        launch = comm.ts.min()
        d1 = loans[(loans.type == "delinquency_registered") & (loans.reason == "dpd_1")
                   & (loans.ts >= launch) & (loans.ts <= w.end - pd.Timedelta(days=7))]
        col = comm[comm.purpose == "collection"][["client_id", "ts"]].rename(columns={"ts": "cts"})
        hit = d1.merge(col, on="client_id")
        hit = hit[(hit.cts >= hit.ts) & (hit.cts <= hit.ts + pd.Timedelta(days=7))]
        rem = len(set(zip(hit.client_id.astype(str), hit.ts))) / max(1, len(d1))
        metric(rows, "M17 напоминание за 7 дней после dpd_1", round(rem, 3), "≥ 0.8", rem >= 0.8)
        d30 = loans[(loans.type == "delinquency_registered") & (loans.reason == "dpd_30") & (loans.ts >= launch)]
        stop = loans[loans.type.isin(["arrears_cleared", "loan_closed", "loan_restructured"])][["contract_id", "ts"]]
        offers = comm[comm.purpose == "offer"]
        total = 0
        for row in d30.itertuples():
            ends = stop[(stop.contract_id == row.contract_id) & (stop.ts > row.ts)].ts
            end = min(ends.min() if len(ends) else w.end, row.ts + pd.Timedelta(days=60))
            total += int(((offers.client_id == row.client_id) & (offers.ts >= row.ts) & (offers.ts < end)).sum())
        metric(rows, "M17 офферов при DPD 30+ (60 дней)", total, "0", total == 0, f"эпизодов {len(d30)}")

    # --- M23: уход ---
    tr = w.truth("transitions")
    reg = tr[tr.component == "regime"].sort_values(["client_id", "time"])
    eps = []
    for c, x in reg.groupby("client_id"):
        start = None
        for t, v in zip(x.time, x.value):
            if v == "lapsed" and start is None:
                start = t
            elif v == "active" and start is not None:
                eps.append((c, start, t)); start = None
        if start is not None:
            eps.append((c, start, w.end))
    ep = pd.DataFrame(eps, columns=["client_id", "a", "b"])
    ep = ep[(ep.b - ep.a >= pd.Timedelta(days=60)) & (ep.a > w.start)]
    if len(ep):
        inc = ok_tx[ok_tx.type.isin(INCOME)][["client_id", "ts"]]
        deb = ok_tx[ok_tx.type.isin(DEBITS)][["client_id", "ts"]]
        def any_in(frame, c, a, b):
            x = frame[frame.client_id == c].ts
            return bool(((x >= a) & (x < b)).any())
        hoard = np.mean([any_in(inc, r.client_id, r.a, r.b) and not any_in(deb, r.client_id, r.a, r.b) for r in ep.itertuples()])
        metric(rows, "M23 уходов с доходом и без трат", round(float(hoard), 3), "↓ к «до»", None, f"уходов ≥60 дн: {len(ep)}")
        own = ok_tx[(ok_tx.type == "transfer_out") & (ok_tx.counterparty.astype(str) == "Own account")].copy()
        own["before"] = own.balance_after + own.amount
        own = own[own.amount >= 0.5 * own.before]
        req = pe[(pe.type == "product_closed") & (pe.reason == "client_request")][["client_id", "ts"]]
        exit_ = np.mean([any_in(own, r.client_id, r.a - pd.Timedelta(days=30), r.a + pd.Timedelta(days=1))
                         or any_in(req, r.client_id, r.a - pd.Timedelta(days=30), r.a + pd.Timedelta(days=1))
                         for r in ep.itertuples()])
        metric(rows, "M23 уходов с выводом остатка или закрытием", round(float(exit_), 3), "заметная доля", None)

    # --- M14: рождения (план жизни текущим кодом) ---
    rows += births(w)
    return rows


def births(w: World) -> list[dict]:
    rows: list[dict] = []
    try:
        sys.path.insert(0, str(REPO))
        from datetime import datetime
        from src.generator import config, params as params_module, rng as rng_module
        from src.generator.emit import _build_params
        from src.generator.life.persona import draw_persona
        from src.generator.life.events import plan_events
        m = w.manifest
        naive = lambda s: datetime.fromisoformat(s[:19])  # noqa: E731
        config.activate_horizon(naive(m["period_start"]), naive(m["period_end"]), naive(m["registration_end"]))
        settings = _build_params(None, None, m.get("community_size"))
        params_module.activate(settings)
        rng_module.configure(m["seed"], settings.fingerprint(), m["world_seed"])
        ids = set(w.profile.client_id)
        n = ages = old = short = 0
        years = 0.0
        for o in range(1, m["total_clients"] + 1):
            per = draw_persona(o)
            if per.client_id not in ids:
                continue
            years += (w.end - w.start).days / 365.25
            bs = sorted(e.ts for e in plan_events(per) if e.kind == "child_birth" and w.start <= pd.Timestamp(e.ts) < w.end)
            n += len(bs)
            short += sum((b - a).days < 270 for a, b in zip(bs, bs[1:]))
            if per.gender == "F":
                for b in bs:
                    ages += 1
                    old += (pd.Timestamp(b) - pd.Timestamp(per.birth_date)).days / 365.25 >= 45
        rate = 1000 * n / years
        metric(rows, "M14 рождений на 1000 клиент-лет", round(rate, 1), "15–25", 15 <= rate <= 25)
        share = old / ages if ages else float("nan")
        metric(rows, "M14 доля матерей 45+", round(share, 3), "< 0.02", share < 0.02)
        metric(rows, "M14 интервалов между рождениями < 270 дн", short, "0", short == 0)
    except Exception as error:  # noqa: BLE001 — метрика необязательна, причина печатается
        metric(rows, "M14 рождения", None, "—", None, f"не посчитано: {error}")
    return rows


def show(rows: list[dict], before: dict | None) -> None:
    head = f"{'метрика':48s} {'значение':>12s}" + (f" {'до':>10s}" if before else "") + f"  {'цель':26s} ok"
    print(head)
    print("-" * len(head))
    for r in rows:
        v = r["value"]
        b = f" {str(before.get(r['key'], '—')):>10s}" if before else ""
        mark = {True: "✓", False: "✗", None: "·"}[r["ok"]]
        print(f"{r['key']:48s} {str(v):>12s}{b}  {r['target']:26s} {mark} {r['note']}")
    print(f"\nпрошло {sum(r['ok'] is True for r in rows)} из {sum(r['ok'] is not None for r in rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--save", type=Path)
    ap.add_argument("--before", type=Path)
    a = ap.parse_args()
    for path in (a.cache, a.save):
        if path is not None and path.resolve().is_relative_to(REPO / "data"):
            raise SystemExit("кэш и отчёт не пишутся в data/")
    result = compute(World(a.raw.resolve(), a.cache.resolve()))
    before = None
    if a.before:
        before = {r["key"]: r["value"] for r in json.loads(a.before.read_text())}
    show(result, before)
    if a.save:
        a.save.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str))
