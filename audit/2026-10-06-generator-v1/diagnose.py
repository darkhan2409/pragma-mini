"""
Диагностика Generator V1 на scratch-выгрузках. Выгрузки пишутся только в
каталог из --out (data/ запрещён); отчёт только читает и печатает.

    # мир seed: окно 2024-01-01 → 2026-09-01 и та же выгрузка до 2026-03-01
    .venv/bin/python audit/2026-10-06-generator-v1/diagnose.py generate --out <scratch> --seed 11
    # префикс: короткая выгрузка — начало длинной
    .venv/bin/python audit/2026-10-06-generator-v1/diagnose.py prefix --out <scratch> --seed 11
    # отчёт; маска действий клиента — churn_baseline (G4)
    churn_baseline/.venv/bin/python audit/2026-10-06-generator-v1/diagnose.py report \
        --raw <scratch>/seed11 [--before <old>/seed11] [--cohorts]

Отчёт:
  A–C  объём, сессии, деньги, договоры, время; полосы params/calibration.py
  D    пути по truth: стабильные, постепенный уход, резкий, возврат, внешние отлучки
  E    тишина ≥ 60 дней: с предвестниками и без, возвращение
  F–G  мошенничество и кредит
  Q    AUC признаков и скрытого состояния к будущей 60-дневной неактивности
  K    когорты: помесячная история клиента и отдельно скрытое состояние
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LOCAL = timedelta(hours=5)
DATA = (ROOT / "data").resolve()


def _outside_data(path: Path) -> Path:
    path = path.resolve()
    if path == DATA or DATA in path.parents:
        raise SystemExit(f"{path}: диагностика не пишет в data/")
    return path


# ============================================================
# ВЫГРУЗКА И ПРЕФИКС
# ============================================================


def generate(out: Path, seed: int, clients: int, start: str, end: str, short: str, workers: int) -> None:

    from src.generator import emit

    out = _outside_data(out)
    begin, finish, cut = (datetime.fromisoformat(item) for item in (start, end, short))

    for name, stop in ((f"seed{seed}", finish), (f"seed{seed}-short", cut)):
        moment = time.time()
        emit.generate_dataset(
            total_clients=clients, out_dir=out / name, seed=seed, world_seed=42,
            history_start=begin, history_end=stop, registration_end=finish,
            workers=workers, quiet=True,
        )
        print(f"{name}: {time.time() - moment:.0f} с", flush=True)


def prefix(out: Path, seed: int) -> None:

    long, short = out / f"seed{seed}", out / f"seed{seed}-short"

    boundary = json.loads((short / "manifest.json").read_text(encoding="utf-8"))["period_end"]
    cut = datetime.fromisoformat(boundary)

    def rows(directory: Path) -> dict[str, list]:
        table = pq.read_table(directory / "events.parquet")
        by = defaultdict(list)
        for client, when, source, payload in zip(
            *[table.column(name).to_pylist() for name in ("client_id", "event_time", "source", "payload")]
        ):
            if datetime.fromisoformat(when) < cut:
                by[client].append((when, source, payload))
        return by

    def truth(directory: Path, name: str) -> dict[str, list]:
        by = defaultdict(list)
        for row in pq.read_table(directory / "truth" / f"{name}.parquet").to_pylist():
            if row["time"] < cut:
                by[row["client_id"]].append(tuple(sorted(row.items())))
        return by

    left, right = rows(short), rows(long)
    diverged = sorted(client for client in left if left[client] != right.get(client))
    print(f"события до {boundary}: клиентов {len(left)}, разошлись {len(diverged)}")

    for name in ("transitions", "states"):
        a, b = truth(short, name), truth(long, name)
        bad = sorted(client for client in a if a[client] != b.get(client))
        print(f"truth/{name} до границы: клиентов {len(a)}, разошлись {len(bad)}")

    profile = {
        directory: {
            row["client_id"]: (
                [(item["type"], item["event_time"]) for item in row["lifelong"] if item["event_time"] < cut],
                [(item["start_date"], item["record_time"]) for item in row["employment"] if item["record_time"] < cut],
            )
            for row in pq.read_table(directory / "profile.parquet").to_pylist()
        }
        for directory in (short, long)
    }
    bad = [client for client, facts in profile[short].items() if facts != profile[long].get(client)]
    print(f"датированные факты анкеты: клиентов {len(profile[short])}, разошлись {len(bad)}")


# ============================================================
# ЧТЕНИЕ
# ============================================================


EXTRA = [
    ("topic", pa.string()),
    ("resolution", pa.string()),
]

MONEY = frozenset({
    "purchase", "cash_withdrawal", "cash_deposit", "transfer_out", "p2p_out", "transfer_in", "p2p_in",
    "bill_payment", "salary_credit", "pension_credit", "other_income_credit", "loan_payment",
    "deposit_topup", "deposit_withdrawal", "fee_charge", "refund", "reversal", "chargeback",
    "interest_credit", "cashback_credit", "loan_disbursement", "early_repayment",
})

CREDIT = frozenset({"cash_loan", "credit_card", "installment", "refinance"})


class World:
    """
    Выгрузка в помесячных и дневных счётчиках по клиентам — без
    таблицы всех строк в памяти.
    """

    CATEGORIES = (
        "all", "action", "money", "visit", "purchase", "decline", "decline_bank", "failed",
        "case", "complaint", "salary", "own_out", "comm", "inbound", "due", "missed",
        "credit_ok", "credit_no", "night", "product_opened", "fraud_alert",
    )

    def __init__(self, raw: Path, persona_info: bool = True):

        from churn.activity import is_client_action, is_visit
        from churn.products import families
        from churn.raw import PAYLOAD, client_blocks

        self.raw = raw
        self.manifest = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
        self.start = datetime.fromisoformat(self.manifest["period_start"]).replace(tzinfo=None)
        self.end = datetime.fromisoformat(self.manifest["period_end"]).replace(tzinfo=None)
        self.days = (self.end - self.start).days

        profile = pq.read_table(raw / "profile.parquet").to_pylist()
        self.clients = [row["client_id"] for row in profile]
        self.index = {client: position for position, client in enumerate(self.clients)}
        self.profile = {row["client_id"]: row for row in profile}

        size = (len(self.clients), self.days)
        self.daily = {name: np.zeros(size, dtype=np.int32) for name in self.CATEGORIES}
        self.cases_by_topic = Counter()
        self.first_action = np.full(len(self.clients), -1)

        product_family = families()
        schema = pa.schema(list(PAYLOAD) + [pa.field(name, kind) for name, kind in EXTRA])

        for block in client_blocks(raw / "events.parquet", schema):

            local = block["t"].dt.tz_convert(None) + LOCAL
            day = ((local - self.start).dt.days).to_numpy()
            keep = (day >= 0) & (day < self.days)
            row = block["client_id"].map(self.index).to_numpy()

            kind = block["type"]
            status = block["status"]
            masks = {
                "all": np.ones(len(block), dtype=bool),
                "action": is_client_action(block),
                "money": (kind.isin(MONEY) & (status != "declined")).to_numpy(),
                "visit": is_visit(block),
                "purchase": ((kind == "purchase") & (block["reason"] == "purchase") & (status == "approved")).to_numpy(),
                "decline": (status == "declined").fillna(False).to_numpy(),
                "decline_bank": ((status == "declined") & (block["decline_reason"] != "insufficient_funds")).fillna(False).to_numpy(),
                "failed": ((kind == "app_operation") & (status == "failed")).fillna(False).to_numpy(),
                "case": (kind == "case_opened").to_numpy(),
                "complaint": ((kind == "case_opened") & (block["topic"] == "complaint")).fillna(False).to_numpy(),
                "salary": (kind == "salary_credit").to_numpy(),
                "own_out": ((kind == "transfer_out") & (block["counterparty"] == "Own account") & (status == "approved")).fillna(False).to_numpy(),
                "comm": (kind == "communication_sent").to_numpy(),
                "inbound": ((kind == "transfer_in") & (block["reason"] == "inbound")).fillna(False).to_numpy(),
                "due": (kind == "installment_due").to_numpy(),
                "missed": (kind == "installment_missed").to_numpy(),
                "night": ((kind == "purchase") & (block["reason"] == "purchase") & (status == "approved")
                          & (local.dt.hour < 6)).fillna(False).to_numpy(),
                "product_opened": ((kind == "product_opened") & (block["reason"] == "application_approved")).fillna(False).to_numpy(),
                "fraud_alert": (kind == "fraud_alert").to_numpy(),
            }
            family = block["product_id"].map(product_family)
            credit = (kind == "application_decision") & family.isin(CREDIT)
            masks["credit_ok"] = (credit & (block["decision"] == "approved")).fillna(False).to_numpy()
            masks["credit_no"] = (credit & (block["decision"] == "rejected")).fillna(False).to_numpy()

            for name, mask in masks.items():
                chosen = mask & keep
                np.add.at(self.daily[name], (row[chosen], day[chosen]), 1)

            for topic in block.loc[kind == "case_opened", "topic"].fillna("-"):
                self.cases_by_topic[topic] += 1

            self.dpd90 = getattr(self, "dpd90", set())
            heavy = block[(kind == "delinquency_registered") & (block["days_past_due"].fillna(0) >= 90)]
            self.dpd90 |= set(heavy["client_id"])

        truth = raw / "truth"
        self.transitions = pq.read_table(truth / "transitions.parquet").to_pandas() if (truth / "transitions.parquet").exists() else None
        self.states = pq.read_table(truth / "states.parquet").to_pandas() if (truth / "states.parquet").exists() else None

        self.mode, self.joined = {}, {}
        if persona_info:
            self._personas()

    def _personas(self) -> None:
        """
        Режим активности и приход в банк — из персоны по seed выгрузки
        (аудит; во вход модели это не идёт).
        """

        from src.generator import config, emit
        from src.generator import params as params_module
        from src.generator import rng as rng_module
        from src.generator.life.persona import draw_persona
        from src.generator.world import communities

        manifest = self.manifest
        registration = datetime.fromisoformat(manifest["registration_end"]).replace(tzinfo=None)
        config.activate_horizon(self.start, self.end, registration)
        settings = emit._build_params(None, None, manifest["community_size"])
        params_module.activate(settings)
        rng_module.configure(manifest["seed"], settings.fingerprint(), manifest["world_seed"])

        for ordinal in range(1, manifest["total_clients"] + 1):
            persona = draw_persona(ordinal)
            client = communities.client_id(ordinal)
            self.mode[client] = persona.activity_mode
            self.joined[client] = persona.relationship_start

    # --------------------------------------------------------

    def months(self) -> list[tuple[int, int]]:
        """
        Границы календарных месяцев окна в днях от начала.
        """
        out, cursor = [], self.start
        while cursor < self.end:
            following = datetime(cursor.year + cursor.month // 12, cursor.month % 12 + 1, 1)
            out.append(((cursor - self.start).days, (min(following, self.end) - self.start).days))
            cursor = following
        return out

    def monthly(self, name: str) -> np.ndarray:
        matrix = self.daily[name]
        return np.stack([matrix[:, left:right].sum(axis=1) for left, right in self.months()], axis=1)

    def alive(self) -> np.ndarray:
        """
        Клиент-месяцы, когда клиент уже пришёл в банк.
        """
        mask = np.zeros((len(self.clients), len(self.months())), dtype=bool)
        for position, client in enumerate(self.clients):
            joined = self.joined.get(client, self.start)
            for column, (left, right) in enumerate(self.months()):
                mask[position, column] = joined < self.start + timedelta(days=right)
        return mask


# ============================================================
# ОТЧЁТ
# ============================================================


def _targets() -> dict:
    from src.generator.params.calibration import DEFAULT_TARGETS
    return {item.metric: item for item in DEFAULT_TARGETS}


def _band(metric: str, value: float) -> str:
    item = _targets().get(metric)
    if item is None:
        return ""
    if item.value is None and (item.low is None or item.high is None):
        return "(эталона нет)"
    if item.value is not None:
        low, high = item.value * (1 - item.tolerance), item.value * (1 + item.tolerance)
    else:
        low, high = item.low, item.high
    mark = "ok" if low <= value <= high else "ВНЕ"
    return f"[{low:.3g}–{high:.3g}] {mark}"


def volume(world: World) -> dict:

    alive = world.alive()
    events = world.monthly("all")[alive]
    actions = world.monthly("action")[alive]

    metrics = {
        "events_per_client_month_mean": float(events.mean()),
        **{f"events_per_client_month_{name}": float(np.percentile(events, q))
           for name, q in (("p10", 10), ("p25", 25), ("median", 50), ("p75", 75), ("p90", 90), ("p95", 95), ("p99", 99))},
        "zero_month_share": float((events == 0).mean()),
        "no_client_action_month_share": float(((actions == 0) & (events > 0)).mean()),
        # Рассылки и антифрод наблюдаются только с запуска своих систем
        # (config.SOURCE_LAUNCH): месяцы до него в среднее не идут.
        "communications_per_client_month": float(world.monthly("comm")[:, 12:][alive[:, 12:]].mean()),
        "inbound_transfers_per_client_month": float(world.monthly("inbound")[alive].mean()),
        "transactions_per_client_month": float(world.monthly("money")[alive].mean()),
    }

    # Сессии на месяц с приложением: месяцы после вехи app_registered.
    visits = world.monthly("visit")
    with_app = np.zeros_like(alive)
    for position, client in enumerate(world.clients):
        marks = [item["event_time"] for item in world.profile[client]["lifelong"] if item["type"] == "app_registered"]
        if marks:
            moment = marks[0].replace(tzinfo=None) + LOCAL
            for column, (left, right) in enumerate(world.months()):
                with_app[position, column] = moment < world.start + timedelta(days=left)
    metrics["app_sessions_per_client_month"] = float(visits[with_app].mean()) if with_app.any() else float("nan")
    metrics["app_visits_per_week"] = metrics["app_sessions_per_client_month"] / (365.25 / 12 / 7)
    metrics["mau_share_of_app_clients"] = float((visits[with_app] > 0).mean()) if with_app.any() else float("nan")
    metrics["active_month_share"] = float((actions > 0).mean())

    purchases = world.daily["purchase"].sum()
    metrics["night_purchase_share"] = float(world.daily["night"].sum() / max(1, purchases))

    decided = world.daily["credit_ok"].sum() + world.daily["credit_no"].sum()
    metrics["approval_rate_credit"] = float(world.daily["credit_ok"].sum() / max(1, decided))
    metrics["installment_missed_share"] = float(world.daily["missed"].sum() / max(1, world.daily["due"].sum()))
    metrics["dpd90_client_share"] = float(len(world.dpd90) / max(1, len(world.clients)))

    years = alive.sum() / 12.0
    after = alive[:, 13:].sum() / 12.0
    metrics["fraud_episodes_per_client_year"] = float(world.monthly("fraud_alert")[:, 13:].sum() / max(1.0, after))
    if world.transitions is not None:
        fraud = (world.transitions["component"] == "fraud").sum()
        metrics["fraud_planned_per_client_year"] = float(fraud / max(1.0, years))

    metrics["declined_share_of_money_attempts"] = float(
        world.daily["decline"].sum() / max(1, world.daily["decline"].sum() + world.daily["money"].sum())
    )
    metrics["support_cases_per_client_year"] = float(world.daily["case"].sum() / max(1.0, years))
    metrics["salary_client_share"] = float((world.daily["salary"].sum(axis=1) > 0).mean())
    metrics["salary_month_share"] = float((world.monthly("salary")[alive] > 0).mean())
    metrics["own_account_out_client_share"] = float((world.daily["own_out"].sum(axis=1) > 0).mean())
    metrics["active_contracts_median"] = float(np.median([row["active_contracts"] or 0 for row in world.profile.values()]))

    # Тишина после действия: среди действовавших за 90 дней до T —
    # без действий 30, 60 и 90 дней после T (среднее по месячным T).
    action = world.daily["action"]
    for length in (30, 60, 90):
        shares = []
        for left, _ in world.months()[3:]:
            if left + length > world.days:
                break
            active = action[:, left - 90:left].sum(axis=1) > 0
            shares.append(float((action[active, left:left + length].sum(axis=1) == 0).mean()))
        metrics[f"inactive_{length}d_share"] = float(np.mean(shares))

    stats, _ = silences(world)
    metrics["return_after_60d_silence_share"] = float(stats.get("returned", 0) / max(1, stats.get("silences", 0)))
    metrics["own_account_out_per_client_month"] = float(world.monthly("own_out")[alive].mean())

    return metrics


def by_mode(world: World) -> dict:

    from src.generator.params.calibration import EVENTS_PER_MONTH_BY_MODE

    alive = world.alive()
    events = world.monthly("all")
    actions = world.monthly("action")
    out = {}

    for mode, band in EVENTS_PER_MONTH_BY_MODE.items():
        rows = [position for position, client in enumerate(world.clients) if world.mode.get(client) == mode]
        if not rows:
            continue
        cells = alive[rows]
        out[mode] = {
            "clients": len(rows),
            "events_per_month": float(events[rows][cells].mean()),
            "band": band,
            "no_action_months": float((actions[rows][cells] == 0).mean()),
        }

    return out


def silences(world: World, length: int = 60) -> dict:
    """
    Тишина: length дней подряд без действий клиента после первого
    действия. Предвестник — в 60 днях до её начала.
    """

    action = world.daily["action"]
    stats = Counter()
    examples = defaultdict(list)

    def span(name: str, left: int, right: int, position: int) -> int:
        return int(world.daily[name][position, max(0, left):max(0, right)].sum())

    for position in range(len(world.clients)):

        days = np.flatnonzero(action[position])
        if len(days) == 0:
            continue

        run_start = None
        for day in range(days[0] + 1, world.days):
            if action[position, day]:
                if run_start is not None and day - run_start >= length:
                    stats["returned"] += 1
                run_start = None
                continue
            if run_start is None:
                run_start = day
            if day - run_start + 1 == length:

                stats["silences"] += 1
                start = run_start

                strong = (
                    span("decline_bank", start - 60, start, position)
                    + span("failed", start - 60, start, position)
                    + span("case", start - 60, start, position)
                )
                salary_stop = span("salary", start - 120, start - 60, position) > 0 and span("salary", start - 60, start, position) == 0
                outflow_up = span("own_out", start - 60, start, position) > span("own_out", start - 120, start - 60, position)
                recent, earlier = span("action", start - 30, start, position) / 30, span("action", start - 90, start - 30, position) / 60
                fading = earlier > 0 and recent < 0.5 * earlier

                flags = {"friction": strong > 0, "salary_stop": salary_stop, "outflow_up": outflow_up, "fading": fading}
                for name, value in flags.items():
                    stats[name] += value
                if not any(flags.values()):
                    stats["no_precursor"] += 1
                    examples["no_precursor"].append(world.clients[position])

    return dict(stats), examples


def truth_report(world: World) -> None:

    if world.transitions is None:
        print("truth/ нет — выгрузка старого генератора")
        return

    states = world.states.copy()
    states["mode"] = states["client_id"].map(world.mode)

    print("\nD. Скрытое состояние: доля клиент-недель по режимам")
    print(f"{'режим':9} {'lapsed':>7} {'away':>6} {'no_app':>7} {'no_cards':>8} {'offline':>7} {'migr':>6} {'salary@':>8} {'F сред':>7}")
    for mode, part in states.groupby("mode"):
        print(
            f"{mode:9} {(part.regime == 'lapsed').mean():7.3f} {part.away.notna().mean():6.3f} "
            f"{(part.away == 'no_app').mean():7.3f} {(part.away == 'no_cards').mean():8.3f} {(part.away == 'offline').mean():7.3f} "
            f"{part.migrating.mean():6.3f} {(~part.salary_here).mean():8.3f} {part.friction.mean():7.3f}"
        )

    print("деньги молчат (lapsed + away offline + no_cards); план пауз 16.3 на seed 541: "
          "silent .253, rare .246, regular .143, high .081, extreme .032")
    for mode, part in states.groupby("mode"):
        money = (part.regime == "lapsed") | part.away.isin(["offline", "no_cards"])
        print(f"  {mode:9} {money.mean():.3f}")

    early = states[states.client_id.map(lambda client: world.joined.get(client, world.start) < world.start)]
    for label, part in (("все", states), ("пришедшие до окна", early)):
        monthly = part.assign(month=part["time"].dt.strftime("%Y-%m")).groupby("month")
        series = monthly.apply(lambda chunk: (chunk.regime == "lapsed").mean(), include_groups=False)
        print(f"доля lapsed по месяцам ({label}):", " ".join(f"{value:.3f}" for value in series.to_numpy()))

    transitions = world.transitions
    print("\nПереходы (component, value, cause):")
    for (component, value, cause), count in Counter(
        zip(transitions.component, transitions.value, transitions.cause)
    ).most_common():
        if component in ("stress", "fraud"):
            continue
        print(f"  {component:9} {value:10} {cause:14} {count}")

    lapses = transitions[
        (transitions.component == "regime") & (transitions.value == "lapsed") & (transitions.cause != "before_window")
    ]
    causes = lapses.cause.value_counts()
    print("причины уходов в окне:", ", ".join(f"{name} {count} ({count / max(1, len(lapses)):.0%})" for name, count in causes.items()))
    if len(lapses):
        by_date = lapses.groupby(lapses["time"].dt.date).size()
        by_day = lapses.groupby((lapses["time"] + LOCAL).dt.day).size()
        print(f"старты lapse по датам: дней {len(by_date)}, медиана {by_date.median():.1f}, максимум {by_date.max()} ({by_date.idxmax()})")
        print(f"старты lapse по дню месяца: максимум/медиана {by_day.max() / max(1, by_day.median()):.2f}")

    paths(world)


def paths(world: World) -> None:
    """
    Пути клиента по truth. Один клиент может пройти несколько путей.
    """

    transitions = world.transitions
    states = world.states
    counts = Counter()

    peak = states.groupby("client_id").friction.max()

    for client in world.clients:

        rows = transitions[transitions.client_id == client].sort_values("time")
        lapse_rows = rows[(rows.component == "regime") & (rows.value == "lapsed")]
        returns = rows[(rows.component == "regime") & (rows.value == "active")]
        migration = rows[(rows.component == "migration") & (rows.value == "start")]
        rollback = rows[(rows.component == "migration") & (rows.value == "end")]
        away = rows[(rows.component == "away") & rows.value.isin(["offline", "no_cards", "no_app"])]

        if lapse_rows.empty and migration.empty and away.empty:
            counts["stable"] += 1
            if world.mode.get(client) in ("silent", "rare"):
                counts["low_frequency_stable"] += 1
        for _, lapse in lapse_rows.iterrows():
            before = migration[(migration.time <= lapse.time) & (migration.time >= lapse.time - timedelta(days=365))]
            if lapse.cause == "never_started":
                counts["never_started"] += 1
            elif len(before) or lapse.cause == "migration":
                counts["gradual (migration → lapse)"] += 1
            elif lapse.cause in ("friction", "stress"):
                counts[f"abrupt ({lapse.cause})"] += 1
            else:
                counts[f"lapse cause {lapse.cause}"] += 1
        if len(returns):
            counts["recovery (return after lapse)"] += 1
        if len(rollback):
            counts["recovery (migration rolled back)"] += 1
        if lapse_rows.empty and peak.get(client, 0.0) >= 1.0:
            counts["bad experience, stayed"] += 1
        if len(away) and lapse_rows.empty:
            counts["exogenous away only"] += 1

    print("\nПути клиентов (клиенты; путей у клиента может быть несколько):")
    for name, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {name:34} {count}")


def quantitative(world: World) -> None:

    from sklearn.metrics import roc_auc_score

    action = world.daily["action"]
    print("\nQ. AUC к будущей неактивности: нет действий 60 дней после T (среди действовавших за 90 до T)")

    states = world.states
    for moment in ("2025-03-01", "2025-09-01", "2026-03-01"):

        cut = (datetime.fromisoformat(moment) - world.start).days
        if cut + 60 > world.days:
            continue

        active = action[:, cut - 90:cut].sum(axis=1) > 0
        label = (action[:, cut:cut + 60].sum(axis=1) == 0)[active]

        def window(name, left, right=0):
            return world.daily[name][:, cut - left:cut - right].sum(axis=1)[active]

        last = np.array([
            (cut - 1 - np.flatnonzero(row[:cut])[-1]) if row[:cut].any() else 9999 for row in action
        ])[active]

        features = {
            "дней с последнего действия": last,
            "действий за 30": -window("action", 30),
            "действий за 90": -window("action", 90),
            "доля 30 к 90": -(window("action", 30) / np.maximum(1, window("action", 90))),
            "отказы за 90": window("decline", 90),
            "сбои операций за 90": window("failed", 90),
            "обращения за 90": window("case", 90),
            "жалобы за 90": window("complaint", 90),
            "зарплата пропала": ((window("salary", 120, 60) > 0) & (window("salary", 60) == 0)).astype(float),
            "переводы себе за 30": window("own_out", 30),
            "визиты за 30": -window("visit", 30),
        }

        if states is not None:
            stamp = datetime.fromisoformat(moment)
            weekly = states[(states.time.dt.tz_convert(None) + LOCAL) <= stamp]
            latest = weekly.sort_values("time").groupby("client_id").tail(1).set_index("client_id")
            clients = [client for position, client in enumerate(world.clients) if active[position]]
            pick = latest.reindex(clients)
            features.update({
                "[скрыто] F": pick.friction.fillna(0).to_numpy(),
                "[скрыто] −x": -pick.affinity.fillna(0).to_numpy(),
                "[скрыто] lapsed": (pick.regime == "lapsed").to_numpy(dtype=float),
                "[скрыто] away": pick.away.notna().to_numpy(dtype=float),
                "[скрыто] migrating": pick.migrating.fillna(False).to_numpy(dtype=float),
            })

        print(f"T {moment}: строк {active.sum()}, неактивных {label.sum()} ({label.mean():.3f})")
        for name, values in features.items():
            if len(set(values)) < 2:
                continue
            auc = roc_auc_score(label, values)
            flag = "  ← близко к 1" if max(auc, 1 - auc) >= 0.97 else ""
            print(f"  {name:28} AUC {auc:.3f}{flag}")

        if states is not None:
            # Внутри режима: недовольство отдельно от частоты операций.
            modes = np.array([world.mode.get(client) for position, client in enumerate(world.clients) if active[position]])
            for mode in ("rare", "regular", "high"):
                chosen = modes == mode
                if label[chosen].sum() >= 3 and (~label[chosen]).sum() >= 3:
                    print(f"  [скрыто] F внутри режима {mode:8} AUC {roc_auc_score(label[chosen], features['[скрыто] F'][chosen]):.3f}")
            f = features["[скрыто] F"]
            for value, name in ((False, "останутся"), (True, "замолчат")):
                part = f[label == value]
                print(f"  F у тех, кто {name}: p25 {np.percentile(part, 25):.3f} p50 {np.percentile(part, 50):.3f} "
                      f"p75 {np.percentile(part, 75):.3f} p90 {np.percentile(part, 90):.3f}")


def prospective(world: World) -> None:
    """
    Уйдёт ли в 60 дней клиент, активный на T (truth), — по всем
    месячным T сразу. Здесь причины видны как причины: недовольство,
    миграция и стресс до ухода, а не след уже случившегося.
    """

    from sklearn.metrics import roc_auc_score

    if world.states is None:
        return

    states = world.states.assign(local=world.states.time.dt.tz_convert(None) + LOCAL)
    transitions = world.transitions
    lapses = transitions[(transitions.component == "regime") & (transitions.value == "lapsed")]
    lapse_time = lapses.assign(local=lapses.time.dt.tz_convert(None) + LOCAL)

    rows = []
    cursor = datetime(world.start.year, world.start.month, 1) + timedelta(days=95)
    cursor = datetime(cursor.year, cursor.month, 1)

    while cursor + timedelta(days=60) <= world.end:
        latest = states[states.local <= cursor].sort_values("local").groupby("client_id").tail(1)
        active = latest[latest.regime == "active"]
        window = lapse_time[(lapse_time.local > cursor) & (lapse_time.local <= cursor + timedelta(days=60))]
        soon = set(window.client_id)
        cut = (cursor - world.start).days
        for row in active.itertuples():
            position = world.index.get(row.client_id)
            if position is None:
                continue
            rows.append((
                row.client_id in soon, row.friction, -row.affinity, float(row.migrating), row.stress,
                world.daily["decline_bank"][position, max(0, cut - 90):cut].sum()
                + world.daily["failed"][position, max(0, cut - 90):cut].sum(),
                world.daily["complaint"][position, max(0, cut - 90):cut].sum(),
                world.daily["own_out"][position, max(0, cut - 30):cut].sum(),
                world.mode.get(row.client_id),
            ))
        cursor = datetime(cursor.year + cursor.month // 12, cursor.month % 12 + 1, 1)

    label = np.array([row[0] for row in rows])
    names = ("[скрыто] F", "[скрыто] −x", "[скрыто] migrating", "[скрыто] стресс",
             "отказы банка и сбои за 90", "жалобы за 90", "переводы себе за 30")

    print(f"\nQ2. Уход в 60 дней среди активных на T, все месячные T: строк {len(rows)}, уходов {label.sum()}")
    for index, name in enumerate(names, start=1):
        values = np.array([row[index] for row in rows], dtype=float)
        if len(set(values)) > 1:
            print(f"  {name:28} AUC {roc_auc_score(label, values):.3f}")
    for mode in ("rare", "regular", "high"):
        chosen = np.array([row[-1] == mode for row in rows])
        if label[chosen].sum() >= 10:
            values = np.array([row[1] for row in rows], dtype=float)[chosen]
            print(f"  [скрыто] F внутри режима {mode:8} AUC {roc_auc_score(label[chosen], values):.3f} (уходов {label[chosen].sum()})")


def autocorrelation(world: World) -> None:

    weeks = world.days // 7
    weekly = world.daily["action"][:, : weeks * 7].reshape(len(world.clients), weeks, 7).sum(axis=2)
    for lag in (1, 4, 13):
        left, right = weekly[:, :-lag].ravel(), weekly[:, lag:].ravel()
        print(f"  автокорреляция недельных действий, лаг {lag:2}: {np.corrcoef(np.log1p(left), np.log1p(right))[0, 1]:.3f}")


def calendar(world: World) -> None:

    action = world.daily["action"].sum(axis=0)
    dates = [world.start + timedelta(days=day) for day in range(world.days)]
    by_dom = defaultdict(list)
    for date, value in zip(dates, action):
        by_dom[date.day].append(value)
    means = {day: float(np.mean(values)) for day, values in by_dom.items()}
    median = float(np.median(list(means.values())))
    print("  действия по дню месяца (к медиане): " + " ".join(f"{day}:{means[day] / median:.2f}" for day in sorted(means)))


def cohorts(world: World) -> None:
    """
    Пример клиента на каждый путь: помесячно — что видит банк, а
    справа отдельно — скрытое состояние (только для аудита).
    """

    transitions, states = world.transitions, world.states
    if transitions is None:
        return

    chosen: dict[str, str] = {}

    def first(name: str, clients) -> None:
        for client in clients:
            if client not in chosen.values():
                chosen[name] = client
                return

    by_client = transitions.groupby("client_id")
    lapsed = transitions[(transitions.component == "regime") & (transitions.value == "lapsed")]
    returned = set(transitions[(transitions.component == "regime") & (transitions.value == "active")].client_id)
    peak = states.groupby("client_id").friction.max()

    moved = set(transitions[transitions.component.isin(["regime", "migration", "away"])].client_id)
    quiet = [
        client for client in world.clients
        if client not in moved and world.joined.get(client, world.start) < world.start
    ]
    first("1 стабильный активный", [client for client in quiet if world.mode.get(client) in ("high", "extreme")])
    first("2 постепенный спад → тишина", lapsed[lapsed.cause == "migration"].client_id)

    # Плохой опыт без ухода: недовольство поднималось до 1 и выше, а
    # ухода в полгода после пика не было.
    calm = []
    for client, part in states[states.friction >= 1.0].groupby("client_id"):
        moment = part.time.min()
        later = lapsed[(lapsed.client_id == client) & (lapsed.time > moment) & (lapsed.time <= moment + timedelta(days=180))]
        if later.empty:
            calm.append(client)
    first("3 плохой опыт → восстановление", calm)
    first("4 финансовый стресс → возврат", [client for client in lapsed[lapsed.cause == "stress"].client_id if client in returned])
    first("5 внешняя отлучка", transitions[(transitions.component == "away") & (transitions.value == "offline")].client_id)
    first("6 миграция в другой банк", transitions[(transitions.component == "salary") & (transitions.value == "elsewhere") & (transitions.cause == "migration")].client_id)
    first("7 мошенничество", transitions[transitions.component == "fraud"].client_id)
    first("8 кредитный стресс", sorted(world.dpd90))
    first("9 рост продуктов", [world.clients[position] for position in np.argsort(-world.daily["product_opened"].sum(axis=1))[:20]])

    columns = ("action", "purchase", "visit", "decline", "failed", "case", "salary", "own_out")

    for name, client in chosen.items():
        position = world.index[client]
        print(f"\n=== {name}: {client} (режим {world.mode.get(client)}) ===")
        print("месяц    " + " ".join(f"{column:>8}" for column in columns) + "   | скрыто: F, x, режим, отлучка")
        weekly = states[states.client_id == client].sort_values("time")
        notes = by_client.get_group(client) if client in by_client.groups else transitions.iloc[:0]
        for column, (left, right) in enumerate(world.months()):
            month = (world.start + timedelta(days=left)).strftime("%Y-%m")
            values = " ".join(f"{int(world.daily[item][position, left:right].sum()):8d}" for item in columns)
            snap = weekly[(weekly.time.dt.tz_convert(None) + LOCAL).dt.strftime("%Y-%m") == month].head(1)
            away = snap.away.iloc[0] if len(snap) else None
            hidden = (
                f"F {snap.friction.iloc[0]:.2f} x {snap.affinity.iloc[0]:+.2f} {snap.regime.iloc[0]:7} "
                f"{away if isinstance(away, str) else '-'}"
                if len(snap) else ""
            )
            events = notes[(notes.time.dt.tz_convert(None) + LOCAL).dt.strftime("%Y-%m") == month]
            marks = "; ".join(f"{row.component}:{row.value}({row.cause})" for row in events.itertuples())
            print(f"{month}  {values}   | {hidden} {marks}")


def report(raw: Path, before: Path | None, show_cohorts: bool, save: Path | None = None) -> None:

    moment = time.time()
    world = World(raw)
    print(f"{raw}: клиентов {len(world.clients)}, дней {world.days}, генератор {world.manifest['generator_version']} "
          f"({time.time() - moment:.0f} с)")

    old = World(before) if before is not None else None

    metrics = volume(world)
    previous = volume(old) if old is not None else {}

    if save is not None:
        modes = by_mode(world)
        save.write_text(json.dumps({"metrics": metrics, "modes": modes}, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\nA–C, F–G. Объём, каналы, кредит, мошенничество (полоса params/calibration.py)")
    for name, value in metrics.items():
        was = f"  до: {previous[name]:.4g}" if name in previous else ""
        print(f"  {name:38} {value:10.4g}  {_band(name, value)}{was}")

    print("\nСобытия на клиент-месяц по режимам (полоса — calibration.EVENTS_PER_MONTH_BY_MODE)")
    modes = by_mode(world)
    earlier = by_mode(old) if old is not None else {}
    for mode, row in modes.items():
        was = earlier.get(mode)
        tail = f"  до: {was['events_per_month']:.1f}, без действий {was['no_action_months']:.3f}" if was else ""
        print(f"  {mode:8} клиентов {row['clients']:4} событий {row['events_per_month']:7.1f} {row['band']}  "
              f"месяцев без действий {row['no_action_months']:.3f}{tail}")

    print("\nОбращения по темам:", dict(world.cases_by_topic.most_common()))

    for label, item in (("сейчас", world), ("до", old)):
        if item is None:
            continue
        print(f"\nC. Время ({label})")
        autocorrelation(item)
        calendar(item)
        per_month = item.monthly("action").sum(axis=0)
        print("  действий по месяцам:", " ".join(str(int(value)) for value in per_month))

    truth_report(world)

    for label, item in (("сейчас", world), ("до", old)):
        if item is None:
            continue
        stats, examples = silences(item)
        print(f"\nE. Тишина ≥ 60 дней без действий клиента ({label}), на 1000 клиентов")
        scale = 1000.0 / len(item.clients)
        for name in ("silences", "returned", "friction", "salary_stop", "outflow_up", "fading", "no_precursor"):
            print(f"  {name:14} {stats.get(name, 0) * scale:8.1f}")

    quantitative(world)
    prospective(world)
    if old is not None:
        print("\n(до)")
        quantitative(old)

    if show_cohorts:
        cohorts(world)


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("generate")
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--seed", type=int, required=True)
    make.add_argument("--clients", type=int, default=1000)
    make.add_argument("--start", default="2024-01-01")
    make.add_argument("--end", default="2026-09-01")
    make.add_argument("--short", default="2026-03-01")
    make.add_argument("--workers", type=int, default=6)

    check = sub.add_parser("prefix")
    check.add_argument("--out", type=Path, required=True)
    check.add_argument("--seed", type=int, required=True)

    look = sub.add_parser("report")
    look.add_argument("--raw", type=Path, required=True)
    look.add_argument("--before", type=Path, default=None)
    look.add_argument("--cohorts", action="store_true")
    look.add_argument("--save", type=Path, default=None, help="метрики в JSON (вне data/)")

    args = parser.parse_args()

    if args.command == "generate":
        generate(args.out, args.seed, args.clients, args.start, args.end, args.short, args.workers)
    elif args.command == "prefix":
        prefix(args.out, args.seed)
    else:
        sys.path.insert(0, str(ROOT / "churn_baseline"))
        report(args.raw, args.before, args.cohorts, _outside_data(args.save) if args.save else None)


if __name__ == "__main__":
    main()
