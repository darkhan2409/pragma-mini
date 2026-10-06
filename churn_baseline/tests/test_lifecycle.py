from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn import lifecycle as lc
from churn.features import compute
from churn.activity import is_client_action


# ============================================================
# ИДЕЯ
# ============================================================
#
# Стадии CAPP на лентах, собранных руками: каждое правило Excel и каждое
# решение владельца — отдельный случай. Отдельно — свойство без утечки
# будущего: обрезка ленты на момент M не меняет ни одного перехода до M.
#
# Регистрация в понедельник 2025-01-06: недели ISO идут ровно от неё.
# ============================================================


LOCAL = timezone(timedelta(hours=5))
COLUMNS = [
    "type", "reason", "channel", "direction", "counterparty", "migration_reason", "change_source",
    "operation", "status", "product_id", "contract_id",
]
REGISTERED = "2025-01-06 10:00"


def at(text: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.fromisoformat(text).replace(tzinfo=LOCAL)).tz_convert("UTC")


def tape(rows: list[tuple], client: str = "c1") -> pd.DataFrame:
    records = [
        {"client_id": client, "t": at(when), "type": kind, **{name: fields.get(name) for name in COLUMNS[1:]}}
        for when, kind, fields in rows
    ]
    frame = pd.DataFrame(records, columns=["client_id", "t", *COLUMNS])
    frame["t"] = pd.to_datetime(frame["t"], utc=True)
    return frame.sort_values("t", kind="stable", ignore_index=True)


def visit(when: str) -> tuple:
    return when, "app_operation", {"operation": "login", "status": "success"}


def purchase(when: str) -> tuple:
    return when, "purchase", {"reason": "purchase", "channel": "pos"}


def opened(when: str, product: str, contract: str, reason: str = "application_approved") -> tuple:
    return when, "product_opened", {"product_id": product, "contract_id": contract, "reason": reason}


def closed(when: str, product: str, contract: str) -> tuple:
    return when, "product_closed", {"product_id": product, "contract_id": contract, "reason": "early_closure"}


def week(monday: str, count: int) -> list[tuple]:
    """
    count визитов за неделю с понедельника monday: по дням недели по кругу.
    """
    start = datetime.fromisoformat(monday)
    return [visit((start + timedelta(days=index % 7, hours=9 + index // 7)).isoformat(" ")) for index in range(count)]


def weeks(monday: str, counts: list[int]) -> list[tuple]:
    start = datetime.fromisoformat(monday)
    out: list[tuple] = []
    for index, count in enumerate(counts):
        out += week((start + timedelta(weeks=index)).isoformat(" "), count)
    return out


def run(rows: list[tuple], until: str = "2025-12-01 00:00") -> pd.DataFrame:
    return lc.history(tape(rows), {"c1": at(REGISTERED)}, at(until))


def stage(history: pd.DataFrame, when: str) -> str:
    return lc.stage_at(history, at(when)).loc["c1", "stage"]


def entered(history: pd.DataFrame, name: str, reason: str | None = None) -> pd.Timestamp:
    rows = history[(history["stage"] == name) & ((history["reason"] == reason) if reason else True)]
    assert len(rows), f"нет перехода в {name} ({reason})"
    return rows["started_at"].iloc[0]


# Клиент с дебетовой и кредитной картой, активный с первого дня.
PRODUCTS = [
    opened("2025-01-06 10:30", "prd_home_card", "k_dc", reason="opened"),
    opened("2025-01-06 10:40", "prd_dos", "k_cc"),
]
ACTIVE = [purchase("2025-01-06 11:00")]


# --- New и Activated ---


def test_a_client_is_new_from_the_app_registration() -> None:

    history = run([visit("2025-01-02 12:00")] + ACTIVE)

    assert history.iloc[0]["stage"] == lc.NEW
    assert history.iloc[0]["started_at"] == at(REGISTERED)
    # До регистрации в приложении стадии нет.
    assert "c1" not in lc.stage_at(history, at("2025-01-05 00:00")).index


def test_a_target_action_activates() -> None:

    history = run([purchase("2025-01-08 15:00")])

    assert entered(history, lc.ACTIVATED) == at("2025-01-09 00:00")
    assert stage(history, "2025-01-08 23:59") == lc.NEW


def test_seven_days_without_a_target_action_put_a_new_client_at_risk() -> None:

    # Карту при регистрации открыл и активировал банк — это не целевое действие.
    rows = [
        opened("2025-01-06 10:30", "prd_home_card", "k_dc", reason="opened"),
        ("2025-01-08 12:00", "card_activated", {"product_id": "prd_home_card", "contract_id": "k_dc"}),
    ]
    history = run(rows)

    # Семь суток от 10:00 06.01 истекают 13.01 в 10:00: стадия — на конец дня.
    assert entered(history, lc.AT_RISK, "new_no_target_action_7d") == at("2025-01-14 00:00")


def test_activated_without_a_further_action_in_fourteen_days_is_at_risk() -> None:

    history = run(ACTIVE)

    assert entered(history, lc.AT_RISK, "activated_no_action_14d") == at("2025-01-21 00:00")

    # Второе действие в эти 14 дней снимает правило.
    history = run(ACTIVE + [purchase("2025-01-15 12:00")])

    assert (history["reason"] != "activated_no_action_14d").all()


# --- Growing ---


def test_a_visit_in_the_month_after_activation_makes_growing() -> None:

    history = run(ACTIVE + [purchase("2025-01-15 12:00"), visit("2025-02-03 09:00")])

    assert entered(history, lc.GROWING) == at("2025-02-04 00:00")


def test_a_visit_two_months_later_does_not_make_growing() -> None:

    history = run(ACTIVE + [purchase("2025-01-15 12:00"), visit("2025-03-03 09:00")])

    assert (history["stage"] != lc.GROWING).all()


def test_a_calendar_month_without_a_visit_puts_growing_at_risk() -> None:

    # Growing в феврале, в марте ни одного визита.
    history = run(ACTIVE + [purchase("2025-01-15 12:00"), visit("2025-02-03 09:00"), purchase("2025-03-10 12:00")])

    assert entered(history, lc.AT_RISK, "growing_not_in_mau") == at("2025-04-01 00:00")


# --- CORE и Loyal ---


# 5 визитов в неделю с 06.01: Growing 04.02. Среднее по 4 неделям
# считается с 4-й недели (27.01–02.02), и четвёртая high-неделя подряд —
# седьмая, 17.02–23.02.
HIGH = weeks("2025-01-06", [5] * 7)


def test_core_needs_four_high_weeks_and_two_products_with_a_debit_card() -> None:

    history = run(PRODUCTS + ACTIVE + HIGH)

    assert entered(history, lc.CORE) == at("2025-02-24 00:00")

    # Без продуктов те же визиты CORE не дают.
    assert (run(ACTIVE + HIGH)["stage"] != lc.CORE).all()

    # Два продукта без дебетовой карты — тоже нет.
    credit_only = [opened("2025-01-06 10:40", "prd_dos", "k_cc"), opened("2025-01-06 10:50", "prd_ozen", "k_cc2")]
    assert (run(credit_only + ACTIVE + HIGH)["stage"] != lc.CORE).all()


@pytest.mark.parametrize(
    ("tail", "at_risk"),
    [
        ([5, 5, 6, 5], False),      # ровная высокая активность
        ([5, 3, 1, 1], True),       # 6 → 5 → 3 → 1: спад H7
    ],
)
def test_core_goes_at_risk_on_a_trend_of_falling_weekly_visits(tail: list[int], at_risk: bool) -> None:

    rows = PRODUCTS + ACTIVE + weeks("2025-01-06", [6] * 8 + tail)
    history = run(rows, until="2025-04-01 00:00")

    assert bool((history["reason"] == "core_wau_decline").any()) is at_risk
    if at_risk:
        # Две недели подряд со средним по 2 неделям ≤ 2: 3-1 (17.03–23.03),
        # затем 1-1 (24.03–30.03).
        assert entered(history, lc.AT_RISK, "core_wau_decline") == at("2025-03-31 00:00")


def test_loyal_is_core_with_a_premium_card() -> None:

    premium = PRODUCTS + [opened("2025-01-06 10:50", "prd_tau", "k_tau")]

    assert entered(run(premium + ACTIVE + HIGH), lc.LOYAL) == at("2025-02-24 00:00")
    assert (run(PRODUCTS + ACTIVE + HIGH)["stage"] != lc.LOYAL).all()


def test_closing_the_premium_card_puts_loyal_at_risk() -> None:

    premium = PRODUCTS + [opened("2025-01-06 10:50", "prd_alem", "k_alem")]
    rows = premium + ACTIVE + weeks("2025-01-06", [5] * 10) + [closed("2025-03-05 12:00", "prd_alem", "k_alem")]
    history = run(rows, until="2025-03-10 00:00")

    assert entered(history, lc.AT_RISK, "loyal_premium_closed") == at("2025-03-06 00:00")


# --- выход из At Risk и Churn ---


def test_a_visit_brings_growing_back_from_risk() -> None:

    rows = ACTIVE + [purchase("2025-01-15 12:00"), visit("2025-02-03 09:00"), visit("2025-04-02 09:00")]
    history = run(rows, until="2025-04-10 00:00")

    assert entered(history, lc.AT_RISK, "growing_not_in_mau") == at("2025-04-01 00:00")
    back = history[history["reason"] == "recovered"].iloc[0]
    assert back["started_at"] == at("2025-04-03 00:00")
    assert back["stage"] == lc.GROWING
    assert back["previous_stage"] == lc.AT_RISK


def test_after_a_core_decline_the_stage_is_counted_anew() -> None:

    # Спад, затем снова 5 визитов в неделю: выход в Growing, и CORE —
    # только после новой high-серии, начавшейся после спада.
    rows = PRODUCTS + ACTIVE + weeks("2025-01-06", [6] * 8 + [5, 3, 1, 1] + [5] * 8)
    history = run(rows, until="2025-05-20 00:00")

    # Спад 31.03; неделя 31.03–06.04 со средним (1 + 5) / 2 > 2 — выход.
    # Новая high-серия: среднее по 4 неделям ≥ 4 с недели 14.04 (1, 5, 5, 5),
    # четвёртая подряд — 05.05–11.05.
    back = history[history["reason"] == "recovered"].iloc[0]
    assert back["stage"] == lc.GROWING
    assert back["started_at"] == at("2025-04-07 00:00")
    again = history[(history["stage"] == lc.CORE) & (history["started_at"] > back["started_at"])]
    assert again["started_at"].iloc[0] == at("2025-05-12 00:00")


def test_sixty_days_without_an_action_is_churn_and_an_action_reactivates() -> None:

    history = run(ACTIVE + [purchase("2025-05-01 12:00")], until="2025-06-01 00:00")

    # Последнее действие 06.01 в 11:00: 60 суток истекают 07.03 в 11:00.
    assert entered(history, lc.CHURN, "no_action_60d") == at("2025-03-08 00:00")
    back = history[history["reason"] == "reactivated"].iloc[0]
    assert back["started_at"] == at("2025-05-02 00:00")
    assert back["stage"] == lc.ACTIVATED


def test_obligations_do_not_keep_a_client_out_of_churn() -> None:

    rows = ACTIVE + [
        ("2025-02-15 10:00", "loan_payment", {"channel": "app"}),
        ("2025-02-15 09:50", "cash_deposit", {"reason": "payment_topup", "channel": "atm"}),
    ]

    assert entered(run(rows), lc.CHURN) == at("2025-03-08 00:00")


# --- без будущего ---


def random_tape(seed: int, client: str) -> pd.DataFrame:
    """
    Год жизни клиента: визиты с меняющейся частотой и тишиной, покупки,
    продукты с открытием и закрытием, взносы по кредиту.
    """
    rng = np.random.default_rng(seed)
    rows: list[tuple] = [
        opened("2025-01-06 10:30", "prd_home_card", f"{client}_dc", reason="opened"),
    ]
    day = datetime(2025, 1, 6)
    rate = 0.8
    for offset in range(330):
        current = day + timedelta(days=offset)
        if rng.random() < 0.03:
            rate = float(rng.choice([0.0, 0.2, 0.8, 1.5]))
        for _ in range(rng.poisson(rate)):
            rows.append(visit((current + timedelta(hours=int(rng.integers(7, 23)))).isoformat(" ")))
        if rng.random() < rate * 0.3:
            rows.append(purchase((current + timedelta(hours=13)).isoformat(" ")))
        if rng.random() < 0.01:
            product = str(rng.choice(["prd_dos", "prd_tau", "prd_deposit_prostoy"]))
            rows.append(opened((current + timedelta(hours=11)).isoformat(" "), product, f"{client}_{offset}"))
        if rng.random() < 0.004:
            rows.append(closed((current + timedelta(hours=12)).isoformat(" "), "prd_tau", f"{client}_tau"))
        if current.day == 10:
            rows.append(((current + timedelta(hours=10)).isoformat(" "), "loan_payment", {"channel": "app"}))
    return tape(rows, client)


CLIENTS = [f"c{index}" for index in range(6)]
END = "2025-12-01 00:00"


@pytest.fixture(scope="module")
def world() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    events = pd.concat([random_tape(index, client) for index, client in enumerate(CLIENTS)], ignore_index=True)
    registered = {client: at(REGISTERED) for client in CLIENTS}
    return events, registered, lc.history(events, registered, at(END))


@pytest.mark.parametrize("cut", ["2025-02-17 00:00", "2025-04-01 00:00", "2025-06-15 13:37", "2025-09-01 00:00"])
def test_the_history_before_m_does_not_depend_on_events_after_m(world, cut: str) -> None:

    events, registered, full = world
    moment = at(cut)

    short = lc.history(events[events["t"] < moment], registered, moment)

    known = full[full["started_at"] <= moment].reset_index(drop=True)
    pd.testing.assert_frame_equal(short.reset_index(drop=True), known)
    pd.testing.assert_frame_equal(lc.stage_at(short, moment), lc.stage_at(full, moment))


def test_a_client_whose_events_all_come_later_still_has_a_past() -> None:
    """
    Все события клиента позже M: на ленте до M событий нет, но стадии от
    регистрации у него те же, что по всей ленте.
    """
    rows = [purchase("2025-05-01 12:00")]
    registered = {"c1": at(REGISTERED)}
    moment = at("2025-04-01 00:00")

    full = lc.history(tape(rows), registered, at(END))
    short = lc.history(tape(rows).iloc[0:0], registered, moment)

    pd.testing.assert_frame_equal(short, full[full["started_at"] <= moment].reset_index(drop=True))
    assert list(short["stage"]) == [lc.NEW, lc.AT_RISK, lc.CHURN]


def test_an_event_at_the_moment_itself_does_not_change_the_stage(world) -> None:

    events, registered, full = world
    moment = at("2025-05-01 00:00")
    extra = tape([purchase("2025-05-01 00:00")], "c0")

    with_event = lc.history(pd.concat([events, extra], ignore_index=True), registered, at(END))

    pd.testing.assert_frame_equal(lc.stage_at(with_event, moment), lc.stage_at(full, moment))


def test_the_same_tape_gives_the_same_history(world) -> None:

    events, registered, full = world

    pd.testing.assert_frame_equal(lc.history(events, registered, at(END)), full)


def test_every_stage_appears_in_the_random_world(world) -> None:
    # Свойства выше проверены не на пустом месте.
    assert set(world[2]["stage"]) >= {lc.NEW, lc.ACTIVATED, lc.GROWING, lc.AT_RISK, lc.CHURN}


# --- стадия не признак ---


def test_the_lifecycle_is_not_a_feature() -> None:

    events = random_tape(0, "c0").assign(
        **{name: None for name in ("is_subscription", "is_online", "amount", "balance_after", "account_id",
                                   "card_id", "mcc", "merchant_name", "decline_reason", "days_past_due",
                                   "decision", "delivered", "template", "session_id", "field_name",
                                   "old_value", "new_value")},
        source="app_operations",
        raw_row=0,
    )
    features, described = compute(events, at("2025-09-01 00:00").to_pydatetime(), is_client_action(events))

    names = " ".join(features.columns).lower()
    assert "stage" not in names and "lifecycle" not in names
    assert not any(name in features.columns for name in lc.STAGES)


def test_feature_builders_never_read_lifecycle_or_truth() -> None:
    """
    Стадия — разметка для анализа после прогноза. Модули признаков, метки
    и обучения её не читают, как и служебную правду генератора truth/.
    """
    churn = Path(__file__).resolve().parents[1] / "churn"
    for name in ("features.py", "profile.py", "build.py", "train.py", "plus_usr.py", "target.py", "raw.py", "sources.py"):
        assert "lifecycle" not in (churn / name).read_text(encoding="utf-8"), name

    pattern = re.compile(r"""["'/]truth\b""")
    assert [path.name for path in churn.glob("*.py") if pattern.search(path.read_text(encoding="utf-8"))] == []
