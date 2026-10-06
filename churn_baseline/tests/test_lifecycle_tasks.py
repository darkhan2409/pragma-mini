from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn import lifecycle as lc
from churn import lifecycle_tasks as tasks
from churn.activity import is_client_action
from churn.config import cutoff
from churn.features import compute
from churn.raw import client_blocks
from world import LOCAL, UTC, event, login, profile, purchase, write_future, write_group


# ============================================================
# ИДЕЯ
# ============================================================
#
# Метки active_to_at_risk (режимы A1–A3) и at_risk_outcome (B1–B3) на
# историях стадий, собранных руками: окно, календарный горизонт,
# население, censoring, первый исход, тип перехода. Сквозные случаи — на
# маленьких выгрузках: стадии по ленте, признаки на T, продолжение train
# и матрица модели без колонок lifecycle.
# ============================================================


T = datetime(2025, 6, 1, tzinfo=LOCAL)


def moment(days: float) -> pd.Timestamp:
    return pd.Timestamp(T + timedelta(days=days)).tz_convert("UTC")


def history(*rows: tuple) -> pd.DataFrame:
    """
    Переходы: (клиент, дни от T, стадия, прежняя стадия, причина).
    """
    frame = pd.DataFrame(
        [(client, moment(days), stage, previous, reason) for client, days, stage, previous, reason in rows],
        columns=list(lc.COLUMNS),
    )
    return frame.sort_values(["client_id", "started_at"], kind="stable", ignore_index=True)


def by_client(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.set_index("client_id")


def task_a(rows: list[tuple], horizon, mode: str, followup: float = 61) -> pd.DataFrame:
    return by_client(tasks.future_at_risk(history(*rows), T, moment(followup), horizon, mode))


def task_b(rows: list[tuple], mode: str, horizon: int | None = None, followup: float = 61) -> pd.DataFrame:
    return by_client(tasks.at_risk_outcome(history(*rows), T, moment(followup), mode, horizon))


# CORE до T и At Risk после: a — через 10 дней, b — через 20.
CORE_THEN_RISK = [
    ("a", -100, lc.NEW, None, "registered"),
    ("a", -60, lc.CORE, lc.GROWING, "promoted"),
    ("a", 10, lc.AT_RISK, lc.CORE, "core_wau_decline"),
    ("b", -100, lc.NEW, None, "registered"),
    ("b", -60, lc.CORE, lc.GROWING, "promoted"),
    ("b", 20, lc.AT_RISK, lc.CORE, "core_wau_decline"),
]

# Activated давно и уходит в Churn через 9 дней, At Risk не было.
DIRECT_CHURN = [
    ("x", -50, lc.NEW, None, "registered"),
    ("x", -49, lc.ACTIVATED, lc.NEW, "promoted"),
    ("x", 9, lc.CHURN, lc.ACTIVATED, "no_action_60d"),
]


# --- active_to_at_risk: окно ---


@pytest.mark.parametrize(
    ("client", "horizon", "target"),
    [("a", 14, 1.0), ("b", 14, 0.0), ("b", 30, 1.0), ("a", 7, 0.0)],
)
def test_the_first_at_risk_inside_the_window_is_positive(client: str, horizon: int, target: float) -> None:

    rows = task_a(CORE_THEN_RISK, horizon, tasks.A1)

    assert rows.loc[client, "target"] == target
    assert not rows.loc[client, "is_censored"]
    assert rows.loc[client, "current_stage"] == lc.CORE
    assert rows.loc[client, "at_risk_previous_stage"] == lc.CORE


def test_a_transition_exactly_at_the_window_end_is_inside() -> None:

    assert task_a(CORE_THEN_RISK, 20, tasks.A1).loc["b", "target"] == 1.0


def test_clients_already_at_risk_or_churned_at_t_are_not_rows() -> None:

    rows = task_a(
        CORE_THEN_RISK + [
            ("risk", -100, lc.NEW, None, "registered"),
            ("risk", -5, lc.AT_RISK, lc.GROWING, "growing_not_in_mau"),
            ("gone", -100, lc.NEW, None, "registered"),
            ("gone", -3, lc.CHURN, lc.AT_RISK, "no_action_60d"),
            # переход ровно в T уже действует на T
            ("edge", -100, lc.NEW, None, "registered"),
            ("edge", 0, lc.AT_RISK, lc.GROWING, "growing_not_in_mau"),
            # регистрация после T: на T стадии нет
            ("late", 3, lc.NEW, None, "registered"),
        ],
        14, tasks.A1,
    )

    assert set(rows.index) == {"a", "b"}


def test_a_window_not_covered_by_data_is_censored_not_negative() -> None:

    rows = task_a(CORE_THEN_RISK, 14, tasks.A1, followup=12)

    assert rows["is_censored"].all()
    assert rows["target"].isna().all()
    # At Risk клиента a виден в обрезанном окне, но строка всё равно
    # цензурирована: неизвестные отрицательные выпали бы, а он остался.
    assert rows.loc["a", "first_at_risk_at"] == moment(10)


def test_rows_keep_the_stage_at_t_for_stratification() -> None:

    rows = task_a(
        CORE_THEN_RISK + [
            ("g", -100, lc.NEW, None, "registered"),
            ("g", -30, lc.GROWING, lc.ACTIVATED, "promoted"),
            ("g", 31, lc.AT_RISK, lc.GROWING, "growing_not_in_mau"),
            ("v", -10, lc.NEW, None, "registered"),
            ("v", -9, lc.ACTIVATED, lc.NEW, "promoted"),
            ("n", -2, lc.NEW, None, "registered"),
            ("n", 6, lc.AT_RISK, lc.NEW, "new_no_target_action_7d"),
        ],
        30, tasks.A1,
    )

    assert rows["current_stage"].to_dict() == {
        "a": lc.CORE, "b": lc.CORE, "g": lc.GROWING, "v": lc.ACTIVATED, "n": lc.NEW,
    }
    assert rows["target"].to_dict() == {"a": 1.0, "b": 1.0, "g": 0.0, "v": 0.0, "n": 1.0}
    assert rows.loc["n", "at_risk_reason"] == "new_no_target_action_7d"
    assert rows.loc["g", "days_to_at_risk"] == 31


# --- active_to_at_risk: режимы ---


def test_a1_direct_churn_is_not_positive() -> None:

    rows = task_a(DIRECT_CHURN, 14, tasks.A1)

    assert rows.loc["x", "target"] == 0.0
    assert rows.loc["x", "deterioration"] == lc.CHURN
    assert rows.loc["x", "first_churn_at"] == moment(9)


def test_a2_direct_churn_is_positive() -> None:

    rows = task_a(DIRECT_CHURN + CORE_THEN_RISK, 14, tasks.A2)

    assert rows["target"].to_dict() == {"x": 1.0, "a": 1.0, "b": 0.0}


def test_a3_counts_churn_only_where_the_stage_rule_says_so() -> None:

    rows = task_a(
        DIRECT_CHURN + [
            # CORE уходит в Churn без At Risk: для CORE это не его исход
            ("c", -100, lc.NEW, None, "registered"),
            ("c", -60, lc.CORE, lc.GROWING, "promoted"),
            ("c", 9, lc.CHURN, lc.CORE, "no_action_60d"),
        ],
        14, tasks.A3,
    )

    assert rows["target"].to_dict() == {"x": 1.0, "c": 0.0}


def test_the_transition_type_is_the_first_deterioration_in_the_window() -> None:

    rows = task_a(
        DIRECT_CHURN + CORE_THEN_RISK + [
            # At Risk, затем Churn: тип — At Risk
            ("g", -100, lc.NEW, None, "registered"),
            ("g", -30, lc.GROWING, lc.ACTIVATED, "promoted"),
            ("g", 3, lc.AT_RISK, lc.GROWING, "growing_not_in_mau"),
            ("g", 10, lc.CHURN, lc.AT_RISK, "no_action_60d"),
            # Growing стал CORE и ушёл в At Risk: тип по стадии на T
            ("u", -100, lc.NEW, None, "registered"),
            ("u", -30, lc.GROWING, lc.ACTIVATED, "promoted"),
            ("u", 2, lc.CORE, lc.GROWING, "promoted"),
            ("u", 12, lc.AT_RISK, lc.CORE, "core_wau_decline"),
        ],
        14, tasks.A1,
    )

    assert rows["transition_type"].to_dict() == {
        "x": "activated→churn", "a": "core→at_risk", "b": tasks.NONE,
        "g": "growing→at_risk", "u": "growing→at_risk",
    }
    assert rows.loc["u", "at_risk_previous_stage"] == lc.CORE


def test_an_unknown_mode_is_refused() -> None:

    with pytest.raises(ValueError, match="режим"):
        tasks.future_at_risk(history(*CORE_THEN_RISK), T, moment(61), 14, "best")


# --- календарный горизонт ---


GROWING_AT_MONTH_END = [
    ("g", -100, lc.NEW, None, "registered"),
    ("g", -40, lc.GROWING, lc.ACTIVATED, "promoted"),
    # T = 01.06: июнь без визита — At Risk в полночь 01.07, T + 30
    ("g", 30, lc.AT_RISK, lc.GROWING, "growing_not_in_mau"),
]


def test_growing_at_risk_at_the_month_end_needs_the_whole_month() -> None:
    """
    В июне 30 дней: правило срабатывает в T + 30. В 31-дневном месяце —
    в T + 31, и окно 30 дней его не видит.
    """
    july = [("g", -100, lc.NEW, None, "registered"), ("g", -40, lc.GROWING, lc.ACTIVATED, "promoted")]
    t_july = datetime(2025, 7, 1, tzinfo=LOCAL)
    at_month_end = pd.Timestamp(datetime(2025, 8, 1, tzinfo=LOCAL)).tz_convert("UTC")
    frame = pd.concat([
        history(*july),
        pd.DataFrame([("g", at_month_end, lc.AT_RISK, lc.GROWING, "growing_not_in_mau")], columns=list(lc.COLUMNS)),
    ], ignore_index=True)
    followup = pd.Timestamp(datetime(2025, 9, 1, tzinfo=LOCAL)).tz_convert("UTC")

    targets = {
        horizon: tasks.future_at_risk(frame, t_july, followup, horizon, tasks.A1)["target"].iloc[0]
        for horizon in (30, 31, 35, tasks.NEXT_MONTH)
    }

    assert targets == {30: 0.0, 31: 1.0, 35: 1.0, tasks.NEXT_MONTH: 1.0}
    assert task_a(GROWING_AT_MONTH_END, 30, tasks.A1).loc["g", "target"] == 1.0


@pytest.mark.parametrize(
    ("start", "days"),
    [
        (datetime(2025, 2, 1), 28),
        (datetime(2024, 2, 1), 29),
        (datetime(2025, 4, 1), 30),
        (datetime(2025, 1, 1), 31),
        (datetime(2025, 12, 1), 31),
        # не 1-е число: первый полный месяц после T — следующий
        (datetime(2025, 3, 15), 47),
    ],
)
def test_the_calendar_horizon_ends_with_the_first_full_month(start: datetime, days: int) -> None:

    moment_local = start.replace(tzinfo=LOCAL)
    end = tasks.window_end(moment_local, tasks.NEXT_MONTH)

    assert (end - pd.Timestamp(moment_local).tz_convert("UTC")) / pd.Timedelta(days=1) == days
    assert (end + pd.Timedelta(hours=5)).day == 1
    assert (end + pd.Timedelta(hours=5)).hour == 0


# --- at_risk_outcome ---


AT_RISK_AT_T = [
    ("core", -50, lc.CORE, lc.GROWING, "promoted"),
    ("core", -5, lc.AT_RISK, lc.CORE, "core_wau_decline"),
    ("core", 12, lc.CORE, lc.AT_RISK, "recovered"),
    ("grow", -40, lc.GROWING, lc.ACTIVATED, "promoted"),
    ("grow", -2, lc.AT_RISK, lc.GROWING, "growing_not_in_mau"),
    ("grow", 5, lc.GROWING, lc.AT_RISK, "recovered"),
    ("gone", -40, lc.GROWING, lc.ACTIVATED, "promoted"),
    ("gone", -2, lc.AT_RISK, lc.GROWING, "growing_not_in_mau"),
    ("gone", 30, lc.CHURN, lc.AT_RISK, "no_action_60d"),
    ("stay", -50, lc.CORE, lc.GROWING, "promoted"),
    ("stay", -5, lc.AT_RISK, lc.CORE, "core_wau_decline"),
    ("flip", -40, lc.GROWING, lc.ACTIVATED, "promoted"),
    ("flip", -2, lc.AT_RISK, lc.GROWING, "growing_not_in_mau"),
    ("flip", 5, lc.GROWING, lc.AT_RISK, "recovered"),
    ("flip", 20, lc.AT_RISK, lc.GROWING, "growing_not_in_mau"),
    ("flip", 40, lc.CHURN, lc.AT_RISK, "no_action_60d"),
    ("well", -40, lc.GROWING, lc.ACTIVATED, "promoted"),
]


def test_the_first_exit_from_at_risk_is_the_outcome() -> None:

    rows = task_b(AT_RISK_AT_T, tasks.B3)

    assert set(rows.index) == {"core", "grow", "gone", "stay", "flip"}
    assert rows["outcome"].to_dict() == {
        "core": tasks.RECOVERY, "grow": tasks.RECOVERY, "gone": tasks.CHURN,
        "stay": tasks.CENSORED, "flip": tasks.RECOVERY,
    }
    assert rows.loc["core", "outcome_stage"] == lc.CORE
    assert rows.loc["grow", "outcome_stage"] == lc.GROWING
    assert rows.loc["core", "previous_active_stage"] == lc.CORE
    assert rows.loc["grow", "at_risk_reason"] == "growing_not_in_mau"
    assert rows.loc["grow", "at_risk_since"] == moment(-2)


def test_b1_keeps_only_completed_cases() -> None:

    rows = task_b(AT_RISK_AT_T, tasks.B1, 14)

    # В 14 дней вышли core (12), grow (5) и flip (5); gone и stay — ещё
    # At Risk и в задачу не входят.
    assert rows["target"].dropna().to_dict() == {"core": 0.0, "grow": 0.0, "flip": 0.0}
    assert rows.loc[["gone", "stay"], "target"].isna().all()
    assert rows.loc[["gone", "stay"], "outcome"].eq(tasks.STILL_AT_RISK).all()

    whole = task_b(AT_RISK_AT_T, tasks.B1)
    assert whole["target"].dropna().to_dict() == {"core": 0.0, "grow": 0.0, "gone": 1.0, "flip": 0.0}
    assert whole.loc["stay", "is_censored"]


def test_b2_keeps_still_at_risk_as_a_class() -> None:

    rows = task_b(AT_RISK_AT_T, tasks.B2, 14)

    assert rows["target"].to_dict() == {"core": 0.0, "grow": 0.0, "gone": 2.0, "stay": 2.0, "flip": 0.0}
    assert not rows["is_censored"].any()

    # Окно длиннее наблюдения: без исхода — censored, не still_at_risk.
    longer = task_b(AT_RISK_AT_T, tasks.B2, 90)
    assert longer.loc["stay", "outcome"] == tasks.CENSORED
    assert np.isnan(longer.loc["stay", "target"])
    assert longer.loc["gone", "target"] == 1.0

    with pytest.raises(ValueError, match="горизонт"):
        tasks.at_risk_outcome(history(*AT_RISK_AT_T), T, moment(61), tasks.B2)


def test_b3_gives_times_and_censoring_for_survival_analysis() -> None:

    rows = task_b(AT_RISK_AT_T, tasks.B3)

    assert rows["time_to_event"].to_dict() == {"core": 12, "grow": 5, "gone": 30, "stay": 61, "flip": 5}
    assert rows["is_censored"].to_dict() == {"core": False, "grow": False, "gone": False, "stay": True, "flip": False}
    assert rows.loc["core", "time_to_recovery"] == 12 and np.isnan(rows.loc["core", "time_to_churn"])
    assert rows.loc["gone", "time_to_churn"] == 30 and np.isnan(rows.loc["gone", "time_to_recovery"])
    assert np.isnan(rows.loc["stay", "time_to_recovery"]) and np.isnan(rows.loc["stay", "time_to_churn"])
    # Не классификация: метки нет.
    assert rows["target"].isna().all()

    with pytest.raises(ValueError, match="без горизонта"):
        tasks.at_risk_outcome(history(*AT_RISK_AT_T), T, moment(61), tasks.B3, 60)


# --- матрица модели ---


def test_the_model_matrix_has_only_features() -> None:

    features = pd.DataFrame(
        {"act_count_30": [3.0, 1.0, 0.0], "fees_90": [0.0, 1.0, 2.0]},
        index=pd.Index(["a", "b", "c"], name="client_id"),
    )
    rows = tasks.future_at_risk(history(*CORE_THEN_RISK), T, moment(61), 14, tasks.A1)

    X, y, meta = tasks.task_matrix(rows, features)

    assert list(X.columns) == ["act_count_30", "fees_90"]
    assert list(X.index) == ["a", "b"]
    assert y.to_dict() == {"a": 1, "b": 0}
    assert {"current_stage", "transition_type", "at_risk_reason"} <= set(meta.columns)
    assert "target" not in meta.columns

    for column in ("current_stage", "previous_active_stage", "transition_type", "deterioration"):
        with pytest.raises(ValueError, match="lifecycle"):
            tasks.task_matrix(rows, features.assign(**{column: "core"}))

    censored = tasks.future_at_risk(history(*CORE_THEN_RISK), T, moment(12), 14, tasks.A1)
    assert tasks.task_matrix(censored, features)[0].empty


# --- сквозь выгрузку ---


END = datetime(2025, 9, 1, tzinfo=LOCAL)
REGISTERED = datetime(2025, 1, 6, 10, tzinfo=LOCAL)


def visits(client: str, start: datetime, weeks: int, per_week: int) -> list[dict]:
    return [
        login(client, start + timedelta(weeks=week, days=index % 7, hours=9 + index // 7), session=f"{client}_{week}_{index}")
        for week in range(weeks)
        for index in range(per_week)
    ]


def products(client: str) -> list[dict]:
    return [
        event(client, REGISTERED + timedelta(minutes=30), "product_events", type="product_opened",
              product_id="prd_home_card", contract_id=f"{client}_dc", reason="opened"),
        event(client, REGISTERED + timedelta(minutes=40), "product_events", type="product_opened",
              product_id="prd_dos", contract_id=f"{client}_cc", reason="application_approved"),
        purchase(client, REGISTERED + timedelta(hours=1)),
    ]


def core_client(client: str, high_until: datetime) -> list[dict]:
    """
    6 визитов в неделю с 06.01 до high_until, дальше 1 в неделю: спад H7
    срабатывает в начале четвёртой недели после high_until.
    """
    high = (high_until - datetime(2025, 1, 6, tzinfo=LOCAL)).days // 7
    return (
        products(client)
        + visits(client, datetime(2025, 1, 6, tzinfo=LOCAL), high, 6)
        + visits(client, high_until, 10, 1)
    )


def app_profile(client: str, as_of: datetime) -> dict:
    return profile(
        client, as_of,
        lifelong=[{"type": "app_registered", "event_time": REGISTERED.astimezone(UTC), "source_id": None}],
    )


def export(tmp_path: Path, events: list[dict], clients: list[str], name: str = "w") -> Path:
    raw = tmp_path / name / "raw"
    write_group(raw, "val", END, events, [app_profile(client, END) for client in clients])
    return raw


def labels(state: tasks.GroupHistory, horizon, mode: str = tasks.A1) -> pd.DataFrame:
    return by_client(tasks.future_at_risk(state.history, state.cutoff, state.followup_end, horizon, mode))


def test_end_to_end_core_clients_get_the_labels_of_their_decline(tmp_path) -> None:

    # T val = 1-е число месяца «конец − 60 дней» = 01.07.2025. Спад:
    # c13 — At Risk 14.07 (T + 13), c20 — 21.07 (T + 20).
    events = core_client("c13", datetime(2025, 6, 23, tzinfo=LOCAL)) + core_client("c20", datetime(2025, 6, 30, tzinfo=LOCAL))
    state = tasks.group_history("val", raw_dir=export(tmp_path, events, ["c13", "c20"]))

    assert state.cutoff == pd.Timestamp(datetime(2025, 7, 1, tzinfo=LOCAL)).tz_convert("UTC")
    assert state.followup_end == pd.Timestamp(END).tz_convert("UTC")

    assert labels(state, 14)["current_stage"].to_dict() == {"c13": lc.CORE, "c20": lc.CORE}
    assert labels(state, 14)["days_to_at_risk"].to_dict() == {"c13": 13, "c20": 20}
    assert labels(state, 14)["transition_type"].to_dict() == {"c13": "core→at_risk", "c20": tasks.NONE}
    assert labels(state, 7)["target"].to_dict() == {"c13": 0.0, "c20": 0.0}
    assert labels(state, 14)["target"].to_dict() == {"c13": 1.0, "c20": 0.0}
    assert labels(state, 30)["target"].to_dict() == {"c13": 1.0, "c20": 1.0}


def features_at(raw: Path) -> pd.DataFrame:
    moment_t = cutoff("val", raw)
    parts = [compute(block, moment_t, is_client_action(block))[0] for block in client_blocks(raw / "val" / "events.parquet")]
    return pd.concat(parts)


def test_an_event_after_t_changes_the_label_but_not_the_features(tmp_path) -> None:

    base = core_client("c13", datetime(2025, 6, 23, tzinfo=LOCAL))
    # Пять визитов на неделе 07.07 снимают спад: At Risk не наступает.
    rescue = visits("c13", datetime(2025, 7, 7, tzinfo=LOCAL), 1, 5)

    plain = export(tmp_path, base, ["c13"], "plain")
    saved = export(tmp_path, base + rescue, ["c13"], "saved")

    first = tasks.group_history("val", raw_dir=plain)
    second = tasks.group_history("val", raw_dir=saved)

    assert labels(first, 14).loc["c13", "target"] == 1.0
    assert labels(second, 14).loc["c13", "target"] == 0.0

    # Префикс до T один — признаки и стадия на T те же.
    pd.testing.assert_frame_equal(features_at(plain), features_at(saved))
    pd.testing.assert_frame_equal(lc.stage_at(first.history, first.cutoff), lc.stage_at(second.history, second.cutoff))


def test_events_after_the_window_do_not_change_the_label(tmp_path) -> None:

    base = core_client("c20", datetime(2025, 6, 30, tzinfo=LOCAL))
    # После T + 14 клиент возвращается к высокой активности.
    later = visits("c20", datetime(2025, 7, 21, tzinfo=LOCAL), 6, 6)

    first = tasks.group_history("val", raw_dir=export(tmp_path, base, ["c20"], "base"))
    second = tasks.group_history("val", raw_dir=export(tmp_path, base + later, ["c20"], "later"))

    for mode in tasks.MODES_A:
        assert labels(first, 14, mode)["target"].tolist() == labels(second, 14, mode)["target"].tolist() == [0.0]


def test_the_history_past_the_window_end_does_not_change_task_a(tmp_path) -> None:

    events = core_client("c13", datetime(2025, 6, 23, tzinfo=LOCAL)) + core_client("c20", datetime(2025, 6, 30, tzinfo=LOCAL))
    state = tasks.group_history("val", raw_dir=export(tmp_path, events, ["c13", "c20"]))

    for horizon in (7, 14, 30, tasks.NEXT_MONTH):
        end = tasks.window_end(state.cutoff, horizon)
        cut = state.history[state.history["started_at"] <= end]
        for mode in tasks.MODES_A:
            full = tasks.future_at_risk(state.history, state.cutoff, state.followup_end, horizon, mode)
            short = tasks.future_at_risk(cut, state.cutoff, state.followup_end, horizon, mode)
            pd.testing.assert_series_equal(full["target"], short["target"])
            pd.testing.assert_series_equal(full["transition_type"], short["transition_type"])


def test_the_horizon_does_not_change_the_features(tmp_path) -> None:

    events = core_client("c13", datetime(2025, 6, 23, tzinfo=LOCAL)) + core_client("c20", datetime(2025, 6, 30, tzinfo=LOCAL))
    raw = export(tmp_path, events, ["c13", "c20"])
    state = tasks.group_history("val", raw_dir=raw)
    features = features_at(raw)

    matrices = [
        tasks.task_matrix(tasks.future_at_risk(state.history, state.cutoff, state.followup_end, horizon, mode), features)[0]
        for horizon in (7, 14, 30, tasks.NEXT_MONTH)
        for mode in tasks.MODES_A
    ]

    for X in matrices[1:]:
        pd.testing.assert_frame_equal(X, matrices[0])


def test_the_matrix_from_a_real_block_carries_no_lifecycle_column(tmp_path) -> None:

    events = core_client("c13", datetime(2025, 6, 23, tzinfo=LOCAL)) + core_client("c20", datetime(2025, 6, 30, tzinfo=LOCAL))
    raw = export(tmp_path, events, ["c13", "c20"])
    state = tasks.group_history("val", raw_dir=raw)
    features = features_at(raw)

    for rows in (
        *(tasks.future_at_risk(state.history, state.cutoff, state.followup_end, 30, mode) for mode in tasks.MODES_A),
        tasks.at_risk_outcome(state.history, state.cutoff, state.followup_end, tasks.B1, 60),
        tasks.at_risk_outcome(state.history, state.cutoff, state.followup_end, tasks.B2, 60),
    ):
        X, _, _ = tasks.task_matrix(rows, features)
        assert list(X.columns) == list(features.columns)
        assert not (tasks.META | set(lc.STAGES)) & set(X.columns)


# --- train: метка из продолжения ---


def test_train_takes_the_window_from_its_continuation(tmp_path) -> None:

    # Выгрузка train кончается в T; события после T — только в продолжении.
    end = datetime(2025, 7, 1, tzinfo=LOCAL)
    events = core_client("c13", datetime(2025, 6, 23, tzinfo=LOCAL)) + core_client("gone", datetime(2025, 6, 23, tzinfo=LOCAL))
    past = [row for row in events if datetime.fromisoformat(row["event_time"]) < end]
    future = [row for row in events if datetime.fromisoformat(row["event_time"]) >= end]

    raw = tmp_path / "raw"
    write_group(raw, "train", end, past, [app_profile(client, end) for client in ("c13", "gone")])
    write_future(tmp_path / "future", raw, "train", future, end + timedelta(days=61), diverged=("gone",))

    state = tasks.group_history("train", raw_dir=raw, future_dir=tmp_path / "future")

    assert state.cutoff == pd.Timestamp(end).tz_convert("UTC")
    assert state.followup_end == pd.Timestamp(end + timedelta(days=61)).tz_convert("UTC")
    assert state.diverged == {"gone"}
    assert set(state.history["client_id"]) == {"c13"}

    # До T — то же, что по одной выгрузке; после T — из продолжения.
    alone = tasks.group_history("val", raw_dir=export(tmp_path, past, ["c13"], "alone"))
    pd.testing.assert_frame_equal(
        state.history[state.history["started_at"] <= state.cutoff].reset_index(drop=True),
        alone.history[alone.history["started_at"] <= state.cutoff].reset_index(drop=True),
    )
    assert labels(state, 14).loc["c13", "target"] == 1.0


# --- источники ---


CHURN = Path(__file__).resolve().parents[1] / "churn"


def test_feature_builders_never_read_lifecycle_or_truth() -> None:

    for name in ("features.py", "profile.py", "build.py", "train.py", "plus_usr.py", "target.py", "raw.py"):
        text = (CHURN / name).read_text(encoding="utf-8")
        assert "lifecycle" not in text, name

    pattern = re.compile(r"""["'/]truth\b""")
    assert [path.name for path in CHURN.glob("*.py") if pattern.search(path.read_text(encoding="utf-8"))] == []
