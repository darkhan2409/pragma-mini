"""
Календарная глубина recent-окон и судьба наблюдаемых связей.

Модель здесь не участвует. Проверяется арифметика окна и то,
что правила связей ловят ровно то, что обещают: подписку с
меняющимся городом ловят, а договор без воронки потерянной
связью не считают.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.model.history_coverage import (
    COLUMNS,
    LINK_RULES,
    WINDOWS,
    Timeline,
    link_stats,
    load_timeline,
    render_link_rules,
    run_history_coverage,
    window_stats,
)


BASE = np.datetime64("2025-01-01T00:00:00", "s")

HOUR = np.timedelta64(1, "h")
DAY = np.timedelta64(1, "D")


# ============================================================
# СИНТЕТИЧЕСКАЯ ЛЕНТА
# ============================================================


def timeline(events: list[dict], examples: list[tuple[int, str, int]]) -> Timeline:
    """
    Лента из списка словарей и примеры вида (клиент, cutoff, seq_end).
    """

    columns = {name: [] for name in COLUMNS}

    for position, event in enumerate(events):
        for name in COLUMNS:
            default = position if name == "seq" else None
            columns[name].append(event.get(name, default))

    made = {}

    for name in COLUMNS:
        if name == "ts":
            made[name] = np.array(columns[name], dtype="datetime64[s]")
        elif name in ("client_id", "seq"):
            made[name] = np.array(columns[name], dtype=np.int64)
        else:
            made[name] = np.array(columns[name], dtype=object)

    starts: dict[int, int] = {}
    ends: dict[int, int] = {}

    for position, client in enumerate(made["client_id"].tolist()):
        starts.setdefault(int(client), position)
        ends[int(client)] = position + 1

    picked = {
        "client_id": np.array([item[0] for item in examples], dtype=np.int64),
        "cutoff": np.array([item[1] for item in examples], dtype="datetime64[s]"),
        "seq_end": np.array([item[2] for item in examples], dtype=np.int64),
    }

    return Timeline(sorted(starts), starts, ends, made, picked)


def plain(client: int, offset_days: float, kind: str = "transaction", **extra) -> dict:
    return {
        "client_id": client,
        "ts": BASE + np.timedelta64(int(offset_days * 86400), "s"),
        "event_type": kind,
        **extra,
    }


# ============================================================
# ОКНО
# ============================================================


def test_window_keeps_the_last_events():
    events = [plain(1, day) for day in range(5)]

    made = timeline(events, [(1, "2025-01-06", 5)])

    stats = window_stats(made, 3)

    assert stats["used_events"]["p50"] == 3
    assert stats["original_events"]["p50"] == 5
    assert stats["kept_events_share"]["p50"] == pytest.approx(0.6)
    assert stats["truncated_examples_share"] == 1.0

    # Первое сохранённое событие это третье по счёту, 3 января.
    assert stats["days_first_to_cutoff"]["p50"] == pytest.approx(3.0)
    assert stats["days_last_to_cutoff"]["p50"] == pytest.approx(1.0)
    assert stats["days_span"]["p50"] == pytest.approx(2.0)


def test_full_window_keeps_everything():
    events = [plain(1, day) for day in range(5)]

    stats = window_stats(timeline(events, [(1, "2025-01-06", 5)]), None)

    assert stats["truncated_examples_share"] == 0.0
    assert stats["dropped_events_share"] == 0.0
    assert stats["kept_events_share"]["p50"] == pytest.approx(1.0)
    assert stats["days_first_to_cutoff"]["p50"] == pytest.approx(5.0)


def test_window_larger_than_history_is_not_truncation():
    events = [plain(1, day) for day in range(3)]

    stats = window_stats(timeline(events, [(1, "2025-01-06", 3)]), 128)

    assert stats["truncated_examples_share"] == 0.0
    assert stats["used_events"]["p50"] == 3


def test_future_events_never_enter_the_window():
    """
    История это префикс seq < seq_end; события после cutoff в
    него не входят по построению.
    """

    events = [plain(1, day) for day in range(6)]

    # seq_end = 3, значит события 3, 4 и 5 января лежат после cutoff.
    made = timeline(events, [(1, "2025-01-04", 3)])

    stats = window_stats(made, None)

    # Остались 1, 2 и 3 января; последнее событие в сутках от cutoff.
    assert stats["used_events"]["p50"] == 3
    assert stats["days_last_to_cutoff"]["p50"] == pytest.approx(1.0)
    assert stats["days_first_to_cutoff"]["p50"] == pytest.approx(3.0)

    cutoff = made.examples["cutoff"][0]

    span = made.slice_of(1, 3)

    assert bool((made.columns["ts"][span] < cutoff).all())


def test_equal_timestamps_are_ordered_by_position():
    """
    Окно берёт последние N по порядку ленты, а не по времени:
    у совпадающих меток порядок задаёт seq.
    """

    events = [plain(1, 0.0) for _ in range(4)]

    stats = window_stats(timeline(events, [(1, "2025-01-02", 4)]), 2)

    assert stats["used_events"]["p50"] == 2
    assert stats["days_span"]["p50"] == pytest.approx(0.0)


def test_empty_history_is_counted_separately():
    events = [plain(1, 0.0)]

    stats = window_stats(timeline(events, [(1, "2025-01-02", 0)]), 128)

    assert stats["n_empty_histories"] == 1
    assert stats["used_events"]["p50"] == 0
    assert stats["days_first_to_cutoff"]["p50"] is None


def test_event_type_counts_sum_to_the_window():
    events = [
        plain(1, 0, "transaction"),
        plain(1, 1, "app_screen"),
        plain(1, 2, "transaction"),
        plain(1, 3, "banner"),
    ]

    stats = window_stats(timeline(events, [(1, "2025-01-06", 4)]), 3)

    counts = {name: item["p50"] for name, item in stats["events_by_type"].items()}

    assert sum(value for value in counts.values() if value) == 3
    assert counts["transaction"] == 1
    assert counts["app_screen"] == 1
    assert counts["banner"] == 1


def test_truncated_examples_and_dropped_events_are_different_numbers():
    """
    «Обрезаны все истории» и «отброшено почти всё» это разные
    утверждения, и путать их нельзя.
    """

    events = [plain(1, day) for day in range(100)] + [plain(2, day) for day in range(3)]

    made = timeline(events, [(1, "2025-06-01", 100), (2, "2025-06-01", 3)])

    stats = window_stats(made, 2)

    assert stats["truncated_examples_share"] == 1.0
    assert stats["dropped_events_share"] == pytest.approx(1 - 4 / 103)

    assert stats["truncated_examples_share"] != stats["dropped_events_share"]


# ============================================================
# СВЯЗИ
# ============================================================


def subscription(client: int, day: float, mcc: str, amount: int, city: str) -> dict:
    return plain(
        client,
        day,
        "transaction",
        transaction__mcc=mcc,
        transaction__amount=amount,
        transaction__merchant_city=city,
        transaction__is_subscription=True,
        transaction__is_online=True,
    )


def test_subscription_links_across_changing_cities():
    events = [
        subscription(1, 0, "6300", 31520, "Almaty"),
        subscription(1, 30, "6300", 31520, "Astana"),
        subscription(1, 60, "6300", 31520, "Almaty"),
    ]

    stats = link_stats(timeline(events, [(1, "2025-04-01", 3)]), (None,))

    cell = stats["subscription_repeat"]["windows"]["full"]

    assert cell["successors"] == 3
    assert cell["with_predecessor"] == 2
    assert cell["no_predecessor"] == 1
    assert cell["inside_share"] == pytest.approx(1.0)


def test_a_different_amount_is_a_different_series():
    events = [
        subscription(1, 0, "6300", 31520, "Almaty"),
        subscription(1, 30, "6300", 9260, "Almaty"),
    ]

    stats = link_stats(timeline(events, [(1, "2025-04-01", 2)]), (None,))

    cell = stats["subscription_repeat"]["windows"]["full"]

    assert cell["with_predecessor"] == 0
    assert cell["no_predecessor"] == 2


def test_a_predecessor_left_of_the_window_is_lost():
    events = [
        subscription(1, 0, "6300", 31520, "Almaty"),
        plain(1, 1, "banner"),
        plain(1, 2, "banner"),
        subscription(1, 30, "6300", 31520, "Astana"),
    ]

    made = timeline(events, [(1, "2025-04-01", 4)])

    stats = link_stats(made, (2, None))

    tight = stats["subscription_repeat"]["windows"]["2"]

    assert tight["successors"] == 1
    assert tight["cut_off"] == 1
    assert tight["inside"] == 0
    assert tight["cut_off_share"] == pytest.approx(1.0)

    whole = stats["subscription_repeat"]["windows"]["full"]

    assert whole["inside"] == 1
    assert whole["cut_off"] == 0


def test_an_event_without_a_predecessor_is_in_neither_share():
    events = [subscription(1, 0, "6300", 31520, "Almaty")]

    cell = link_stats(timeline(events, [(1, "2025-04-01", 1)]), (None,))[
        "subscription_repeat"
    ]["windows"]["full"]

    assert cell["successors"] == 1
    assert cell["with_predecessor"] == 0
    assert cell["no_predecessor"] == 1
    assert cell["inside_share"] is None


def test_funnel_session_links_screens_by_identifier():
    events = [
        plain(1, 0, "app_screen", app_screen__session_id="a", app_screen__funnel_stage="view"),
        plain(1, 0.001, "app_screen", app_screen__session_id="a", app_screen__funnel_stage="application"),
        plain(1, 0.002, "app_screen", app_screen__session_id="b", app_screen__funnel_stage="view"),
    ]

    cell = link_stats(timeline(events, [(1, "2025-04-01", 3)]), (None,))["funnel_session"][
        "windows"
    ]["full"]

    assert cell["successors"] == 3
    assert cell["inside"] == 1
    assert cell["no_predecessor"] == 2


def test_browse_screens_are_not_part_of_the_funnel_rule():
    events = [
        plain(1, 0, "app_screen", app_screen__session_id="a"),
        plain(1, 0.001, "app_screen", app_screen__session_id="a"),
    ]

    cell = link_stats(timeline(events, [(1, "2025-04-01", 2)]), (None,))["funnel_session"][
        "windows"
    ]["full"]

    assert cell["successors"] == 0


def test_contract_floored_to_midnight_is_still_found():
    """
    timestamp_quality = date_only опускает договор до полуночи,
    и он встаёт в ленте РАНЬШЕ одобрившего экрана. Одностороннее
    окно потеряло бы эту связь.
    """

    events = [
        plain(
            1,
            0.0,
            "product_event",
            product_event__product_type="cash_loan",
        ),
        plain(
            1,
            0.5,
            "app_screen",
            app_screen__session_id="a",
            app_screen__funnel_stage="approved",
            app_screen__product="cash_loan",
        ),
    ]

    cell = link_stats(timeline(events, [(1, "2025-04-01", 2)]), (None,))[
        "funnel_to_contract"
    ]["windows"]["full"]

    assert cell["successors"] == 1
    assert cell["with_predecessor"] == 1


def test_a_contract_without_a_funnel_is_not_a_lost_link():
    events = [plain(1, 0.0, "product_event", product_event__product_type="debit_card")]

    cell = link_stats(timeline(events, [(1, "2025-04-01", 1)]), (None,))[
        "funnel_to_contract"
    ]["windows"]["full"]

    assert cell["successors"] == 1
    assert cell["no_predecessor"] == 1
    assert cell["cut_off"] == 0


def test_operation_follows_a_screen_of_its_domain():
    events = [
        plain(1, 0.0, "app_screen", app_screen__firebase_screen="s_401_loan_calc"),
        plain(1, 20 / 86400, "app_operation", app_operation__domain="loans"),
    ]

    cell = link_stats(timeline(events, [(1, "2025-04-01", 2)]), (None,))[
        "screen_to_operation"
    ]["windows"]["full"]

    assert cell["successors"] == 1
    assert cell["with_predecessor"] == 1


def test_login_is_not_attributed_to_a_screen():
    events = [
        plain(1, 0.0, "app_screen", app_screen__firebase_screen="s_401_loan_calc"),
        plain(1, 5 / 86400, "app_operation", app_operation__domain="auth"),
    ]

    cell = link_stats(timeline(events, [(1, "2025-04-01", 2)]), (None,))[
        "screen_to_operation"
    ]["windows"]["full"]

    assert cell["successors"] == 0


def test_a_late_operation_gets_no_predecessor():
    events = [
        plain(1, 0.0, "app_screen", app_screen__firebase_screen="s_401_loan_calc"),
        plain(1, 120 / 86400, "app_operation", app_operation__domain="loans"),
    ]

    cell = link_stats(timeline(events, [(1, "2025-04-01", 2)]), (None,))[
        "screen_to_operation"
    ]["windows"]["full"]

    assert cell["successors"] == 1
    assert cell["no_predecessor"] == 1


def test_links_never_look_past_the_cutoff():
    events = [
        subscription(1, 0, "6300", 31520, "Almaty"),
        subscription(1, 30, "6300", 31520, "Almaty"),
    ]

    # seq_end = 1: второе списание лежит за cutoff.
    cell = link_stats(timeline(events, [(1, "2025-01-15", 1)]), (None,))[
        "subscription_repeat"
    ]["windows"]["full"]

    assert cell["successors"] == 1
    assert cell["with_predecessor"] == 0


def test_small_samples_are_flagged_as_unreliable():
    events = [
        subscription(1, 0, "6300", 31520, "Almaty"),
        subscription(1, 30, "6300", 31520, "Almaty"),
    ]

    cell = link_stats(timeline(events, [(1, "2025-04-01", 2)]), (None,))[
        "subscription_repeat"
    ]["windows"]["full"]

    assert cell["reliable"] is False


# ============================================================
# ПРАВИЛА И ОТЧЁТ
# ============================================================


def test_every_rule_declares_its_status():
    statuses = {rule["status"] for rule in LINK_RULES}

    assert statuses <= {"identifier", "series_identifier", "heuristic", "weak_heuristic"}

    for rule in LINK_RULES:
        assert rule["key"]
        assert rule["why"]
        assert rule["not_proof"]


def test_rules_render_with_their_limits():
    text = render_link_rules({"link_rules": list(LINK_RULES)})

    assert "Чего в данных нет" in text
    assert "идентификатора магазина" in text

    for rule in LINK_RULES:
        assert rule["name"] in text


# ============================================================
# РЕАЛЬНЫЕ ДАННЫЕ
# ============================================================


@pytest.fixture(scope="module")
def coverage(prep_run, tmp_path_factory):
    out = tmp_path_factory.mktemp("coverage")

    report = run_history_coverage(
        prep_run["processed"],
        out,
        splits={"train": 4, "val_time": 4},
        windows=(8, 64, None),
        quiet=True,
    )

    return {"report": report, "out": out}


def test_real_splits_are_measured(coverage):
    report = coverage["report"]

    assert set(report["splits"]) == {"train", "val_time"}

    for item in report["splits"].values():
        assert item["clients"] == 4
        assert item["examples"] > 0
        assert item["events_in_memory"] > 0


def test_tighter_windows_drop_more(coverage):
    windows = coverage["report"]["coverage"]["train"]

    assert windows["8"]["dropped_events_share"] > windows["64"]["dropped_events_share"]
    assert windows["full"]["dropped_events_share"] == 0.0

    assert windows["8"]["days_first_to_cutoff"]["p50"] < windows["full"]["days_first_to_cutoff"]["p50"]


def test_links_are_reported_for_every_window(coverage):
    links = coverage["report"]["links"]["train"]

    assert set(links) == {rule["name"] for rule in LINK_RULES}

    for section in links.values():
        assert set(section["windows"]) == {"8", "64", "full"}
        for cell in section["windows"].values():
            assert cell["inside"] + cell["cut_off"] == cell["with_predecessor"]


def test_artefacts_are_written(coverage):
    out = coverage["out"]

    for name in ("history_coverage.json", "history_coverage.md", "link_rules.md"):
        assert (out / name).exists(), name

    text = (out / "history_coverage.md").read_text(encoding="utf-8")

    assert "Календарная глубина" in text
    assert "обрезано примеров" in text
    assert "отброшено событий" in text

    saved = json.loads((out / "history_coverage.json").read_text(encoding="utf-8"))

    assert saved["windows"] == ["8", "64", "full"]
