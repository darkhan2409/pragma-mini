from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from churn.build import KEYS, build
from churn.config import HORIZON, RECENT, cutoff
from churn.target import task_rows
from world import LOCAL, UTC, event, login, profile, profile_change, purchase, push, salary, write_future, write_group

END = datetime(2026, 4, 1, tzinfo=LOCAL)
T = END - HORIZON
AS_OF = END


def run(tmp_path, events, profiles, name="w") -> pd.DataFrame:
    root = tmp_path / name
    write_group(root / "raw", "val", END, events, profiles)
    build("val", raw_dir=root / "raw", data_dir=root / "data", reports_dir=root / "reports")
    return pd.read_parquet(root / "data" / "val" / "features.parquet").set_index("client_id")


def base(client: str) -> list[dict]:
    return [
        purchase(client, T - timedelta(days=40)),
        login(client, T - timedelta(days=3), session=f"{client}_s1"),
        salary(client, T - timedelta(days=20)),
        push(client, T - timedelta(days=5)),
    ]


def test_cutoff_leaves_a_full_window() -> None:
    assert T == datetime(2026, 1, 31, tzinfo=LOCAL)


def test_target_window_is_open_at_T_and_closed_at_the_horizon(tmp_path) -> None:
    events = []
    for client in ("at_T", "at_end", "after_end", "bank_only", "active"):
        events += base(client)
    events += [
        purchase("at_T", T),
        purchase("at_end", T + HORIZON),
        purchase("after_end", T + HORIZON + timedelta(microseconds=1000)),
        salary("bank_only", T + timedelta(days=10)),
        push("bank_only", T + timedelta(days=11)),
        login("active", T + timedelta(days=30)),
    ]
    profiles = [profile(c, AS_OF) for c in ("at_T", "at_end", "after_end", "bank_only", "active")]
    table = run(tmp_path, events, profiles)
    assert table.loc["at_T", "churn"] == 1
    assert table.loc["at_end", "churn"] == 0
    assert table.loc["after_end", "churn"] == 1
    assert table.loc["bank_only", "churn"] == 1
    assert table.loc["active", "churn"] == 0


def test_population_needs_a_client_action_before_T(tmp_path) -> None:
    events = base("regular")
    events += [salary("passive", T - timedelta(days=10)), push("passive", T - timedelta(days=2))]
    events += [login("newcomer", T + timedelta(days=2))]
    profiles = [profile(c, AS_OF) for c in ("regular", "passive", "newcomer")]
    table = run(tmp_path, events, profiles)
    assert list(table.index) == ["regular"]


def test_active90_is_a_client_action_in_the_90_days_before_T(tmp_path) -> None:
    """
    churn_active90 — строки популяции с действием клиента в [T − 90 дней, T):
    граница T − 90 дней входит, T — нет; события банка и будущее не в счёт.
    """
    old = T - timedelta(days=200)
    events = base("recent") + [
        purchase("edge", old),
        purchase("edge", T - RECENT),
        purchase("old", T - RECENT - timedelta(microseconds=1000)),
        salary("old", T - timedelta(days=10)),
        push("old", T - timedelta(days=2)),
        purchase("at_T", old),
        purchase("at_T", T),
        purchase("later", old),
        login("later", T + timedelta(days=5)),
    ]
    clients = ("recent", "edge", "old", "at_T", "later")
    table = run(tmp_path, events, [profile(c, AS_OF) for c in clients])

    # Все пятеро — в популяции: действие до T у каждого есть.
    assert sorted(table.index) == sorted(clients)
    assert table["active90"].to_dict() == {"at_T": False, "edge": True, "later": False, "old": False, "recent": True}
    assert (table["active90"] == (table["act_count_90"] > 0)).all()
    assert sorted(task_rows("churn_active90", table).index) == ["edge", "recent"]


def test_test_is_built_only_for_the_final_evaluation(monkeypatch) -> None:
    import churn.build as module

    # Сборка подменена: тест ничего не читает и не пишет.
    built: list[str] = []
    monkeypatch.setattr(module, "build", lambda group: built.append(group) or {})

    with pytest.raises(SystemExit):
        module.main(["test"])
    module.main(["test", "--final-test"])
    module.main(["val"])
    assert built == ["test", "val"]


def test_event_at_T_is_not_a_feature(tmp_path) -> None:
    events = base("c") + [purchase("c", T, amount=777)]
    table = run(tmp_path, events, [profile("c", AS_OF)])
    assert table.loc["c", "purchase_count_7"] == 0
    assert table.loc["c", "act_days_since_last"] == pytest.approx(3.0)


def test_window_edges(tmp_path) -> None:
    events = base("c") + [
        purchase("c", T - timedelta(days=7), amount=10),
        purchase("c", T - timedelta(days=7, microseconds=1000), amount=20),
    ]
    table = run(tmp_path, events, [profile("c", AS_OF)])
    assert table.loc["c", "purchase_count_7"] == 1
    assert table.loc["c", "purchase_count_30"] == 2
    assert table.loc["c", "purchase_sum_30"] == 30


def test_future_does_not_change_features(tmp_path) -> None:
    clients = ("a", "b", "c")
    past = [row for client in clients for row in base(client)]
    past += [profile_change("b", T - timedelta(days=100), "city", "Astana", "Almaty")]
    quiet = run(tmp_path, past, [profile(c, AS_OF) for c in clients], name="quiet")

    future = [
        purchase("a", T + timedelta(days=1), amount=999999),
        login("a", T + timedelta(days=90)),
        salary("b", T + timedelta(days=15)),
        push("c", T + timedelta(minutes=1), template="WB_MISS_YOU"),
        event("c", T + timedelta(days=5), "product_events", type="product_closed", reason="early_closure"),
        # Анкета после T поменялась: снимок на as_of уже новый, а событие
        # несёт прежнее значение.
        profile_change("a", T + timedelta(days=2), "family_status", "single", "married"),
        profile_change("b", T, "declared_income", "300000", "500000", source="application"),
    ]
    changed_profiles = [
        profile("a", AS_OF, family_status="married", contracts_count=9, active_contracts=7, holds_deposit=True, relationship_months=99),
        profile("b", AS_OF, declared_income=500000, credit_limit=150000.0, credit_utilization=0.9),
        profile("c", AS_OF, holds_credit_card=True),
    ]
    busy = run(tmp_path, past + future, changed_profiles, name="busy")

    columns = [name for name in quiet.columns if name not in KEYS]
    pd.testing.assert_frame_equal(quiet[columns], busy[columns])
    pd.testing.assert_series_equal(quiet["active90"], busy["active90"])
    assert (quiet["churn"] == 1).all() and busy.loc["a", "churn"] == 0


def test_profile_is_rolled_back_to_T(tmp_path) -> None:
    events = base("c") + [
        profile_change("c", T - timedelta(days=30), "region", "Astana", "Almaty"),
        profile_change("c", T + timedelta(days=5), "income_type", "unemployed", "employed", source="application"),
        profile_change("c", T + timedelta(days=9), "income_type", "employed", "self_employed", source="application"),
        profile_change("c", T + timedelta(days=7), "children", "0", "1", source="application"),
    ]
    snapshot = profile(
        "c",
        AS_OF,
        region="Almaty",
        income_type="self_employed",
        children=1,
        birth_date=datetime(1990, 1, 31).date(),
        lifelong=[
            {"type": "bank_registered", "event_time": (T - timedelta(days=100)).astimezone(UTC), "source_id": None},
            {"type": "app_registered", "event_time": (T + timedelta(days=1)).astimezone(UTC), "source_id": None},
        ],
    )
    table = run(tmp_path, events, [snapshot])
    row = table.loc["c"]
    assert row["region"] == "Almaty"
    assert row["income_type"] == "unemployed"
    assert row["children"] == 0
    # 31 января 2026 — ровно 36-й день рождения.
    assert row["age"] == 36
    # Работа по найму есть в записях, но вид дохода на T — безработный.
    assert np.isnan(row["job_tenure_months"])
    assert row["days_since_bank_registered"] == pytest.approx(100.0)
    assert np.isnan(row["days_since_app_registered"])


def test_snapshot_fields_are_not_features(tmp_path) -> None:
    table = run(tmp_path, base("c"), [profile("c", AS_OF)])
    for name in ("contracts_count", "active_contracts", "holds_credit_card", "holds_debit_card",
                 "holds_deposit", "credit_limit", "credit_utilization", "relationship_months", "birth_date", "as_of"):
        assert name not in table.columns


def test_job_tenure_uses_records_known_before_T(tmp_path) -> None:
    records = [
        {"start_date": datetime(2020, 1, 15).date(), "record_time": datetime(2020, 2, 1, tzinfo=UTC)},
        {"start_date": datetime(2025, 6, 1).date(), "record_time": (T + timedelta(days=3)).astimezone(UTC)},
    ]
    table = run(tmp_path, base("c"), [profile("c", AS_OF, employment=records)])
    # С 15.01.2020 по 31.01.2026 — 72 полных месяца; запись, о которой банк
    # узнал после T, не видна.
    assert table.loc["c", "job_tenure_months"] == 72


def test_feature_list_describes_every_column(tmp_path) -> None:
    root = tmp_path / "w"
    write_group(root / "raw", "val", END, base("c"), [profile("c", AS_OF)])
    build("val", raw_dir=root / "raw", data_dir=root / "data", reports_dir=root / "reports")
    table = pd.read_parquet(root / "data" / "val" / "features.parquet")
    listed = (root / "reports" / "features.md").read_text()
    for name in table.columns:
        if name not in KEYS:
            assert f"`{name}`" in listed


# ============================================================
# TRAIN: T — КОНЕЦ ВЫГРУЗКИ, TARGET — ПРОДОЛЖЕНИЕ
# ============================================================

TRAIN_END = datetime(2026, 1, 1, tzinfo=LOCAL)
TRAIN_T = TRAIN_END
FUTURE_END = TRAIN_END + HORIZON + timedelta(days=1)


def history(client: str) -> list[dict]:
    return [
        purchase(client, TRAIN_T - timedelta(days=40)),
        login(client, TRAIN_T - timedelta(days=3), session=f"{client}_s1"),
        salary(client, TRAIN_T - timedelta(days=20)),
    ]


def run_train(tmp_path, events, future, profiles, name="t", diverged=()) -> pd.DataFrame:
    root = tmp_path / name
    write_group(root / "raw", "train", TRAIN_END, events, profiles)
    write_future(root / "future", root / "raw", "train", future, FUTURE_END, diverged)
    build("train", raw_dir=root / "raw", data_dir=root / "data", reports_dir=root / "reports", future_dir=root / "future")
    return pd.read_parquet(root / "data" / "train" / "features.parquet").set_index("client_id")


def test_train_T_is_the_end_of_its_export_and_val_keeps_its_window(tmp_path) -> None:
    write_group(tmp_path / "raw", "train", TRAIN_END, history("c"), [profile("c", TRAIN_END)])
    write_group(tmp_path / "raw", "val", END, base("c"), [profile("c", AS_OF)])
    assert cutoff("train", tmp_path / "raw") == TRAIN_END
    assert cutoff("val", tmp_path / "raw") == END - HORIZON


def test_train_target_comes_from_the_continuation_with_strict_edges(tmp_path) -> None:
    clients = ("stays", "bank_only", "at_T", "at_end", "after_end", "silent")
    events = [row for client in clients for row in history(client)]
    future = [
        login("stays", TRAIN_T + timedelta(days=10), session="stays_f"),
        salary("bank_only", TRAIN_T + timedelta(days=5)),
        push("bank_only", TRAIN_T + timedelta(days=6)),
        purchase("at_T", TRAIN_T),
        purchase("at_end", TRAIN_T + HORIZON),
        purchase("after_end", TRAIN_T + HORIZON + timedelta(microseconds=1000)),
    ]
    table = run_train(tmp_path, events, future, [profile(c, TRAIN_END) for c in clients])
    assert table["churn"].to_dict() == {
        "after_end": 1, "at_T": 1, "at_end": 0, "bank_only": 1, "silent": 1, "stays": 0,
    }
    meta = json.loads((tmp_path / "t" / "data" / "train" / "meta.json").read_text())
    assert meta["T"] == TRAIN_T.isoformat() and meta["target_source"] == "future"
    assert meta["target_window"] == f"({TRAIN_T.isoformat()}, {(TRAIN_T + HORIZON).isoformat()}]"
    assert meta["feature_history_events_sha256"] == "synthetic-train"
    assert meta["feature_profile_sha256"] == "synthetic-profile-train"
    assert meta["target_events_sha256"] != meta["feature_history_events_sha256"]


def test_the_continuation_changes_the_target_but_not_features_or_population(tmp_path) -> None:
    """
    Любые события продолжения — действия, выбросы, другой ритм — меняют
    только target. Признаки, популяция и active90 берутся из истории до T.
    """
    clients = ("a", "b", "old")
    events = [row for client in ("a", "b") for row in history(client)]
    events += [purchase("old", TRAIN_T - timedelta(days=200))]
    profiles = [profile(c, TRAIN_END) for c in clients]
    quiet = run_train(tmp_path, events, [], profiles, name="quiet")
    busy = run_train(
        tmp_path,
        events,
        [
            purchase("a", TRAIN_T + timedelta(days=1), amount=99_999_999),
            login("b", TRAIN_T + timedelta(days=2), session="b_f"),
            login("old", TRAIN_T + timedelta(days=3), session="old_f"),
            purchase("old", TRAIN_T + timedelta(days=4)),
        ],
        profiles,
        name="busy",
    )
    columns = [name for name in quiet.columns if name != "churn"]
    pd.testing.assert_frame_equal(quiet[columns], busy[columns])
    assert (quiet["churn"] == 1).all() and (busy["churn"] == 0).all()
    # Давно молчавший клиент не становится активным от действий после T.
    assert not busy.loc["old", "active90"] and busy.loc["a", "active90"]


@pytest.mark.parametrize("damage", ["events", "profile", "file", "short", "missing"])
def test_a_continuation_that_does_not_continue_this_export_is_refused(tmp_path, damage) -> None:
    root = tmp_path / "w"
    write_group(root / "raw", "train", TRAIN_END, history("c"), [profile("c", TRAIN_END)])
    out = write_future(root / "future", root / "raw", "train", [login("c", TRAIN_T + timedelta(days=1))], FUTURE_END)
    record = json.loads((out / "future.json").read_text())
    if damage == "events":
        record["source"]["events_sha256"] = "previous-generation"
    elif damage == "profile":
        record["source"]["profile_sha256"] = "previous-generation"
    elif damage == "short":
        record["period_end"] = (TRAIN_T + HORIZON).isoformat()
    elif damage == "missing":
        (out / "future.json").unlink()
    if damage == "file":
        (out / "events.parquet").write_bytes(b"not the continuation")
    elif damage != "missing":
        (out / "future.json").write_text(json.dumps(record))
    message = {"events": "не этой выгрузки", "profile": "не этой выгрузки", "file": "не совпадает",
               "short": "не покрывает", "missing": "нет"}[damage]
    with pytest.raises(ValueError, match=message):
        build("train", raw_dir=root / "raw", data_dir=root / "data", reports_dir=root / "reports", future_dir=root / "future")


def test_train_history_at_T_is_refused(tmp_path) -> None:
    """
    Выгрузка train кончается в T. Событие в T в ней — уже не та
    выгрузка: оно попало бы в признаки рядом с target из продолжения.
    """
    events = history("c") + [purchase("c", TRAIN_T)]
    with pytest.raises(ValueError, match="не раньше T"):
        run_train(tmp_path, events, [], [profile("c", TRAIN_END)])


def test_clients_whose_continuation_diverged_are_excluded_not_labelled(tmp_path) -> None:
    """
    Прошлое клиента в продолжении разошлось с выгрузкой — его метка
    неизвестна: он исключён из строк train, а не получает churn = 1 от
    того, что в хвосте его нет.
    """
    clients = ("kept", "diverged")
    events = [row for client in clients for row in history(client)]
    table = run_train(tmp_path, events, [], [profile(c, TRAIN_END) for c in clients], diverged=("diverged",))
    assert list(table.index) == ["kept"] and table.loc["kept", "churn"] == 1
    meta = json.loads((tmp_path / "t" / "data" / "train" / "meta.json").read_text())
    assert meta["excluded_diverged_continuation"] == 1 and meta["population"] == 1

