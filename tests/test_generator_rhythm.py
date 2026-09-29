from __future__ import annotations

import json
from collections import Counter
from datetime import datetime

import pyarrow.parquet as pq
import pytest

from src.generator import config, emit
from src.generator import params as params_module
from src.generator import rng as rng_module
from src.generator.life import calendar as cal
from src.generator.life import income


# ============================================================
# ИДЕЯ
# ============================================================
#
# Два ритма, которые выдавали генератор:
#
#   - счёт по автоплатежу проводился в момент-формулу от номера
#     счёта (09:00, 10:07, 11:14), и три четверти платежей ленты
#     приходились на три момента суток. Теперь час и минута свои у
#     каждого счёта и месяца;
#   - зарплата росла на доли процента каждый месяц. Теперь
#     индексация — ступенька раз в год в месяц индексации
#     (у работодателя общий), а повышение — ступенька на годовщине;
#   - подписка выбиралась с возвращением, и один сервис списывался
#     дважды в месяц с разными суммами. Теперь без повторов.
# ============================================================


START = datetime(2024, 1, 1)
END = datetime(2025, 1, 1)


@pytest.fixture
def active():

    saved = (config.HISTORY_START, config.HISTORY_END, config.REGISTRATION_END)

    config.activate_horizon(START, datetime(2026, 1, 1))

    settings = emit._build_params(None, None, 4)
    params_module.activate(settings)
    rng_module.configure(1, settings.fingerprint(), 42)

    yield settings

    config.activate_horizon(*saved)
    rng_module.clear_caches()


def stream(payer: str, stream_id: str = "salary-1") -> income.IncomeStream:
    return income.IncomeStream(
        stream_id=stream_id, kind="salary", payer=payer, schedule="monthly", payday=10,
        landing="card", base_amount=300_000, valid_from=START, valid_to=None,
    )


def test_a_salary_changes_only_in_steps(active):

    item = stream("emp_42")
    month = income._indexation_month(item)

    months = [cal.month_start(datetime(2024 + index // 12, index % 12 + 1, 1)) for index in range(24)]
    amounts = [income._amount_at(item, moment, 7, active.income) for moment in months]

    changes = [
        moment for moment, before, after in zip(months[1:], amounts, amounts[1:]) if before != after
    ]

    assert changes, "за два года индексация обязана случиться"

    for moment in changes:
        anniversary = (cal.month_index(moment) - cal.month_index(START)) % 12 == 0
        assert moment.month == month or anniversary, moment

    # Не каждый месяц: ступенек не больше двух в год.
    assert len(changes) <= 4


def test_employees_of_one_employer_are_indexed_in_one_month(active):

    assert income._indexation_month(stream("emp_7", "a")) == income._indexation_month(stream("emp_7", "b"))


@pytest.fixture(scope="module")
def tape(tmp_path_factory):

    out = tmp_path_factory.mktemp("rhythm") / "tape"

    emit.generate_dataset(
        total_clients=40, out_dir=out, seed=77, world_seed=42, history_start=START,
        history_end=END, registration_end=END, workers=1, community_size=4, quiet=True,
    )

    rows = pq.read_table(out / "events.parquet", columns=["client_id", "event_time", "payload"]).to_pylist()

    return [(row["client_id"], row["event_time"], json.loads(row["payload"])) for row in rows]


def test_bill_payments_are_not_three_moments_of_the_day(tape):

    moments = Counter(
        event_time[11:16] for _, event_time, payload in tape if payload["type"] == "bill_payment"
    )

    total = sum(moments.values())

    assert total >= 30
    assert moments.most_common(1)[0][1] / total < 0.1, moments.most_common(3)



def test_a_client_never_pays_one_subscription_service_twice_a_month(tape):

    charges = Counter(
        (client_id, event_time[:7], payload["merchant_name"].casefold())
        for client_id, event_time, payload in tape
        if payload["type"] == "purchase" and payload.get("merchant_category") == "subscription"
        and payload.get("merchant_name")
    )

    assert charges, "в ленте есть подписки с названием"
    assert max(charges.values()) == 1, charges.most_common(3)
