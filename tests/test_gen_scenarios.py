from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime

import pytest

from src.generator import params as params_module
from src.generator import rng as rng_module
from src.generator.finance import invariants as invariants_module
from src.generator.life.scenarios import BY_NAME


# ============================================================
# СЦЕНАРНЫЕ ПРОВЕРКИ
# ============================================================
#
# Редкое состояние проверяется ДЕТЕРМИНИРОВАННЫМ сценарием:
# пресет поднимает интенсивность нужного механизма, и явление
# обязано появиться. Надеяться, что оно случайно выпадет на
# двух десятках клиентов, нельзя.
# ============================================================


CLIENTS = 16
COMMUNITY_SIZE = 8

BASE = {
    "merchants": {"catalog_scale": 0.05},
    "relationships": {"community_size": COMMUNITY_SIZE},
}


_CACHE: dict = {}


def _run(scenario_name: str | None = None, clients: int = CLIENTS) -> dict:
    """
    Небольшая популяция под конкретным пресетом.

    Один и тот же пресет не пересчитывается: розыгрыш
    детерминирован, и второй прогон дал бы ровно то же самое.
    """

    key = (scenario_name, clients)

    if key in _CACHE:
        return _CACHE[key]

    overrides = {section: dict(values) for section, values in BASE.items()}

    if scenario_name is not None:
        for section, values in BY_NAME[scenario_name].overrides.items():
            overrides.setdefault(section, {})
            overrides[section].update(values)

    settings = params_module.DEFAULT.with_overrides(overrides)

    params_module.activate(settings)
    rng_module.configure(42, settings.fingerprint())

    from src.generator.engine import run_community

    size = settings.relationships.community_size

    events: list = []
    truth: list = []
    clients_truth: list = []

    for community_id in range((clients + size - 1) // size):

        members = tuple(
            range(community_id * size + 1, min((community_id + 1) * size, clients) + 1)
        )

        if not members:
            continue

        result = run_community(community_id, members)

        events.extend(result.events)
        truth.extend(result.truth_events)
        clients_truth.extend(result.truth_clients)

    for row in events:
        row["payload"] = json.loads(row["payload"])

    by_client = defaultdict(list)

    for row in events:
        by_client[row["client_id"]].append(row)

    result = {
        "events": events,
        "types": Counter(row["event_type"] for row in events),
        "truth": truth,
        "clients": clients_truth,
        "by_client": by_client,
    }

    _CACHE[key] = result

    return result


@pytest.fixture(scope="module")
def baseline() -> dict:
    return _run(None)


def _reset_defaults() -> None:
    params_module.activate(params_module.DEFAULT)
    rng_module.configure(42, params_module.DEFAULT.fingerprint())


@pytest.fixture(autouse=True, scope="module")
def _restore():
    yield
    _reset_defaults()


# ------------------------------------------------------------
# ДЕНЬГИ
# ------------------------------------------------------------


def test_internal_transfer_has_two_sides(baseline):
    """
    Внутрибанковский перевод создаёт обе стороны с одним
    transfer_id, и полученные деньги видны получателю.
    """

    data = _run("transfer_pair")

    pairs = defaultdict(dict)

    for row in data["events"]:
        if row["event_type"] in ("p2p_out", "p2p_in") and row["payload"].get("status") == "approved":
            pairs[row["correlation_id"]][row["event_type"]] = row

    complete = [sides for sides in pairs.values() if len(sides) == 2]

    assert complete, "парных внутрибанковских переводов не оказалось"

    for sides in complete:

        out_row, in_row = sides["p2p_out"], sides["p2p_in"]

        assert out_row["payload"]["amount"] == in_row["payload"]["amount"]
        assert out_row["event_time"] <= in_row["event_time"]
        assert out_row["client_id"] != in_row["client_id"]
        assert in_row["payload"]["balance_after"] is not None

    # Полученные деньги влияют на последующие решения: у
    # получателя после зачисления есть расходные операции.
    moved_on = 0

    for sides in complete:

        in_row = sides["p2p_in"]

        later = [
            row
            for row in data["by_client"][in_row["client_id"]]
            if row["event_time"] > in_row["event_time"]
            and row["payload"].get("status") == "approved"
            and row["payload"].get("direction") == "debit"
        ]

        if later:
            moved_on += 1

    assert moved_on > 0


def test_transfer_shortfall_has_four_outcomes(baseline):
    """
    Нехватка средств не гарантируется искусственным внешним
    пополнением: возможны пополнение, уменьшение суммы, отказ
    клиента и отклонение банком.
    """

    data = _run("transfer_pair", clients=32)

    declined = [
        row
        for row in data["events"]
        if row["event_type"] in ("p2p_out", "transfer_out")
        and row["payload"].get("status") == "declined"
    ]

    assert declined, "отклонённых переводов не оказалось"

    for row in declined:
        assert row["payload"]["decline_reason"] == "insufficient_funds"
        assert row["payload"].get("balance_after") is None

    cancelled = [row for row in data["truth"] if row["kind"] == "transfer_intent"]

    topups = [
        row
        for row in data["events"]
        if row["event_type"] == "transfer_in"
        and row["payload"].get("reason") == "topup_before_transfer"
    ]

    assert cancelled or topups, "ни отказа клиента, ни пополнения не нашлось"


def test_money_is_conserved(baseline):

    problems = invariants_module.check_all(baseline["by_client"])

    assert not problems, [str(item) for item in problems[:5]]


# ------------------------------------------------------------
# ДОХОД
# ------------------------------------------------------------


def test_income_can_be_late_partial_or_missing(baseline):

    data = _run("income_late_and_missed")

    outcomes = Counter(
        row["key"] for row in data["truth"] if row["kind"] == "income_event"
    )

    assert outcomes["late"] > 0
    assert outcomes["partial"] > 0 or outcomes["partial_topup"] > 0

    credits = [row for row in data["events"] if row["event_type"] == "salary_credit"]

    assert credits

    by_client = defaultdict(list)

    for row in credits:
        by_client[row["client_id"]].append(row["event_time"])

    gaps = []

    for moments in by_client.values():
        moments.sort()
        gaps.extend((right - left).days for left, right in zip(moments, moments[1:]))

    assert gaps
    assert max(gaps) > 35, "пропущенных выплат не видно"


# ------------------------------------------------------------
# СТРЕСС
# ------------------------------------------------------------


def test_stress_leads_to_delinquency_and_recovery(baseline):

    data = _run("stress_recovery", clients=32)

    assert data["types"]["delinquency_registered"] > 0
    assert data["types"]["installment_missed"] > 0
    assert data["types"]["arrears_cleared"] > 0

    episodes = [row for row in data["truth"] if row["kind"] == "stress_start"]

    assert episodes

    triggers = {row["key"] for row in episodes}

    assert "job_loss" in triggers or "random_shock" in triggers

    # Восстановление: после погашения просрочки клиент снова
    # платит по графику.
    for client_id, rows in data["by_client"].items():

        cleared = [row for row in rows if row["event_type"] == "arrears_cleared"]

        if not cleared:
            continue

        moment = min(row["event_time"] for row in cleared)

        after = [
            row
            for row in rows
            if row["event_type"] == "installment_paid" and row["event_time"] > moment
        ]

        if after:
            return

    pytest.fail("после погашения просрочки платежей не нашлось")


# ------------------------------------------------------------
# МОШЕННИЧЕСТВО
# ------------------------------------------------------------


def test_fraud_chain_is_complete(baseline):
    """
    Компрометация карты: срабатывание, решение, блокировка,
    обращение и возврат по оспариванию связаны ссылками.
    """

    data = _run("card_compromise", clients=32)

    alerts = [row for row in data["events"] if row["event_type"] == "fraud_alert"]

    assert alerts

    ids = {row["event_id"]: row for row in data["events"]}

    for row in alerts:
        assert row["payload"]["cause_event_id"] in ids

    decisions = [row for row in data["events"] if row["event_type"] == "fraud_decision"]

    assert decisions

    for row in decisions:
        assert row["payload"]["cause_event_id"] in ids

    assert data["types"]["card_blocked"] > 0
    assert data["types"]["case_opened"] > 0
    assert data["types"]["chargeback"] > 0

    for row in data["events"]:
        if row["event_type"] == "chargeback":
            assert row["payload"]["cause_event_id"] in ids


def test_false_positive_is_confirmed_and_released(baseline):

    data = _run("false_positive", clients=32)

    assert data["types"]["card_blocked"] > 0
    assert data["types"]["card_unblocked"] > 0

    for client_id, rows in data["by_client"].items():

        blocked = [row for row in rows if row["event_type"] == "card_blocked"]
        released = [row for row in rows if row["event_type"] == "card_unblocked"]

        if blocked and released:
            assert min(row["event_time"] for row in released) >= min(
                row["event_time"] for row in blocked
            )
            return

    pytest.fail("не нашлось карты, которую заблокировали и вернули в строй")


def test_no_fraud_flag_in_raw(baseline):

    data = _run("card_compromise")

    keys = {name for row in data["events"] for name in row["payload"]}

    assert "fraud_persona" not in keys
    assert "is_fraudster" not in keys

    kinds = {row["kind"] for row in data["truth"]}

    assert "fraud_episode" in kinds


# ------------------------------------------------------------
# ПАУЗЫ И ПРИВЫЧКИ
# ------------------------------------------------------------


def test_pause_silences_the_client_but_not_the_bank(baseline):
    """
    Полная пауза гасит ровно те потоки, которые объявлены в
    params.activity.pause_silences: траты, сессии, переводы,
    наличные и счета. Банк при этом продолжает работать.
    """

    data = _run("pause_and_return", clients=32)

    # Ровно те потоки, которые объявлены в pause_silences["full"]:
    # траты, сессии, переводы, наличные и счета. Пополнение
    # только что открытого вклада это шаг договора, а не поток
    # повседневной активности, и в список не входит.
    silenced_types = {
        "purchase",
        "cash_withdrawal",
        "cash_deposit",
        "p2p_out",
        "transfer_out",
        "bill_payment",
        "app_screen",
        "app_operation",
        "banner_clicked",
    }

    pauses = [row for row in data["truth"] if row["kind"] == "pause_start"]

    assert pauses

    checked = 0
    bank_side = 0

    for row in pauses:

        value = json.loads(row["value"])

        assert value["planned_end"]

        start = row["ts"]
        end = datetime.fromisoformat(value["actual_end"])

        inside = [
            item
            for item in data["by_client"][row["client_id"]]
            if start <= item["event_time"] < end
        ]

        if not inside:
            continue

        checked += 1

        leaked = [
            item
            for item in inside
            if item["event_type"] in silenced_types
            and item["change_initiator"] == "client"
        ]

        assert not leaked, [item["event_type"] for item in leaked[:5]]

        bank_side += sum(1 for item in inside if item["change_initiator"] != "client")

    assert checked > 0, "ни одна пауза не пересеклась с событиями"
    assert bank_side > 0, "банк замолчал вместе с клиентом"


def test_returning_client_is_marked_in_truth(baseline):

    data = _run("pause_and_return", clients=32)

    kinds = Counter(row["kind"] for row in data["truth"])

    assert kinds["pause_start"] > 0
    assert kinds["pause_end"] > 0


def test_habits_repeat_at_favourite_outlets(baseline):

    data = _run("habitual_client", clients=16)

    by_client = defaultdict(Counter)

    for row in data["events"]:
        if row["event_type"] == "purchase" and row["payload"].get("outlet_id"):
            by_client[row["client_id"]][row["payload"]["outlet_id"]] += 1

    shares = []

    for counter in by_client.values():
        total = sum(counter.values())
        if total < 30:
            continue
        shares.append(sum(count for _, count in counter.most_common(3)) / total)

    assert shares, "покупок слишком мало для проверки привычек"

    assert max(shares) > 0.25, "любимых точек не видно"


def test_counterparties_repeat(baseline):

    data = _run("transfer_pair", clients=16)

    by_client = defaultdict(Counter)

    for row in data["events"]:
        if row["event_type"] in ("p2p_out", "transfer_out"):
            name = row["payload"].get("counterparty")
            if name:
                by_client[row["client_id"]][name] += 1

    repeated = [
        counter
        for counter in by_client.values()
        if sum(counter.values()) >= 5 and any(count > 1 for count in counter.values())
    ]

    assert repeated, "повторных контрагентов не оказалось"


# ------------------------------------------------------------
# ПРОДУКТЫ
# ------------------------------------------------------------


def test_products_open_through_the_funnel(baseline):

    data = _run("product_migration", clients=32)

    assert data["types"]["application_submitted"] > 0
    assert data["types"]["application_decision"] > 0
    assert data["types"]["product_opened"] > 0

    opened = [row for row in data["events"] if row["event_type"] == "product_opened"]

    with_application = [row for row in opened if row["link_type"] == "application"]

    assert with_application


def test_contract_keeps_its_version_after_a_tariff_change(baseline):
    """
    Смена тарифа действующему договору это отдельное событие,
    а не переписывание прошлого.
    """

    data = _run("product_migration", clients=32)

    repriced = [
        row
        for row in data["events"]
        if row["event_type"] in ("product_repriced", "contract_terms_changed")
    ]

    if not repriced:
        pytest.skip("в этой выборке тариф действующим договорам не менялся")

    for row in repriced:
        assert row["payload"]["contract_id"]
        assert row["effective_at"] <= row["event_time"]


def test_forbidden_fields_never_appear(baseline):

    from src.generator.config import FORBIDDEN_RAW_FIELDS

    keys = {name for row in baseline["events"] for name in row["payload"]}

    assert not (keys & FORBIDDEN_RAW_FIELDS)

    columns = set(baseline["events"][0])

    assert not (columns & FORBIDDEN_RAW_FIELDS)
