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
            # Обслуживание кредита это обязательство, а не
            # повседневная активность: в паузе клиент всё равно
            # заводит деньги на счёт к дате платежа.
            and item["payload"].get("reason") != "topup_before_installment"
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


# ------------------------------------------------------------
# КРЕДИТ КАК ПОВЕДЕНИЕ
# ------------------------------------------------------------


def test_debt_service_rule_rejects_oversized_requests(baseline):
    """
    Долговая нагрузка это правило, а не ярлык после отказа:
    при запредельном запросе отказ приходит именно по ней.
    """

    data = _run("dsr_reject", clients=32)

    decisions = [row for row in data["events"] if row["event_type"] == "application_decision"]

    assert decisions, "решений по заявкам не оказалось"

    reasons = Counter(
        row["payload"].get("reject_reason")
        for row in decisions
        if row["payload"].get("decision") != "approved"
    )

    assert reasons["debt_service_ratio"] > 0

    approved = [row for row in decisions if row["payload"].get("decision") == "approved"]

    credit = [
        row
        for row in approved
        if row["payload"].get("requested_amount")
        and row["payload"].get("requested_term")
    ]

    # Кредитные заявки при нулевой допустимой нагрузке не проходят.
    assert len(credit) < len(decisions)


def test_disciplined_borrower_pays_on_time(baseline):

    data = _run("credit_discipline", clients=32)

    assert data["types"]["installment_paid"] > 0

    paid = [row for row in data["events"] if row["event_type"] == "installment_paid"]

    with_cause = [row for row in paid if row["payload"].get("cause_event_id")]

    assert with_cause, "платёж не ссылается на выставленный взнос"

    # Пропусков заметно меньше, чем платежей.
    assert data["types"]["installment_missed"] < data["types"]["installment_paid"]


def test_refinance_closes_previous_loans(baseline):

    data = _run("refinance_closes_debt", clients=32)

    closed = [
        row
        for row in data["events"]
        if row["event_type"] == "loan_closed" and row["payload"].get("reason") == "refinanced"
    ]

    if not closed:
        pytest.skip("в этой выборке рефинансирование не состоялось")

    for row in closed:

        client = row["client_id"]

        disbursements = [
            item
            for item in data["by_client"][client]
            if item["event_type"] == "loan_disbursement"
            and item["event_time"] <= row["event_time"]
        ]

        assert disbursements, "закрытие рефинансированием без выдачи"

        # Долг гасится деньгами, а не списывается молча.
        payments = [
            item
            for item in data["by_client"][client]
            if item["event_type"] == "loan_payment"
            and item["payload"].get("contract_id") == row["payload"]["contract_id"]
        ]

        assert payments


def test_money_comes_in_from_outside(baseline):

    data = _run("inbound_money", clients=16)

    inbound = [
        row
        for row in data["events"]
        if row["event_type"] == "transfer_in" and row["payload"].get("reason") == "inbound"
    ]

    assert inbound, "входящих переводов извне не оказалось"

    for row in inbound[:20]:
        assert row["payload"]["counterparty"]
        assert row["payload"]["balance_after"] is not None
        assert row["change_initiator"] == "external_source"


# ------------------------------------------------------------
# МОШЕННИЧЕСТВО БЕЗ ГОТОВОГО ПРИЗНАКА
# ------------------------------------------------------------


def test_fraud_has_no_constant_merchant(baseline):
    """
    Мошенническая покупка не помечена постоянным именем точки и
    единственным MCC: это был бы готовый признак для модели.
    """

    data = _run("card_compromise", clients=32)

    names = {
        row["payload"].get("merchant_name")
        for row in data["events"]
        if row["event_type"] == "purchase"
    }

    assert "UNKNOWN MERCHANT" not in names

    alerts = [row for row in data["events"] if row["event_type"] == "fraud_alert"]

    assert alerts

    causes = {row["payload"]["cause_event_id"] for row in alerts}

    mccs = {
        row["payload"].get("mcc")
        for row in data["events"]
        if row["event_id"] in causes
    }

    assert len(mccs) > 1, f"все мошеннические операции в одном MCC: {mccs}"


def test_transfer_fraud_is_a_transfer(baseline):

    data = _run("transfer_fraud", clients=32)

    alerts = [row for row in data["events"] if row["event_type"] == "fraud_alert"]

    assert alerts

    subjects = {row["payload"].get("subject") for row in alerts}

    assert subjects - {"card"}, f"перевод оформлен как карта: {subjects}"

    causes = {row["payload"]["cause_event_id"] for row in alerts}

    kinds = {row["event_type"] for row in data["events"] if row["event_id"] in causes}

    assert "transfer_out" in kinds


# ------------------------------------------------------------
# КАРТА РАССРОЧКИ
# ------------------------------------------------------------


def test_card_purchases_become_a_debt(baseline):
    """
    Покупка по карте рассрочки делится на части, наличные копят
    проценты, а платёж по выписке гасит долг переводом со
    своего счёта.
    """

    data = _run("card_installments", clients=32)

    statements = [
        row
        for row in data["events"]
        if row["event_type"] == "installment_due"
        and row["payload"].get("reason") == "card_statement"
    ]

    if not statements:
        pytest.skip("в этой выборке карт рассрочки не оказалось")

    for row in statements[:20]:
        assert row["payload"]["amount_due"] > 0
        assert row["payload"]["contract_id"]

    payments = [
        row
        for row in data["events"]
        if row["event_type"] == "loan_payment"
        and row["payload"].get("reason") == "card_statement"
    ]

    for row in payments[:20]:
        # Обе стороны перевода между своими счетами помечены,
        # иначе деньги клиента исчезали бы.
        assert row["payload"]["counterparty"] == "own_account"

    interest = [
        row
        for row in data["events"]
        if row["event_type"] == "fee_charge"
        and row["payload"].get("accrual_period")
        and row["payload"].get("cause_event_id") is None
    ]

    assert interest, "начислений по договору не оказалось"


def test_dispute_does_not_require_a_block(baseline):
    """
    Оспорить чужую операцию клиент может и тогда, когда банк
    карту не заблокировал: деньги ушли, спор идёт с точкой.

    Раньше возврат был вложен в ветку блокировки и не случался
    ни разу за всю историю.
    """

    data = _run("fraud_chargeback", clients=32)

    assert data["types"]["card_blocked"] == 0, "в этом пресете банк не блокирует"
    assert data["types"]["chargeback"] > 0

    ids = {row["event_id"]: row for row in data["events"]}

    for row in data["events"]:

        if row["event_type"] != "chargeback":
            continue

        cause = ids[row["payload"]["cause_event_id"]]

        # Возвращают только то, что действительно списали, и
        # ровно столько, сколько списали.
        assert cause["payload"]["status"] == "approved"
        assert row["payload"]["amount"] == cause["payload"]["amount"]

    # Одна операция не возвращается дважды.
    returned = Counter(row["payload"]["cause_event_id"]
                       for row in data["events"] if row["event_type"] == "chargeback")

    assert max(returned.values()) == 1


def test_client_freeze_is_temporary(baseline):
    """
    Временную заморозку клиент ставит сам и сам же снимает.
    """

    data = _run("card_freeze", clients=32)

    blocked = [row for row in data["events"] if row["event_type"] == "card_blocked"]

    assert blocked

    assert all(row["payload"]["reason"] == "client_freeze" for row in blocked)
    assert all(row["change_initiator"] == "client" for row in blocked)

    assert data["types"]["card_unblocked"] > 0

    for rows in data["by_client"].values():

        first = next((row for row in rows if row["event_type"] == "card_blocked"), None)
        back = next((row for row in rows if row["event_type"] == "card_unblocked"), None)

        if first and back:
            assert back["event_time"] >= first["event_time"]
            return

    pytest.fail("замороженной и размороженной карты не нашлось")


def test_lost_card_is_never_released(baseline):
    """
    Потерянную карту не размораживает ни таймер, ни поддержка.
    Обслуживание возвращает только перевыпуск, а прежняя карта
    остаётся заблокированной навсегда.
    """

    data = _run("card_lost", clients=32)

    blocked = [row for row in data["events"] if row["event_type"] == "card_blocked"]

    assert blocked

    assert all(row["payload"]["reason"] == "lost_or_stolen" for row in blocked)

    lost_cards = {row["payload"]["card_id"] for row in blocked}

    released = {
        row["payload"]["card_id"]
        for row in data["events"]
        if row["event_type"] == "card_unblocked"
    }

    assert not (lost_cards & released), "утраченную карту вернули в строй"

    # Перевыпущенная карта это НОВАЯ карта, а не прежняя. Сама
    # она потом тоже может потеряться, поэтому сравнивать надо с
    # заменённой картой, а не со всем списком утраченных.
    reissues = [row for row in data["events"] if row["event_type"] == "card_reissued"]

    assert reissues

    for row in reissues:
        assert row["payload"]["card_id"] != row["correlation_id"]
        assert row["payload"]["reason"] == "lost_or_stolen"
