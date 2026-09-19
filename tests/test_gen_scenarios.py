from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import pytest

from src.generator import params as params_module
from src.generator.config import HISTORY_START
from src.generator import rng as rng_module
from src.generator.finance import cards as card_rules
from src.generator.finance import deposits as deposit_rules
from src.generator.finance import invariants as invariants_module
from src.generator.finance import loans as loan_rules
from src.generator.finance.entities import (
    ACCOUNT_CARD,
    ACCOUNT_CREDIT_CARD,
    ACCOUNT_DEPOSIT,
    Account,
    Contract,
)
from src.generator.finance.ledger import Ledger
from src.generator.life.scenarios import BY_NAME
from src.generator.life.stress import StressEpisode


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


def _run(scenario_name: str | None = None, clients: int = CLIENTS, extra: dict | None = None) -> dict:
    """
    Небольшая популяция под конкретным пресетом.

    Один и тот же пресет не пересчитывается: розыгрыш
    детерминирован, и второй прогон дал бы ровно то же самое.

    extra: точечные правки параметров поверх пресета, когда
    нужного пресета нет. Розыгрыш ключуется идентичностью
    события, поэтому правка дефектов наблюдаемости меняет только
    их и не сдвигает саму историю.
    """

    key = (scenario_name, clients, json.dumps(extra, sort_keys=True) if extra else None)

    if key in _CACHE:
        return _CACHE[key]

    overrides = {section: dict(values) for section, values in BASE.items()}

    if scenario_name is not None:
        for section, values in BY_NAME[scenario_name].overrides.items():
            overrides.setdefault(section, {})
            overrides[section].update(values)

    for section, values in (extra or {}).items():
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


NO_OUTAGE = {"defects": {"outage_days_per_year": {}}}

HEAVY_OUTAGE = {
    "defects": {
        "outage_days_per_year": {"transactions": 90.0},
        "outage_recovers_share": 0.0,
    }
}


def test_lost_row_makes_a_gap_not_a_rewrite():
    """
    Строку, потерянную сбоем источника, витрина не доносит. Но
    деньги по ней двигались: остатки остальных строк обязаны
    остаться прежними, а разрыв цепочки объясняется потерей.

    Раньше остатки пересчитывались по наблюдаемой ленте, и
    потерянное зачисление переписывало всю последующую историю
    счёта так, будто денег не было вовсе.
    """

    clean = _run(None, extra=NO_OUTAGE)
    broken = _run(None, extra=HEAVY_OUTAGE)

    lost = invariants_module.unobserved_rows(broken["truth"])

    money = [row for row in lost if row.get("account_id")]

    assert money, "денежных потерь не случилось: проверять нечего"

    # Выжившая строка несёт ТОТ ЖЕ остаток, что и без потерь.
    before = {
        (row["event_id"], row["event_version"]): row["payload"].get("balance_after")
        for row in clean["events"]
    }

    checked = 0

    for row in broken["events"]:
        key = (row["event_id"], row["event_version"])
        if key not in before:
            continue
        assert row["payload"].get("balance_after") == before[key], key
        checked += 1

    assert checked > len(broken["events"]) // 2

    # Разрыв цепочки объясняется ровно потерянными строками.
    truth_by_client = defaultdict(list)

    for row in broken["truth"]:
        truth_by_client[row["client_id"]].append(row)

    unobserved = {
        client_id: invariants_module.unobserved_rows(rows)
        for client_id, rows in truth_by_client.items()
    }

    assert invariants_module.check_all(broken["by_client"]), "потери обязаны рвать наблюдаемую цепочку"

    problems = invariants_module.check_all(broken["by_client"], unobserved)

    assert not problems, [str(item) for item in problems[:5]]


def test_copies_keep_business_link_type(baseline):
    """
    Повторная доставка и исправление наследуют вид деловой связи.

    Метки доставки в конверте нет вовсе: исправленный перевод
    остаётся переводом, иначе представление переводов показывало
    бы сумму, которую банк уже исправил.
    """

    from src.generator.config import LINK_TYPES

    assert "correction" not in LINK_TYPES and "duplicate" not in LINK_TYPES

    by_id = defaultdict(list)

    for row in baseline["events"]:
        by_id[row["event_id"]].append(row)

    copies = 0

    for rows in by_id.values():

        if len(rows) < 2:
            continue

        copies += 1

        assert len({row["link_type"] for row in rows}) == 1, rows[0]["event_id"]

    assert copies, "копий записей в наборе не оказалось"

    assert all(row["link_type"] in (None, *LINK_TYPES) for row in baseline["events"])


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


def _unobserved(data: dict) -> dict:
    """
    Потерянные наблюдением строки по клиентам.

    Без них любая потеря источника выглядела бы разрывом
    арифметики, хотя деньги по такой строке двигались.
    """

    truth_by_client = defaultdict(list)

    for row in data["truth"]:
        truth_by_client[row["client_id"]].append(row)

    return {
        client_id: invariants_module.unobserved_rows(rows)
        for client_id, rows in truth_by_client.items()
    }


def test_money_is_conserved(baseline):

    problems = invariants_module.check_all(baseline["by_client"], _unobserved(baseline))

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


# ------------------------------------------------------------
# ФИНАНСОВАЯ МЕХАНИКА
# ------------------------------------------------------------


def _credit_card_client():
    """
    Минимальный клиент с одной картой рассрочки.
    """

    from src.generator.observe.envelope import EventFactory
    from src.generator.simulate import ClientState
    from src.generator.life.persona import draw_persona

    _reset_defaults()

    persona = draw_persona(1)

    state = ClientState(
        persona=persona,
        factory=EventFactory(persona.client_id, persona.is_test_account),
        ledger=Ledger(persona.client_id),
    )

    state.ledger.add_account(
        Account(
            account_id="acc1",
            client_id=persona.client_id,
            kind=ACCOUNT_CREDIT_CARD,
            contract_id="k1",
            credit_limit=500_000,
            opened_at=datetime(2025, 1, 1),
        )
    )

    state.card_credits["k1"] = card_rules.open_credit(
        type("C", (), {"contract_id": "k1", "account_id": "acc1"})(),
        {"installment_months": 6},
    )

    return state


def test_refund_reduces_card_debt_and_happens_in_time():
    """
    Возврат по карте рассрочки снимает долг, а не только кладёт
    деньги на счёт.

    Раньше покупка ставила части рассрочки в график, полный
    возврат их не убирал, и выписка требовала денег за то, чего
    клиент не покупал. Вдобавок возвраты рождались после всей
    симуляции и не успевали ни на одну выписку.
    """

    credit = card_rules.open_credit(
        type("C", (), {"contract_id": "k1", "account_id": "acc1"})(),
        {"installment_months": 6},
    )

    card_rules.add_purchase(credit, 12_000, month_index=100, cause_event_id="p1")

    assert sum(value for _, value, _ in credit.parts) == 12_000

    card_rules.reverse_purchase(credit, 12_000, "p1")

    assert credit.parts == []
    assert credit.cash_principal == 0

    # Частичный возврат снимает ровно свою часть, начиная с
    # самой поздней: вернувшая деньги покупка не тянет график.
    card_rules.add_purchase(credit, 12_000, month_index=100, cause_event_id="p2")
    card_rules.reverse_purchase(credit, 4_000, "p2")

    assert sum(value for _, value, _ in credit.parts) == 8_000
    assert max(due for due, _, _ in credit.parts) < 100 + 6

    # --- проводка возврата действительно снимает долг ---

    from src.generator.config import INITIATOR_SYSTEM
    from src.generator.engine import _emit_money

    state = _credit_card_client()

    body = {
        "channel": "pos",
        "merchant_country": "KZ",
        "mcc": "5411",
        "outlet_id": "o1",
    }

    purchase = _emit_money(
        state, datetime(2025, 2, 10, 12, 0), "purchase", "acc1", 12_000, "debit",
        "merchant:o1", dict(body), INITIATOR_SYSTEM,
    )

    debt = state.card_credits["k1"]

    assert sum(value for _, value, _ in debt.parts) == 12_000

    _emit_money(
        state, datetime(2025, 2, 12, 12, 0), "refund", "acc1", 12_000, "credit",
        "merchant:o1", dict(body, cause_event_id=purchase.event_id, reason="refund"),
        INITIATOR_SYSTEM, correlation_id=purchase.event_id, link_type="refund",
    )

    assert debt.parts == []
    assert state.ledger.balance("acc1") == 0

    _reset_defaults()


def test_refund_never_touches_cash_debt():
    """
    Возврат гасит долг только своей покупки. Наличный долг это
    другие деньги.

    Часть рассрочки не помнила покупку, и возврат снимал долг без
    разбора: у уже выплаченной покупки частей не оставалось, и её
    возврат добирался до снятых наличных. Клиент получал деньги
    от магазина, а банк списывал ему посторонний долг.
    """

    credit = card_rules.open_credit(
        type("C", (), {"contract_id": "k1", "account_id": "acc1"})(),
        {"installment_months": 6},
    )

    card_rules.add_cash(credit, 5_000)
    card_rules.add_purchase(credit, 12_000, month_index=100, cause_event_id="p1")

    # Покупка выплачена выписками целиком: частей не осталось.
    assert card_rules.apply_card_payment(credit, 12_000, month_index=106) == 12_000
    assert credit.parts == []
    assert credit.cash_principal == 5_000

    # Возврат выплаченной покупки долга не снимает: его нет.
    assert card_rules.reverse_purchase(credit, 12_000, "p1") == 0
    assert credit.cash_principal == 5_000
    assert credit.parts == []

    # Возврат без причины долга не снимает тоже: непонятно, чьего.
    card_rules.add_purchase(credit, 6_000, month_index=100, cause_event_id="p2")

    assert card_rules.reverse_purchase(credit, 6_000, None) == 0
    assert credit.outstanding == 11_000


def test_refund_releases_only_its_own_purchase():
    """
    Две покупки в рассрочке, возврат одной: снимаются части только
    её, вторая покупка остаётся в графике целиком.
    """

    credit = card_rules.open_credit(
        type("C", (), {"contract_id": "k1", "account_id": "acc1"})(),
        {"installment_months": 6},
    )

    card_rules.add_purchase(credit, 12_000, month_index=100, cause_event_id="p1")
    card_rules.add_purchase(credit, 6_000, month_index=100, cause_event_id="p2")

    assert card_rules.reverse_purchase(credit, 12_000, "p1") == 12_000

    assert all(cause == "p2" for _, _, cause in credit.parts)
    assert credit.outstanding == 6_000

    # Возврат больше остатка своей покупки снимает только остаток.
    assert card_rules.reverse_purchase(credit, 9_000, "p2") == 6_000
    assert credit.parts == []

    # --- через проводку: причина берётся из payload возврата ---

    from src.generator.config import INITIATOR_SYSTEM
    from src.generator.engine import _emit_money

    state = _credit_card_client()

    body = {"channel": "pos", "merchant_country": "KZ", "mcc": "5411", "outlet_id": "o1"}

    first = _emit_money(
        state, datetime(2025, 2, 10, 12, 0), "purchase", "acc1", 12_000, "debit",
        "merchant:o1", dict(body), INITIATOR_SYSTEM,
    )
    _emit_money(
        state, datetime(2025, 2, 11, 12, 0), "purchase", "acc1", 6_000, "debit",
        "merchant:o1", dict(body), INITIATOR_SYSTEM,
    )
    _emit_money(
        state, datetime(2025, 2, 12, 12, 0), "cash_withdrawal", "acc1", 5_000, "debit",
        "external:atm", dict(body, channel="atm"), INITIATOR_SYSTEM,
    )

    debt = state.card_credits["k1"]

    assert debt.outstanding == 23_000

    _emit_money(
        state, datetime(2025, 2, 13, 12, 0), "refund", "acc1", 12_000, "credit",
        "merchant:o1", dict(body, cause_event_id=first.event_id, reason="refund"),
        INITIATOR_SYSTEM, correlation_id=first.event_id, link_type="refund",
    )

    # Снялись части первой покупки; вторая и наличные на месте.
    assert debt.cash_principal == 5_000
    assert sum(value for _, value, _ in debt.parts) == 6_000

    _reset_defaults()


def test_refund_happens_inside_the_life(baseline):
    """
    Возврат исполняется в своё время, а не после симуляции.

    Раньше возвраты рождались в самом конце сборки клиента: их
    деньги не влияли ни на одно решение, потому что решения уже
    были приняты. Теперь возврат стоит в очереди своего дня, и
    после него у клиента идёт обычная жизнь.
    """

    moments = {row["event_id"]: row["event_time"] for row in baseline["events"]}

    refunds = [
        row
        for row in baseline["events"]
        if row["event_type"] in ("refund", "reversal")
    ]

    assert refunds

    followed = 0

    for row in refunds:

        cause = row["payload"].get("cause_event_id")

        assert cause is not None

        if cause in moments:
            assert row["event_time"] > moments[cause]

        later = [
            item
            for item in baseline["by_client"][row["client_id"]]
            if item["event_time"] > row["event_time"]
        ]

        if later:
            followed += 1

    # Возврат это событие внутри жизни клиента, а не хвост после
    # неё: почти у каждого есть продолжение.
    assert followed >= len(refunds) * 0.9, (followed, len(refunds))


def test_partial_payments_leave_no_residue():
    """
    Кредит, выплаченный частями, закрывается полностью.

    Основная часть платежа считалась долей от суммы взноса и
    округлялась вниз, поэтому после всех платежей на договоре
    оставался хвост в несколько тенге, и кредит формально не
    гасился.
    """

    loan = loan_rules.open_loan(
        contract_id="k1",
        principal=100_000,
        annual_rate=0.29,
        months=12,
        disbursed_at=datetime(2025, 1, 15),
        autopay=False,
    )

    for item in loan.schedule:

        loan_rules.register_due(loan, item, "ev1")

        half = item.amount // 2

        loan_rules.apply_payment(loan, item, half, item.due_date)
        loan_rules.apply_payment(loan, item, item.amount - half, item.due_date)

        assert item.status == "paid"

    assert loan.principal_outstanding == 0
    assert sum(item.principal_paid for item in loan.schedule) == 100_000


def _deposit_interest(opened: datetime) -> int:
    """
    Проценты за январь по вкладу, открытому в этот момент.
    """

    ledger = Ledger("c1")

    ledger.add_account(
        Account(account_id="dep1", client_id="c1", kind=ACCOUNT_DEPOSIT, opened_at=opened)
    )

    ledger.post(opened, "ev1", ledger.cash_id, "dep1", 1_000_000)

    state = deposit_rules.open_deposit(
        contract_id="k1",
        account_id="dep1",
        amount=1_000_000,
        rate=0.14,
        opened_at=opened,
        term_months=12,
        topup=True,
        withdrawal=True,
        capitalisation="monthly",
    )

    return deposit_rules.monthly_interest(state, ledger, datetime(2025, 1, 15))


def test_deposit_interest_depends_on_days_held():
    """
    Вклад, открытый тридцать первого января, получает процент за
    один день, а не за весь месяц.

    Проценты считались от остатка на конец месяца, поэтому вклад
    последнего дня зарабатывал столько же, сколько пролежавший
    весь январь.
    """

    whole_month = _deposit_interest(datetime(2025, 1, 1, 10, 0))
    last_day = _deposit_interest(datetime(2025, 1, 31, 10, 0))

    assert whole_month > 0
    assert last_day > 0

    share = last_day / whole_month

    assert 1 / 31 - 0.01 < share < 1 / 31 + 0.01, share


def test_daily_capitalisation_compounds_within_month():
    """
    При ежедневной капитализации процент, начисленный до
    пополнения, сам зарабатывает процент после него.

    База следующего отрезка месяца складывалась из остатка и
    движения, а уже начисленный процент в неё не входил: ошибка
    маленькая, но систематическая, и всегда в пользу банка.
    """

    opened = datetime(2025, 1, 1, 10, 0)
    topup = datetime(2025, 1, 16, 10, 0)

    ledger = Ledger("c1")

    ledger.add_account(
        Account(account_id="dep1", client_id="c1", kind=ACCOUNT_DEPOSIT, opened_at=opened)
    )

    ledger.post(opened, "ev1", ledger.cash_id, "dep1", 1_000_000)
    ledger.post(topup, "ev2", ledger.cash_id, "dep1", 500_000)

    state = deposit_rules.open_deposit(
        contract_id="k1",
        account_id="dep1",
        amount=1_000_000,
        rate=0.14,
        opened_at=opened,
        term_months=12,
        topup=True,
        withdrawal=True,
        capitalisation="daily",
    )

    daily = 0.14 / 365.0

    first = 1_000_000 * ((1.0 + daily) ** 15 - 1.0)
    second = (1_000_000 + first + 500_000) * ((1.0 + daily) ** 16 - 1.0)

    interest = deposit_rules.monthly_interest(state, ledger, datetime(2025, 1, 20))

    assert abs(interest - (first + second)) <= 1, (interest, first + second)

    # Прежний расчёт не включал процент первого отрезка в базу
    # второго: разница — процент с процента, около 35 тенге.
    old_way = 1_000_000 * ((1.0 + daily) ** 15 - 1.0) + 1_500_000 * ((1.0 + daily) ** 16 - 1.0)

    assert interest >= round(old_way) + 30, (interest, old_way)


def _deposit_client():
    """
    Минимальный клиент с картой и одним вкладом на миллион.
    """

    from src.generator.observe.envelope import EventFactory
    from src.generator.simulate import ClientState
    from src.generator.life.persona import draw_persona

    _reset_defaults()

    persona = draw_persona(1)

    opened = datetime(2025, 1, 1, 10, 0)

    state = ClientState(
        persona=persona,
        factory=EventFactory(persona.client_id, persona.is_test_account),
        ledger=Ledger(persona.client_id),
    )

    state.ledger.add_account(
        Account(
            account_id="card1", client_id=persona.client_id, kind=ACCOUNT_CARD,
            contract_id="c1", opened_at=datetime(2024, 6, 1),
        )
    )
    state.ledger.add_account(
        Account(
            account_id="dep1", client_id=persona.client_id, kind=ACCOUNT_DEPOSIT,
            contract_id="k1", opened_at=opened,
        )
    )

    state.ledger.post(opened, "ev_open", state.ledger.cash_id, "dep1", 1_000_000)

    state.deposits["k1"] = deposit_rules.open_deposit(
        contract_id="k1",
        account_id="dep1",
        amount=1_000_000,
        rate=0.14,
        opened_at=opened,
        term_months=12,
        topup=True,
        withdrawal=True,
        capitalisation="daily",
    )

    state.contracts["k1"] = Contract(
        contract_id="k1",
        client_id=persona.client_id,
        product_id="p1",
        product_code="SYNTH_DEPOSIT",
        product_family="deposit",
        product_version=1,
        tariff_version=1,
        opened_at=opened,
        account_id="dep1",
        amount_or_limit=1_000_000,
        term=12,
        rate=0.14,
    )

    return state


def test_early_deposit_closure_forfeits_interest():
    """
    Досрочно закрытый вклад теряет начисленные проценты, а
    закрытие называется своим именем.

    Флаг early лишь запрещал пролонгацию: клиент получал весь
    остаток вместе с процентами, штраф не списывался, а закрытие
    записывалось как matured.
    """

    from src.generator import engine  # noqa: F401  (engine раньше engine_month)
    from src.generator.engine_month import _close_deposit, _credit_deposit_interest

    state = _deposit_client()

    deposit = state.deposits["k1"]

    for month, last_day in (
        (datetime(2025, 1, 1), datetime(2025, 1, 31, 23, 50)),
        (datetime(2025, 2, 1), datetime(2025, 2, 28, 23, 50)),
    ):
        assert _credit_deposit_interest(state, last_day, deposit, month) > 0

    accrued = deposit.accrued

    assert accrued > 0
    assert state.ledger.balance("dep1") == 1_000_000 + accrued

    _close_deposit(state, datetime(2025, 3, 10, 15, 0), deposit, early=True)

    fees = [event for event in state.events if event.event_type == "fee_charge"]

    assert [event.payload["reason"] for event in fees] == ["early_closure"]
    assert fees[0].payload["amount"] == accrued

    payouts = [
        event
        for event in state.events
        if event.event_type == "deposit_withdrawal" and event.payload["direction"] == "debit"
    ]

    assert [event.payload["reason"] for event in payouts] == ["early_closure"]
    assert payouts[0].payload["amount"] == 1_000_000

    closed = [event for event in state.events if event.event_type == "product_closed"]

    assert [event.payload["reason"] for event in closed] == ["early_closure"]

    assert deposit.closed
    assert state.ledger.balance("dep1") == 0
    assert state.ledger.balance("card1") == 1_000_000

    # Деньги сходятся: у клиента остался ровно миллион, проценты
    # ушли банку.
    assert sum(item.balance for item in state.ledger.accounts.values() if item.visible) == 1_000_000

    _reset_defaults()


def test_own_transfer_moves_money_once():
    """
    Перевод между своими счетами — две записи, одна проводка.

    Проводились обе ноги: пополнение вклада удваивало его остаток
    в ledger и снимало с карты вдвое больше, а проценты считались
    с удвоенного остатка. RAW этого не показывал, потому что
    balance_after там пересчитывается по ленте, и дефект жил в
    симуляции незамеченным.
    """

    from src.generator import engine  # noqa: F401  (engine раньше engine_app)
    from src.generator.engine_app import _own_transfer

    state = _deposit_client()

    _own_transfer(
        state, datetime(2025, 3, 10, 15, 0), "deposit_withdrawal", "dep1", "card1", 400_000, "k1", "matured",
    )

    assert state.ledger.balance("dep1") == 600_000
    assert state.ledger.balance("card1") == 400_000

    legs = [event for event in state.events if event.event_type == "deposit_withdrawal"]

    assert [(event.payload["direction"], event.payload["balance_after"]) for event in legs] == [
        ("debit", 600_000),
        ("credit", 400_000),
    ]

    # Проводка одна, но видна обоим счетам.
    window = (datetime(2025, 3, 1), datetime(2025, 4, 1))

    assert state.ledger.signed_moves("dep1", *window) == [(datetime(2025, 3, 10, 15, 0), -400_000)]
    assert state.ledger.signed_moves("card1", *window) == [(datetime(2025, 3, 10, 15, 0), 400_000)]

    _reset_defaults()


def _acting(rows: list) -> list:
    """
    Действующие строки: у события берётся наибольшая версия,
    повторные доставки схлопываются.
    """

    best: dict = {}

    for row in rows:
        key = row["event_id"]
        if key not in best or row["event_version"] > best[key]["event_version"]:
            best[key] = row

    return list(best.values())


def _deposit_openings(rows: list, before_window: bool = False) -> dict:
    """
    product_opened вкладов: contract_id → сумма договора.
    """

    return {
        row["payload"]["contract_id"]: row["payload"].get("amount_or_limit")
        for row in rows
        if row["event_type"] == "product_opened"
        and row["payload"].get("product_family") in ("deposit", "deposit_certificate")
        and ((row["event_time"] < HISTORY_START) if before_window else (row["event_time"] >= HISTORY_START))
    }


def test_every_opened_deposit_is_funded(baseline):
    """
    Открытый в окне вклад либо получает деньги, либо аннулируется
    сразу.

    Сумма вклада считалась от всех денег клиента, включая наличные
    и другой банк, а финансирование требовало одного счёта с этой
    суммой. Большинство договоров оставались открытыми без остатка,
    процентов и срока: на split3 — все шестнадцать.
    """

    rows = _acting(baseline["events"])

    opened = _deposit_openings(rows)

    assert opened, "в выборке нет вкладов, открытых в окне"

    funded = {
        row["payload"].get("contract_id"): row["payload"]["amount"]
        for row in rows
        if row["event_type"] == "deposit_topup"
        and row["payload"].get("reason") == "initial_deposit"
        and row["payload"].get("direction") == "credit"
    }

    cancelled = {
        row["payload"].get("contract_id")
        for row in rows
        if row["event_type"] == "product_closed" and row["payload"].get("reason") == "not_funded"
    }

    for contract_id, amount in opened.items():
        assert contract_id in funded or contract_id in cancelled, contract_id
        if contract_id in funded:
            assert funded[contract_id] == amount, contract_id

    # Аннулирование — исключение, а не правило.
    assert len(funded) >= len(cancelled), (len(funded), len(cancelled))

    # Профинансированный вклад зарабатывает проценты уже в первый
    # месяц.
    interest = {row["payload"].get("contract_id") for row in rows if row["event_type"] == "interest_credit"}

    early_enough = {
        contract_id
        for contract_id in funded
        if any(
            row["payload"].get("contract_id") == contract_id and row["event_time"] < datetime(2026, 8, 1)
            for row in rows
            if row["event_type"] == "product_opened"
        )
    }

    assert early_enough <= interest, early_enough - interest


def test_prehistory_deposits_live(baseline):
    """
    Вклад, открытый до окна, зарабатывает проценты и имеет срок.

    У вкладов предыстории не было состояния: деньги на счёте
    лежали, а проценты не начислялись, срок не наступал, досрочное
    закрытие их не касалось.
    """

    rows = _acting(baseline["events"])

    opened = _deposit_openings(rows, before_window=True)

    closed_before = {
        row["payload"].get("contract_id")
        for row in rows
        if row["event_type"] == "product_closed" and row["event_time"] < HISTORY_START
    }

    alive = set(opened) - closed_before

    if not alive:
        pytest.skip("в выборке нет вкладов предыстории")

    interest = {row["payload"].get("contract_id") for row in rows if row["event_type"] == "interest_credit"}

    assert alive <= interest, alive - interest


def test_forced_early_closure_forfeits_interest_live():
    """
    Досрочное закрытие в живой симуляции: вклад закрывается своим
    именем, штраф не больше начисленных процентов, а выплата равна
    остатку после штрафа.
    """

    result = _run(None, CLIENTS, {"products": {"deposit_early_close_share_per_year": 1200.0}})

    rows = sorted(_acting(result["events"]), key=lambda row: (row["event_time"], row["event_id"]))

    funded = {
        row["payload"].get("contract_id")
        for row in rows
        if row["event_type"] == "deposit_topup"
        and row["payload"].get("reason") == "initial_deposit"
        and row["payload"].get("direction") == "credit"
        and row["event_time"] < datetime(2026, 8, 1)
    }

    assert funded, "в выборке нет профинансированных вкладов"

    closed = {
        row["payload"].get("contract_id")
        for row in rows
        if row["event_type"] == "product_closed" and row["payload"].get("reason") == "early_closure"
    }

    # Каждый профинансированный вклад закрыт досрочно в первый же
    # конец месяца.
    assert funded <= closed, funded - closed

    interest: dict = defaultdict(int)

    for row in rows:
        if row["event_type"] == "interest_credit":
            interest[row["payload"].get("contract_id")] += row["payload"]["amount"]

    for contract_id in closed:

        fees = [
            row for row in rows
            if row["event_type"] == "fee_charge"
            and row["payload"].get("reason") == "early_closure"
            and row["payload"].get("contract_id") == contract_id
        ]

        payouts = [
            row for row in rows
            if row["event_type"] == "deposit_withdrawal"
            and row["payload"].get("reason") == "early_closure"
            and row["payload"].get("direction") == "debit"
            and row["payload"].get("contract_id") == contract_id
        ]

        assert len(fees) <= 1 and len(payouts) == 1, contract_id

        if fees:
            fee = fees[0]["payload"]
            assert 0 < fee["amount"] <= interest[contract_id], contract_id
            # Выплачивается ровно то, что осталось после штрафа.
            assert payouts[0]["payload"]["amount"] == fee["balance_after"], contract_id
        else:
            assert interest[contract_id] == 0, contract_id

    # Деньги сходятся и здесь: потерянные наблюдением строки
    # объясняются truth.
    truth_by_client: dict = defaultdict(list)

    for row in result["truth"]:
        truth_by_client[row["client_id"]].append(row)

    for client_id, client_rows in result["by_client"].items():
        lost = invariants_module.unobserved_rows(truth_by_client.get(client_id, []))
        assert invariants_module.check_client(client_rows, lost) == [], client_id

    _reset_defaults()


def test_deposits_are_not_payment_accounts():
    """
    Вклад не платёжный счёт: покупки, счета и взносы идут с карт и
    текущих счетов, а деньги со вклада выводятся отдельной операцией
    с проверкой условий продукта.

    Ledger предлагал вклад как запасной источник денег, и клиенты
    платили со срочных вкладов, запрещённых к снятию: на check это
    было 287 списаний на 5,6 млн тенге.
    """

    state = _deposit_client()

    ts = datetime(2025, 3, 10, 15, 0)

    # Миллион на вкладе платёжной ёмкости не даёт.
    assert state.ledger.payment_sources(ts, 5_000) == []
    assert state.ledger.payment_capacity(ts) == 0

    state.ledger.post(ts, "ev_cash", state.ledger.cash_id, "card1", 20_000)

    assert [item.account_id for item in state.ledger.payment_sources(ts, 5_000)] == ["card1"]
    assert state.ledger.payment_capacity(ts) == 20_000

    # Снятие со вклада — отдельная операция, и только по условиям.
    from src.generator import engine  # noqa: F401  (engine раньше остальных модулей движка)
    from src.generator.engine_credit import _withdraw_from_deposit

    deposit = state.deposits["k1"]
    card = state.ledger.get("card1")

    deposit.withdrawal_allowed = False

    assert _withdraw_from_deposit(state, ts, card, 100_000) is False
    assert state.ledger.balance("dep1") == 1_000_000
    assert not [event for event in state.events if event.event_type == "deposit_withdrawal"]

    deposit.withdrawal_allowed = True

    assert _withdraw_from_deposit(state, ts, card, 100_000) is True
    assert state.ledger.balance("dep1") == 900_000
    assert state.ledger.balance("card1") == 120_000
    assert deposit.principal == 900_000

    legs = [event for event in state.events if event.event_type == "deposit_withdrawal"]

    assert [event.payload["reason"] for event in legs] == ["withdrawal_before_payment"] * 2

    _reset_defaults()


def test_no_payments_leave_deposit_accounts(baseline):
    """
    В живой ленте со счёта вклада уходят только снятия и комиссии:
    ни покупок, ни счетов, ни переводов, ни наличных.
    """

    rows = _acting(baseline["events"])

    deposit_accounts = {
        row["payload"].get("account_id")
        for row in rows
        if row["event_type"] == "product_opened"
        and row["payload"].get("product_family") in ("deposit", "deposit_certificate")
    }
    deposit_accounts.discard(None)

    assert deposit_accounts, "в выборке нет вкладов"

    debits = Counter(
        row["event_type"]
        for row in rows
        if row["payload"].get("account_id") in deposit_accounts
        and row["payload"].get("direction") == "debit"
        and row["payload"].get("status") == "approved"
        and row["event_type"] != "balance_snapshot"
    )

    assert set(debits) <= {"deposit_withdrawal", "fee_charge"}, dict(debits)


def test_rollover_resets_accrued_interest():
    """
    Пролонгация открывает новый срок: проценты прошлого срока
    заработаны и капитализированы, досрочное закрытие нового срока
    их не отнимает.

    accrued копился через все сроки, и штраф при досрочном закрытии
    забирал проценты уже завершённых сроков: 511 624 тенге вместо
    130 819 текущего срока.
    """

    from src.generator import engine  # noqa: F401  (engine раньше engine_month)
    from src.generator.engine_month import _close_deposit, _credit_deposit_interest

    state = _deposit_client()

    # Продукт из каталога и гарантированная пролонгация.
    state.contracts["k1"].product_code = "DEPOSIT_HOOM"
    params_module.activate(
        params_module.DEFAULT.with_overrides({"products": {"deposit_rollover_share": 1.0}})
    )

    deposit = state.deposits["k1"]

    for month, last_day in (
        (datetime(2025, 1, 1), datetime(2025, 1, 31, 23, 50)),
        (datetime(2025, 2, 1), datetime(2025, 2, 28, 23, 50)),
    ):
        _credit_deposit_interest(state, last_day, deposit, month)

    earned = deposit.accrued

    assert earned > 0

    _close_deposit(state, datetime(2025, 3, 31, 23, 50), deposit)

    lifecycle = [
        event.event_type for event in state.events if event.event_type in ("product_renewed", "product_closed")
    ]

    assert lifecycle == ["product_renewed"]
    assert not deposit.closed
    assert deposit.accrued == 0
    assert deposit.principal == state.ledger.balance("dep1") == 1_000_000 + earned

    fresh = _credit_deposit_interest(state, datetime(2025, 4, 30, 23, 50), deposit, datetime(2025, 4, 1))

    assert fresh > 0

    _close_deposit(state, datetime(2025, 5, 10, 23, 52), deposit, early=True)

    fees = [event for event in state.events if event.event_type == "fee_charge"]

    # Штраф — только проценты нового срока.
    assert [event.payload["amount"] for event in fees] == [fresh]
    assert state.ledger.balance("card1") == 1_000_000 + earned
    assert deposit.closed

    _reset_defaults()


def test_unresolved_stress_keeps_a_tail():
    """
    Неразрешённый эпизод не заканчивается вместе со своим окном.

    Исход unresolved означает, что причина осталась. Раньше
    уровень стресса после `end` падал в ноль одинаково у всех
    исходов, и «не разрешилось» ничем не отличалось от
    «доход восстановился».
    """

    tail_share = params_module.active().stress.unresolved_tail_share

    def episode(resolution: str) -> StressEpisode:
        start = datetime(2025, 3, 1)
        return StressEpisode(
            trigger="job_loss",
            start=start,
            peak_start=start + timedelta(days=10),
            peak_end=start + timedelta(days=40),
            end=start + timedelta(days=60),
            intensity=0.8,
            resolution=resolution,
            resolved_at=None if resolution == "unresolved" else start + timedelta(days=60),
        )

    after = datetime(2025, 3, 1) + timedelta(days=120)

    assert episode("income_restored").level(after) == 0.0

    left = episode("unresolved").level(after)

    assert left == pytest.approx(0.8 * tail_share)
    assert 0.0 < left < 0.8
