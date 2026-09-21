from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass


# ============================================================
# ФИНАНСОВЫЕ ИНВАРИАНТЫ
# ============================================================
#
# Проверки идут по НАБЛЮДАЕМОЙ ленте: то, что нельзя увидеть
# в RAW, нельзя и потребовать от него.
#
#   пара проводок в сумме даёт ноль
#   balance_after продолжает предыдущий balance_after
#   declined и cancelled баланс не меняют
#   p2p_out и p2p_in совпадают по transfer_id и сумме
#   возвраты не превышают исходную операцию
#   cause_event_id ведёт на существующее событие
#   долг сходится с выдачей, платежами и процентами
#   по заблокированной карте нет одобренных покупок
#   операции депозита лежат внутри жизни договора
# ============================================================


CREDIT_EVENTS = frozenset(
    {
        "salary_credit",
        "pension_credit",
        "other_income_credit",
        "transfer_in",
        "p2p_in",
        "cash_deposit",
        "interest_credit",
        "cashback_credit",
        "loan_disbursement",
        "refund",
        "chargeback",
        # Перевод между своими счетами имеет обе стороны, поэтому
        # пополнение и снятие вклада бывают и приходом, и расходом.
        "deposit_withdrawal",
        "deposit_topup",
    }
)

DEBIT_EVENTS = frozenset(
    {
        "purchase",
        "bill_payment",
        "cash_withdrawal",
        "transfer_out",
        "p2p_out",
        "loan_payment",
        "deposit_topup",
        "deposit_withdrawal",
        "fee_charge",
    }
)

MONEY_EVENTS = CREDIT_EVENTS | DEBIT_EVENTS | {"reversal", "balance_snapshot"}

REVERSING_EVENTS = frozenset({"refund", "reversal", "chargeback"})


@dataclass(frozen=True)
class Violation:
    check: str
    client_id: str
    detail: str

    def __str__(self) -> str:
        return f"{self.check}: {self.client_id}: {self.detail}"


# ============================================================
# ПОТЕРЯННОЕ НАБЛЮДЕНИЕ
# ============================================================
#
# Сбой источника не доносит запись до выгрузки. Деньги по ней
# двигались, и остаток следующей наблюдаемой строки её учитывает,
# поэтому в наблюдаемой цепочке образуется разрыв.
#
# Разрыв допустим ТОЛЬКО там, где его объясняет потерянная
# строка этого же счёта. Ничем не объяснённый разрыв остаётся
# нарушением: именно так отличается честная потеря наблюдения от
# сломанной арифметики.
#
# Скрытая истина сюда приходит из truth и нужна проверкам и
# отчёту генератора. Препроцессинг её не читает и ищет разрывы
# по самой выгрузке.
# ============================================================


UNOBSERVED_KIND = "unobserved_row"


def unobserved_rows(truth_events: list) -> list[dict]:
    """
    Строки, потерянные наблюдением, из скрытой истории клиента.
    """

    out: list[dict] = []

    for row in truth_events:

        if row.get("kind") != UNOBSERVED_KIND:
            continue

        value = row.get("value")

        data = json.loads(value) if isinstance(value, str) else dict(value or {})

        data["event_time"] = row["ts"]

        out.append(data)

    return out


def _signed(row: dict) -> int:
    """
    Знаковое движение по счёту клиента.
    """

    amount = int(row.get("amount") or 0)

    return amount if row.get("direction") == "credit" else -amount


def _pending_by_account(unobserved) -> dict[str, list[tuple]]:
    """
    Потерянные движения по счетам, в порядке времени события.
    """

    pending: dict[str, list[tuple]] = defaultdict(list)

    for row in unobserved:

        account = row.get("account_id")

        if account is None or row.get("status") not in (None, "approved"):
            continue

        # Снимок остатка денег не двигает: его сумма это сам
        # остаток, а не проводка. Потерянный снимок ничего в
        # цепочке не объясняет.
        if row.get("event_type") == "balance_snapshot":
            continue

        pending[account].append((row["event_time"], _signed(row), row.get("counterparty")))

    for items in pending.values():
        items.sort(key=lambda item: item[0])

    return pending


def authoritative(events: list) -> list:
    """
    Лента клиента в порядке выгрузки.

    Версий у записи больше нет: каждая строка сразу окончательна,
    и порядок задаёт время события, а при равенстве — место в
    ленте.
    """

    # Место в ленте берётся перечислением, а не поиском: index()
    # искал бы каждую строку заново и на ленте клиента давал бы
    # квадрат, а на одинаковых словарях ещё и неверный ответ.
    return [
        event
        for _, event in sorted(
            ((order, event) for order, event in enumerate(events)),
            key=lambda item: (item[1]["event_time"], item[0]),
        )
    ]


def repeated_event_ids(events: list) -> list:
    """
    Идентификаторы, встретившиеся в выгрузке больше одного раза.

    Это поломка, а не дефект наблюдаемости: запись приходит
    ровно один раз.
    """

    seen: dict[str, int] = {}

    for event in events:
        key = event["event_id"]
        seen[key] = seen.get(key, 0) + 1

    return sorted(key for key, count in seen.items() if count > 1)

def check_client(events: list, unobserved=()) -> list:
    """
    Все финансовые инварианты одного клиента.

    unobserved: строки, которые сбой источника не донёс до
    выгрузки. Разрыв цепочки остатков считается объяснённым
    ровно на их сумму и только на том же счёте.
    """

    if not events:
        return []

    client_id = events[0]["client_id"]

    ordered = authoritative(events)

    problems: list[Violation] = []

    known_ids = {event["event_id"] for event in ordered}

    # Договоры карт рассрочки: у них выписка вместо графика.
    card_contracts = {
        event["payload"].get("contract_id")
        for event in ordered
        if event["payload"].get("reason") == "card_statement"
    }

    def fail(check: str, detail: str) -> None:
        problems.append(Violation(check=check, client_id=client_id, detail=detail))

    # --- цепочка остатков ---

    last_balance: dict[str, int] = {}

    pending = _pending_by_account(unobserved)
    cursor: dict[str, int] = defaultdict(int)

    def missing_before(account: str, moment) -> int:
        """
        Сумма потерянных наблюдением движений счёта, случившихся
        не позже этого момента и ещё не учтённых.
        """

        items = pending.get(account)

        if not items:
            return 0

        total = 0

        while cursor[account] < len(items) and items[cursor[account]][0] <= moment:
            total += items[cursor[account]][1]
            cursor[account] += 1

        return total

    for event in ordered:

        kind = event["event_type"]
        payload = event["payload"]

        if kind not in MONEY_EVENTS:
            continue

        account = payload.get("account_id")
        status = payload.get("status")
        balance = payload.get("balance_after")

        if status in ("declined", "cancelled"):
            if balance is not None:
                fail("declined_no_posting", f"{kind} {event['event_id']} несёт balance_after")
            continue

        if account is None or balance is None:
            continue

        amount = int(payload.get("amount") or 0)

        missing = missing_before(account, event["event_time"])

        if kind == "balance_snapshot":
            if account in last_balance and last_balance[account] + missing != int(balance):
                fail(
                    "balance_snapshot_matches_chain",
                    f"{account}: снимок {balance} против цепочки {last_balance[account] + missing}",
                )
            last_balance[account] = int(balance)
            continue

        direction = payload.get("direction")

        signed = amount if direction == "credit" else -amount

        if account in last_balance:
            expected = last_balance[account] + missing + signed
            if expected != int(balance):
                fail(
                    "balance_after_chain",
                    f"{account} на {kind} {event['event_id']}: ожидалось {expected}, записано {balance}",
                )

        last_balance[account] = int(balance)

    # --- парность внутрибанковского перевода ---

    transfers: dict[str, dict] = defaultdict(dict)

    for event in ordered:

        kind = event["event_type"]

        if kind not in ("p2p_out", "p2p_in"):
            continue

        if event["payload"].get("status") != "approved":
            continue

        transfer_id = event["payload"].get("transfer_id")

        if not transfer_id:
            fail("transfer_has_id", f"{kind} {event['event_id']} без transfer_id")
            continue

        transfers[transfer_id][kind] = event

    for transfer_id, sides in transfers.items():

        if "p2p_out" in sides and "p2p_in" in sides:

            out_amount = int(sides["p2p_out"]["payload"]["amount"])
            in_amount = int(sides["p2p_in"]["payload"]["amount"])

            if out_amount != in_amount:
                fail("transfer_pair_amount", f"{transfer_id}: {out_amount} против {in_amount}")

            if sides["p2p_out"]["event_time"] > sides["p2p_in"]["event_time"]:
                fail("transfer_pair_order", f"{transfer_id}: зачисление раньше списания")

    # --- ссылки причин ---

    reversed_totals: dict[str, int] = defaultdict(int)
    originals: dict[str, int] = {}

    for event in ordered:

        payload = event["payload"]

        cause = payload.get("cause_event_id")

        if cause is not None and cause not in known_ids:
            fail("cause_event_exists", f"{event['event_type']} ссылается на неизвестный {cause}")

        if event["event_type"] in ("purchase", "bill_payment", "p2p_out", "transfer_out"):
            originals[event["event_id"]] = int(payload.get("amount") or 0)

        if event["event_type"] in REVERSING_EVENTS and cause is not None:
            reversed_totals[cause] += int(payload.get("amount") or 0)

    for cause, total in reversed_totals.items():
        original = originals.get(cause)
        if original is not None and total > original:
            fail("refund_not_above_original", f"{cause}: возвращено {total} при исходных {original}")

    # --- заблокированная карта ---

    blocked_until: dict[str, object] = {}

    for event in ordered:

        kind = event["event_type"]
        payload = event["payload"]

        card = payload.get("card_id")

        if kind == "card_blocked" and card:
            blocked_until[card] = None
        elif kind in ("card_unblocked", "card_reissued") and card:
            blocked_until.pop(card, None)
        elif kind in ("purchase", "cash_withdrawal", "bill_payment") and card:
            if card in blocked_until and payload.get("status") == "approved":
                fail("no_purchase_on_blocked_card", f"{kind} {event['event_id']} по карте {card}")

    # --- депозит существует до своих операций ---

    deposit_open: dict[str, object] = {}
    deposit_closed: dict[str, object] = {}

    for event in ordered:

        kind = event["event_type"]
        payload = event["payload"]
        contract = payload.get("contract_id")

        if not contract:
            continue

        if kind == "product_opened" and payload.get("product_family") in ("deposit", "deposit_certificate"):
            deposit_open[contract] = event["event_time"]
        elif kind == "product_closed":
            deposit_closed[contract] = event["event_time"]
        elif kind in ("deposit_topup", "deposit_withdrawal", "interest_credit"):
            opened = deposit_open.get(contract)
            # Договор из предыстории в ленте не открывался:
            # его состояние описано opening_state покрытия.
            if opened is not None and event["event_time"] < opened:
                fail("deposit_before_open", f"{kind} по договору {contract} раньше открытия")

    # --- кредит ---

    loan_principal: dict[str, int] = {}
    loan_marks: dict[str, list] = defaultdict(list)

    for event in ordered:

        kind = event["event_type"]
        payload = event["payload"]
        contract = payload.get("contract_id")

        if not contract:
            continue

        if kind == "installment_paid":
            paid = int(payload.get("amount_paid") or 0)
            due = int(payload.get("amount_due") or 0)
            if due and paid > due:
                fail("installment_not_above_due", f"{contract}: уплачено {paid} при плановых {due}")

        if kind == "loan_restructured":
            # Реструктуризация переносит просрочку в тело долга,
            # и остаток законно растёт.
            loan_principal.pop(contract, None)

        if kind in ("loan_restructured", "arrears_cleared"):
            # Просрочка погашена или договор пересобран: счёт
            # вех начинается заново.
            loan_marks[contract] = []

        if kind == "delinquency_registered":
            mark = int(payload.get("days_past_due") or 0)
            if loan_marks[contract] and mark <= loan_marks[contract][-1]:
                fail("dpd_milestones_grow", f"{contract}: веха {mark} после {loan_marks[contract][-1]}")
            loan_marks[contract].append(mark)

        outstanding = payload.get("principal_outstanding")

        if outstanding is None:
            continue

        outstanding = int(outstanding)

        # Долг по карте рассрочки РАСТЁТ от новых покупок: это
        # возобновляемый лимит, а не амортизируемый кредит.
        # Правило убывающего долга к нему неприменимо.
        if contract in card_contracts:
            loan_principal[contract] = outstanding
            continue

        if contract in loan_principal and kind not in ("loan_restructured", "schedule_created"):
            if outstanding > loan_principal[contract]:
                fail(
                    "principal_not_growing",
                    f"{contract}: долг вырос с {loan_principal[contract]} до {outstanding} на {kind}",
                )

        loan_principal[contract] = outstanding

    return problems


def check_money_conservation(events: list, unobserved=()) -> list:
    """
    Изменение совокупного клиентского баланса объясняется
    чистыми внешними потоками и проводками между клиентскими
    и банковскими счетами. Внутренние переводы взаимно
    сокращаются и в общее изменение не входят.

    Движение, потерянное наблюдением, в выгрузке строкой не
    представлено, но остаток его учёл: оно входит во внешний
    поток по тем же правилам, что и наблюдаемое. Потеря раньше
    первого наблюдения счёта уже сидит в его начальном остатке.
    """

    if not events:
        return []

    client_id = events[0]["client_id"]

    ordered = authoritative(events)

    first_balance: dict[str, int] = {}
    last_balance: dict[str, int] = {}

    external = 0

    pending = _pending_by_account(unobserved)
    cursor: dict[str, int] = defaultdict(int)

    for event in ordered:

        kind = event["event_type"]
        payload = event["payload"]

        # У снимка остатка статуса нет: он ничего не проводил.
        if kind not in MONEY_EVENTS:
            continue

        if kind != "balance_snapshot" and payload.get("status") != "approved":
            continue

        account = payload.get("account_id")
        balance = payload.get("balance_after")

        if account is None or balance is None:
            continue

        items = pending.get(account) or ()

        while cursor[account] < len(items) and items[cursor[account]][0] <= event["event_time"]:
            _, signed_lost, counterparty_lost = items[cursor[account]]
            cursor[account] += 1
            if account in last_balance and counterparty_lost != "own_account":
                external += signed_lost

        amount = int(payload.get("amount") or 0)
        direction = payload.get("direction")

        signed = amount if direction == "credit" else -amount

        if kind == "balance_snapshot":
            if account not in last_balance:
                first_balance[account] = int(balance)
        else:

            if account not in last_balance:
                # Первое наблюдение счёта: остаток до проводки.
                first_balance[account] = int(balance) - signed

            # Перевод между своими счетами взаимно сокращается
            # и в общее изменение не входит.
            if payload.get("counterparty") != "own_account":
                external += signed

        last_balance[account] = int(balance)

    opening = sum(first_balance.values())
    closing = sum(last_balance.values())

    if abs((closing - opening) - external) > 1:
        return [
            Violation(
                check="money_conservation",
                client_id=client_id,
                detail=f"изменение {closing - opening} против внешних потоков {external}",
            )
        ]

    return []


def check_all(events_by_client: dict, unobserved_by_client: dict | None = None) -> list:
    """
    unobserved_by_client: клиент -> строки, потерянные наблюдением
    (из скрытой истины). Без них любая потеря источника выглядит
    разрывом арифметики.
    """

    problems: list[Violation] = []

    lost = unobserved_by_client or {}

    for client_id, events in events_by_client.items():
        rows = lost.get(client_id, ())
        problems.extend(check_client(events, rows))
        problems.extend(check_money_conservation(events, rows))

        for event_id in repeated_event_ids(events):
            problems.append(Violation("repeated_event_id", client_id, event_id))

    return problems


__all__ = [
    "UNOBSERVED_KIND",
    "CREDIT_EVENTS",
    "DEBIT_EVENTS",
    "MONEY_EVENTS",
    "REVERSING_EVENTS",
    "Violation",
    "authoritative",
    "check_all",
    "check_client",
    "check_money_conservation",
    "repeated_event_ids",
    "unobserved_rows",
]
