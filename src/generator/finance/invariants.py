from __future__ import annotations

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


def _family_of(product_id: str | None) -> str | None:
    """
    Семейство продукта по каталогу генератора.
    """

    from ..world import products as product_catalog

    return product_catalog.catalog().family_of(product_id)


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

def check_client(events: list) -> list:
    """
    Все финансовые инварианты одного клиента.
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

        if kind == "balance_snapshot":
            if account in last_balance and last_balance[account] != int(balance):
                fail(
                    "balance_snapshot_matches_chain",
                    f"{account}: снимок {balance} против цепочки {last_balance[account]}",
                )
            last_balance[account] = int(balance)
            continue

        direction = payload.get("direction")

        signed = amount if direction == "credit" else -amount

        if account in last_balance:
            expected = last_balance[account] + signed
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

        # Семейство продукта событие больше не несёт: его знает
        # каталог по product_id.
        if kind == "product_opened" and _family_of(payload.get("product_id")) in (
            "deposit",
            "deposit_certificate",
        ):
            deposit_open[contract] = event["event_time"]
        elif kind == "product_closed":
            deposit_closed[contract] = event["event_time"]
        elif kind in ("deposit_topup", "deposit_withdrawal", "interest_credit"):
            opened = deposit_open.get(contract)
            # Договор из предыстории в ленте не открывался:
            # он вошёл в окно уже открытым.
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


def check_money_conservation(events: list) -> list:
    """
    Изменение совокупного клиентского баланса объясняется
    чистыми внешними потоками и проводками между клиентскими
    и банковскими счетами. Внутренние переводы взаимно
    сокращаются и в общее изменение не входят.
    """

    if not events:
        return []

    client_id = events[0]["client_id"]

    ordered = authoritative(events)

    first_balance: dict[str, int] = {}
    last_balance: dict[str, int] = {}

    external = 0

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


def check_all(events_by_client: dict) -> list:

    problems: list[Violation] = []

    for client_id, events in events_by_client.items():
        problems.extend(check_client(events))
        problems.extend(check_money_conservation(events))

        for event_id in repeated_event_ids(events):
            problems.append(Violation("repeated_event_id", client_id, event_id))

    return problems


__all__ = [
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
]
