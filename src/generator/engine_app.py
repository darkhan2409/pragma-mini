from __future__ import annotations

from datetime import datetime, timedelta

from .behaviour import communications as comm_module
from .behaviour import outcomes as outcome_module
from . import config
from . import params as params_module
from .config import (
    SOURCE_AVAILABILITY,
)
from .engine import _HANDLERS, _emit_money, _touch_client
from .finance import deposits as deposit_rules
from .finance import loans as loan_rules
from .finance.entities import CARD_BLOCKED, Offer
from .finance.ledger import COUNTERPART_GOVERNMENT
from .world.dictionaries import MCC_TRANSFER
from .world.relationships import _external
from .life import stress as stress_module
from .rng import COMPONENT_OUTCOME, NS_SESSION, event_rng, stable_hash
from .simulate import ClientState, _transfer_id, in_window
from .world.dictionaries import (
    BANNER_OFFERS,
    BANNER_OFFER_FAMILY,
    BANNER_SLOTS,
    BANNER_SLOT_WEIGHTS,
    ERROR_CODES,
)


# ============================================================
# ПРИЛОЖЕНИЕ И КОММУНИКАЦИИ
# ============================================================
#
# Число экранов зависит от ЦЕЛИ сессии. Защищённое действие
# возможно только после успешного входа.
#
# Денежная операция исполняется только тогда, когда ей есть что
# менять: открытый счёт к оплате, просроченный платёж,
# заблокированная карта или действующий депозит. Успеха без
# последствия не бывает.
# ============================================================


BANNERS_AVAILABLE_FROM = SOURCE_AVAILABILITY["banners"]


def _on_session(sim, state: ClientState, ts: datetime, payload: dict) -> None:

    session = payload["session"]

    stress = stress_module.level_at(state.stress_episodes, ts)

    authorized = False

    # Что уже показано за это посещение приложения.
    shown_offers: set = set()

    for position, step in enumerate(session.steps):

        if step.ts >= config.HISTORY_END:
            break

        rng = event_rng(
            NS_SESSION,
            state.ordinal,
            ts.toordinal(),
            payload["index"] * 100 + position,
            COMPONENT_OUTCOME,
        )

        if step.kind == "screen":

            if not authorized:
                continue

            state.emit(
                state.factory.make(
                    "app_screen",
                    step.ts,
                    {
                        "firebase_screen": step.screen,
                        "domain": step.domain,
                        "product_id": None,
                        "funnel_stage": None,
                        "reject_reason": None,
                        "session_id": session.session_id,
                        "application_id": None,
                    },
                )
            )

            if step.offers:
                _show_banners(state, step.ts, session, rng, shown_offers)

            continue

        operation = step.operation

        feasible = True
        insufficient = False

        target_bill = None
        target_loan = None
        target_card = None
        target_deposit = None
        transfer_amount = 0
        transfer_pair = None
        deposit_amount = 0

        # Момент денежного следствия выбирается ЗАРАНЕЕ: только
        # так можно проверить, что оно поместится в окно, ещё до
        # того как операция объявлена успешной.
        money_ts = step.ts + timedelta(seconds=int(rng.integers(5, 60)))

        if operation in ("login", "biometry_login"):
            feasible = True
        elif not authorized:
            feasible = False

        if operation and operation.startswith("pay_"):
            target_bill = state.open_bills[0] if state.open_bills else None
            if target_bill is None:
                feasible = False
            elif not state.ledger.payment_sources(step.ts, target_bill["amount"]):
                feasible = False
                insufficient = True
            elif not in_window(money_ts):
                # Списание выпадает за конец выгрузки: показывать
                # успех нечем.
                feasible = False

        if operation == "card_unblock":
            target_card = next(
                (item for item in state.cards.values() if item.releasable()), None
            )
            if target_card is None:
                feasible = False

        if operation == "loan_repay":
            target_loan = next(
                (
                    item
                    for item in state.loans.values()
                    if not item.closed and loan_rules.arrears_amount(item) > 0
                ),
                None,
            )
            if target_loan is None:
                feasible = False
            elif not _can_repay(state, step.ts, loan_rules.arrears_amount(target_loan)):
                # Платить нечем: успех операции остался бы без
                # денежного следствия.
                feasible = False
                insufficient = True

        if operation == "deposit_topup":
            target_deposit = next(
                (
                    item
                    for item in state.deposits.values()
                    if not item.closed
                    and item.topup_allowed
                    and deposit_rules.can_topup(item, step.ts)
                ),
                None,
            )
            if target_deposit is None:
                feasible = False
            else:
                # Пополнение это тот же перевод себе: две стороны,
                # вторая секундой позже. Проверяется и она.
                deposit_amount = _topup_amount(state, step.ts, rng)
                sources = [
                    item
                    for item in state.ledger.payment_sources(step.ts, deposit_amount)
                    if item.account_id != target_deposit.account_id
                ]
                if not sources or not in_window(step.ts + timedelta(seconds=1)):
                    feasible = False
                    insufficient = not sources

        # --- перевод ---
        #
        # У успешного перевода обязана быть сумма и настоящее
        # денежное событие. Поэтому сумма выбирается ДО розыгрыша
        # исхода, а невозможность перевода делает его невыполнимым,
        # а не «успешным, но без денег».
        if operation in TRANSFER_OPERATIONS:

            transfer_amount = _transfer_amount(state, step.ts, rng)

            if transfer_amount <= 0:
                feasible = False
            elif not state.ledger.payment_sources(step.ts, transfer_amount):
                feasible = False
                insufficient = True

            # Зачисление второй ноги датируется секундой позже. На
            # самом краю окна её уже не записать, и операция не
            # должна показывать успех: денег за ним не будет.
            if not in_window(step.ts + timedelta(seconds=1)):
                feasible = False

            if operation == "transfer_own" and feasible:

                transfer_pair = _own_pair(state, step.ts, transfer_amount)

                # Своих счетов может не быть двух, а на том, что
                # есть, может не хватать денег. Раньше пара
                # бралась первыми двумя видимыми счетами подряд,
                # и перевод уходил со счёта без нужной суммы —
                # хоть с кредита, хоть со вклада.
                if transfer_pair is None:
                    feasible = False

        card_blocked = any(item.is_blocked_at(step.ts) for item in state.cards.values())

        context = outcome_module.OperationContext(
            operation=operation,
            domain=step.domain,
            attempt=1,
            outage=outcome_module.in_outage(step.domain, step.ts),
            stress=stress,
            digital=state.persona.trait("digital_affinity", step.ts),
            card_blocked=card_blocked,
            insufficient=insufficient,
            feasible=feasible,
        )

        status = outcome_module.draw_status(context, rng)

        state.emit(
            state.factory.make(
                "app_operation",
                step.ts,
                {
                    "domain": step.domain,
                    "operation": operation,
                    "status": status,
                    "amount": (
                        target_bill["amount"] if target_bill
                        else (transfer_amount if operation in TRANSFER_OPERATIONS else None)
                    ),
                    "error_code": str(rng.choice(list(ERROR_CODES))) if status == "failed" else None,
                    "device_new": session.device_new,
                    "session_id": session.session_id,
                    "contract_id": None,
                },
            )
        )

        _touch_client(state, step.ts)

        if status == "failed":
            state.recent_failure_at = step.ts

        if status != "success":
            continue

        if operation in ("login", "biometry_login"):
            authorized = True
            continue

        if target_bill is not None:
            _pay_bill_in_app(state, step.ts, target_bill, session, rng, money_ts)

        if target_card is not None:
            unblock_card(state, step.ts, target_card, "client_request")

        if target_loan is not None:
            from .engine_credit import repay_loan

            repay_loan(
                state,
                step.ts,
                target_loan,
                loan_rules.arrears_amount(target_loan),
                "app",
                session_id=session.session_id,
            )

        if target_deposit is not None:
            _topup_deposit(state, step.ts, target_deposit, deposit_amount, session)

        if operation in TRANSFER_OPERATIONS:
            _transfer_in_app(sim, state, step.ts, operation, transfer_amount, transfer_pair, session)


def _can_repay(state: ClientState, ts: datetime, amount: int) -> bool:
    """
    Спишет ли repay_loan хоть что-то прямо сейчас.

    Правило то же, что у самого платежа: кредитка не в счёт, а
    частичный платёж не меньше минимальной доли. Подтягивание
    денег из другого банка в приложении не ждут: клиент видит
    нехватку на экране сразу.
    """

    from .engine_credit import _payment_capacity, _payment_sources

    if _payment_sources(state, ts, amount):
        return True

    capacity = _payment_capacity(state, ts)

    return capacity > 0 and capacity >= params_module.active().products.partial_payment_min_share * amount


def _show_banners(state: ClientState, ts: datetime, session, rng, shown_offers: set) -> None:
    """
    Баннеры на экране.

    shown_offers копит уже показанное В ЭТОЙ СЕССИИ. Один и тот
    же оффер в одном и том же слоте не показывают дважды за одно
    посещение: в ленте это выглядело двумя одинаковыми строками,
    у которых совпадало всё вплоть до секунды.
    """

    if ts < BANNERS_AVAILABLE_FROM:
        return

    if rng.random() >= params_module.active().activity.banner_screen_share:
        return

    count = 1 + int(rng.random() < 0.30)

    for index in range(count):

        slot = str(rng.choice(list(BANNER_SLOTS), p=list(BANNER_SLOT_WEIGHTS)))
        offer = str(rng.choice(list(BANNER_OFFERS)))

        if (slot, offer) in shown_offers:
            continue

        shown_offers.add((slot, offer))

        family = BANNER_OFFER_FAMILY.get(offer)

        offer_id = None
        campaign = None

        if family:
            # Показ уникален сессией, слотом и оффером: день и номер
            # на экране повторяются в каждой сессии дня.
            offer_id = f"off_{stable_hash('banner', state.client_id, session.session_id, slot, offer) % 10 ** 12:012d}"
            campaign = comm_module.campaign_for_family(family)

        # Секунда своя у каждого баннера экрана: два показа в одно
        # и то же мгновение неразличимы в ленте.
        shown_at = ts + timedelta(seconds=index * 2 + int(rng.integers(0, 2)))

        body = {
            "slot": slot,
            "offer": offer,
            "offer_id": offer_id,
            "product_id": None,
            "campaign_code": campaign,
            "session_id": session.session_id,
        }

        state.emit(
            state.factory.make(
                "banner_shown",
                shown_at,
                body,
            )
        )

        # База подобрана так, чтобы ПОСЛЕ множителей по чертам и
        # семейству продукта наблюдаемый CTR держался около
        # эталонных 2.35 % из отчёта банка.
        ctr = 0.0168 * (0.6 + 1.2 * state.persona.trait("digital_affinity", ts))

        if family in ("cash_loan", "credit_card", "installment"):
            ctr *= 1.0 + 1.5 * state.persona.trait("credit_appetite", ts)

        if family and family in state.owned_families(ts):
            ctr *= 0.4

        if rng.random() >= min(0.30, ctr):
            continue

        state.emit(
            state.factory.make(
                "banner_clicked",
                shown_at + timedelta(seconds=int(rng.integers(2, 25))),
                dict(body),
            )
        )

        if family:
            state.offers.append(
                Offer(
                    offer_id=offer_id,
                    client_id=state.client_id,
                    product_id=None,
                    product_code=None,
                    product_family=family,
                    created_at=shown_at,
                    campaign_code=campaign or "",
                    channel="banner",
                )
            )


def _pay_bill_in_app(
    state: ClientState, ts: datetime, bill: dict, session, rng, moment: datetime
) -> None:
    """
    Оплата счёта внутри сессии.

    moment — момент самого списания, выбранный ДО розыгрыша
    исхода операции. Раньше он разыгрывался здесь, и на краю
    окна выгрузки списание выпадало за HISTORY_END: операция
    показывала успех, денег за ним не было, а счёт всё равно
    уходил из open_bills.
    """

    if not in_window(moment):
        return

    sources = state.ledger.payment_sources(moment, bill["amount"])

    if not sources:
        return

    account = sources[0]

    body = dict(bill["body"])
    body["channel"] = "app"

    card = state.usable_card(account.account_id, moment)
    body["card_id"] = card.card_id if card else None
    body["session_id"] = session.session_id

    counterpart = (
        f"merchant:{body.get('merchant_id')}" if body.get("merchant_id") else COUNTERPART_GOVERNMENT
    )

    paid = _emit_money(
        state,
        moment,
        "bill_payment",
        account.account_id,
        bill["amount"],
        "debit",
        counterpart,
        body,
    )

    # Отклонённый платёж счёт не закрывает.
    if paid.payload.get("status") != "approved":
        return

    if bill in state.open_bills:
        state.open_bills.remove(bill)


def _topup_amount(state: ClientState, ts: datetime, rng) -> int:
    """
    Сколько клиент кладёт на вклад.

    Сумма разыгрывается ОТДЕЛЬНО от исполнения, чтобы её можно
    было проверить на выполнимость до розыгрыша исхода операции.
    """

    return int(round(max(5_000, state.persona.true_income * rng.uniform(0.05, 0.35)) / 1_000) * 1_000)


def _topup_deposit(state: ClientState, ts: datetime, deposit, amount: int, session) -> None:
    """
    Пополнение вклада из сессии приложения.

    Всё, что могло помешать — условия продукта, деньги на счёте,
    граница окна — проверено до того, как операция объявлена
    успешной. Здесь остаётся только движение денег.
    """

    sources = [
        item
        for item in state.ledger.payment_sources(ts, amount)
        if item.account_id != deposit.account_id
    ]

    if not sources:
        return

    moved = _own_transfer(
        state,
        ts,
        "deposit_topup",
        sources[0].account_id,
        deposit.account_id,
        amount,
        deposit.contract_id,
        "deposit_topup",
        session_id=session.session_id,
    )

    if not moved:
        return

    deposit.principal += amount


# Операции приложения, за которыми обязаны стоять живые деньги.
TRANSFER_OPERATIONS = frozenset(
    {"transfer_phone", "transfer_card", "transfer_own", "transfer_template"}
)


# Счета, между которыми клиент переводит сам. Вклад и кредит
# сюда не входят: пополнение вклада и платёж по кредиту это
# отдельные операции со своими условиями, а не «перевод себе».
OWN_TRANSFER_KINDS = frozenset({"current", "card"})


# «Договор второй ноги не задан» отличается от «договора нет».
# Без этого различия None у счёта-получателя молча подставлял бы
# договор плательщика — ту самую ошибку, из-за которой обе ноги
# ссылались на один договор.
_SAME_CONTRACT = object()


def _own_pair(state: ClientState, ts: datetime, amount: int):
    """
    Счёт списания и счёт зачисления для перевода себе.

    Списание идёт с того счёта, на котором сумма ЕСТЬ, а не с
    первого попавшегося. Если подходящей пары нет, перевода не
    будет вовсе.
    """

    sources = [
        account
        for account in state.ledger.payment_sources(ts, amount)
        if account.kind in OWN_TRANSFER_KINDS
    ]

    if not sources:
        return None

    source = sources[0]

    target = next(
        (
            account
            for account in state.ledger.visible_accounts(ts)
            if account.kind in OWN_TRANSFER_KINDS and account.account_id != source.account_id
        ),
        None,
    )

    return None if target is None else (source, target)


def _second_of_day(ts: datetime) -> int:
    return ts.hour * 3600 + ts.minute * 60 + ts.second


def _transfer_amount(state: ClientState, ts: datetime, rng) -> int:
    """
    Сколько клиент переводит из приложения.

    Желание считается от дохода: перевод в тысячу тенге у
    человека с зарплатой в миллион выглядел бы так же, как у
    человека с зарплатой в сто тысяч.

    Но человек не отправляет того, чего у него нет, и желание
    упирается в остаток счёта. Без этого предела почти каждый
    перевод в приложении оказывался невыполнимым: на счёте
    лежала тысяча, а в форму подставлялись двадцать две.

    Попытка отправить больше, чем есть, остаётся — её доля та
    же, что у остальных отказов. Она и даёт честную нехватку
    средств вместо сплошной.
    """

    income = max(50_000, int(state.persona.true_income))

    wish = max(1_000, int(income * float(rng.uniform(0.02, 0.35)) / 1_000) * 1_000)

    if rng.random() < params_module.active().activity.decline_attempt_share:
        return wish

    capacity = int(state.ledger.payment_capacity(ts) / 1_000) * 1_000

    # Денег нет вовсе: попытка всё равно делается и упирается в
    # нехватку — это и есть наблюдаемый отказ, а не молчание.
    return wish if capacity < 1_000 else min(wish, capacity)


def _transfer_in_app(
    sim,
    state: ClientState,
    ts: datetime,
    operation: str,
    amount: int,
    pair,
    session,
) -> None:
    """
    Денежное событие успешного перевода из приложения.

    Перевод между своими счетами двигает деньги внутри клиента и
    записывается парой сторон. Перевод по телефону, карте или
    шаблону клиенту этого банка тоже записывается парой: списание
    у отправителя и зачисление у получателя. Перевод в другой банк
    уходит наружу одной ногой.
    """

    if operation == "transfer_own":

        if pair is None:
            return

        source, target = pair

        # Перевод между своими счетами наблюдается парой сторон
        # с отметкой Own account: деньги не покидают клиента.
        #
        # Стороны РАЗНЫЕ: со счёта списали (transfer_out), на счёт
        # зачислили (transfer_in). Один тип на обе ноги делал из
        # прихода второй расход, и канонический слой видел два
        # списания вместо перевода.
        #
        # Договор у каждой ноги свой: списание относится к
        # договору счёта-плательщика, зачисление — к договору
        # счёта-получателя.
        _own_transfer(
            state, ts, "transfer_out",
            source.account_id, target.account_id, amount,
            source.contract_id, "own_transfer",
            credit_event_type="transfer_in",
            credit_contract_id=target.contract_id,
            transfer_id=_transfer_id(
                state.client_id, ts, _second_of_day(ts), scope=f"app:{session.session_id}"
            ),
            session_id=session.session_id,
        )

        return

    sources = state.ledger.payment_sources(ts, amount)

    if not sources:
        return

    account = sources[0]

    counterpart = _app_payee(sim, state, ts, session)

    transfer_id = _transfer_id(
        state.client_id, ts, _second_of_day(ts), scope=f"app:{session.session_id}"
    )

    body = {
        "channel": "app",
        "counterparty": counterpart.masked_name,
        "mcc": MCC_TRANSFER,
        "merchant_country": "KZ",
        "reason": "transfer",
        "session_id": session.session_id,
        "transfer_id": transfer_id,
    }

    # Получатель — клиент этого банка: перевод внутренний, и
    # зачисление видно в его ленте, как у планового перевода.
    # Обе ноги или ни одной: на краю окна зачисление уже не
    # попадает в выгрузку.
    other = sim.clients.get(counterpart.client_ordinal) if counterpart.client_ordinal is not None else None
    target = other.primary_card_account(ts) if other is not None else None

    if target is not None and in_window(ts + timedelta(seconds=1)):

        sent = _emit_money(
            state, ts, "p2p_out", account.account_id, amount, "debit", target.account_id, body,
        )

        if sent.payload.get("status") == "approved":
            from .engine import graph_counterpart_name

            _emit_money(
                other, ts + timedelta(seconds=1), "p2p_in", target.account_id, amount, "credit",
                account.account_id,
                {
                    "channel": "system",
                    "counterparty": graph_counterpart_name(state),
                    "mcc": MCC_TRANSFER,
                    "merchant_country": "KZ",
                    "reason": "transfer",
                    "transfer_id": transfer_id,
                },
            )

        return

    # Получатель в другом банке: вторая нога наблюдается его
    # собственной выгрузкой. transfer_id всё равно нужен: по нему
    # канонический слой узнаёт сторону перевода, а без него
    # списание выглядит покупкой без мерчанта.
    _emit_money(
        state, ts, "transfer_out", account.account_id, amount, "debit",
        f"external:{counterpart.counterpart_id}", body,
    )


def _app_payee(sim, state: ClientState, ts: datetime, session):
    """
    Кому клиент переводит из приложения.

    Получатель это другой человек: деньги уходят из системы
    клиента, а не на его же скрытый счёт. Берётся одна из живых
    связей клиента; без связей — постоянный внешний получатель.
    """

    people = [
        item.counterpart
        for item in sim.graph.active(state.ordinal, ts)
        if item.relation_type not in ("employer", "own_account_other_bank")
    ]

    if not people:
        return _external("person", f"app_payee:{state.client_id}")

    return people[stable_hash("app_payee", session.session_id, _second_of_day(ts)) % len(people)]


def _own_transfer(
    state: ClientState,
    ts: datetime,
    event_type: str,
    source_id: str,
    target_id: str,
    amount: int,
    contract_id: str,
    reason: str,
    credit_event_type: str | None = None,
    credit_contract_id=_SAME_CONTRACT,
    transfer_id: str | None = None,
    session_id: str | None = None,
) -> bool:
    """
    Перевод между своими счетами: две стороны с одинаковой
    суммой и отметкой Own account, чтобы деньги не выглядели
    ни внешним расходом, ни внешним доходом.

    Проводка одна: её делает первая нога, вторая только записывает
    событие с остатком. Иначе деньги двигались бы дважды.

    Пополнение вклада и снятие с него — ОДНА операция договора
    вклада, поэтому у обеих ног там один тип и один договор.
    Перевод себе — две разные стороны: для этого и нужны
    credit_event_type и credit_contract_id.

    Возвращает, состоялся ли перевод. Ложь значит, что вторая
    нога не поместилась в окно выгрузки: зачисление датируется
    секундой позже списания, и на самом краю окна его уже некуда
    записать. Половина перевода хуже, чем его отсутствие —
    деньги ушли бы со счёта и не пришли ни на какой другой.
    Вызывающий обязан свериться с ответом: остаток вклада,
    статус договора и прочее состояние меняются только при
    состоявшемся переводе.
    """

    credit_ts = ts + timedelta(seconds=1)

    if not in_window(ts) or not in_window(credit_ts):
        return False

    # Денег на счёте списания нет — перевода не будет вовсе.
    # Остаток берётся на момент списания: между решением и
    # моментом операции порядок ленты и порядок решений
    # расходятся.
    source = state.ledger.get(source_id)

    if source is None or state.ledger.available_at(source_id, ts) < int(amount):
        return False

    # Канал списания зависит от того, была ли сессия. Пополнение
    # вклада из приложения — app с session_id; то же пополнение
    # при открытии договора или списание по графику сессии не
    # принадлежит, и каналом app называться не может.
    debit = _emit_money(
        state, ts, event_type, source_id, amount, "debit", target_id,
        {
            "channel": "app" if session_id else "ecom",
            "contract_id": contract_id,
            "counterparty": "Own account",
            "reason": reason,
            "merchant_country": "KZ",
            "transfer_id": transfer_id,
            "session_id": session_id,
        },
    )

    # Отказ первой ноги оставил бы зачисление без списания:
    # деньги появились бы из ниоткуда, а у перевода осталась бы
    # одна сторона.
    if debit.payload.get("status") != "approved":
        return False

    _emit_money(
        state, credit_ts, credit_event_type or event_type,
        target_id, amount, "credit", source_id,
        {
            "channel": "system",
            "contract_id": (
                contract_id if credit_contract_id is _SAME_CONTRACT else credit_contract_id
            ),
            "counterparty": "Own account",
            "reason": reason,
            "merchant_country": "KZ",
            "transfer_id": transfer_id,
            "session_id": session_id,
        },
        post=False,
    )

    return True


def unblock_card(state: ClientState, ts: datetime, card, reason: str) -> None:

    from .finance import cards as card_rules

    # Постоянную блокировку не снимает никто. Проверка стоит
    # здесь, а не только у вызывающих: путей разблокировки
    # четыре, и пропустить один слишком легко.
    if not card.releasable():
        return

    card_rules.unblock(card, ts)

    state.emit(
        state.factory.make(
            "card_unblocked",
            ts,
            dict(state.card_facts(card), reason=reason),
        )
    )


def _on_communication(sim, state: ClientState, ts: datetime, payload: dict) -> None:

    contact = payload["contact"]

    state.comm_fatigue += 1

    # Сервисное сообщение и есть уведомление: после него повода
    # для повышенного веса сервисных рассылок больше нет.
    if contact.purpose == "service":
        state.pending_notice = False

    product_id = None

    if contact.product_family:
        for item in payload["candidates"]:
            if item.view.family == contact.product_family:
                product_id = item.view.record.product_id
                break

    state.emit(
        state.factory.make(
            "communication_sent",
            ts,
            {
                "channel": contact.channel,
                "template": contact.template,
                "campaign_code": contact.campaign_code,
                "offer_id": contact.offer_id,
                "product_id": product_id,
                "purpose": contact.purpose,
                "delivered": contact.delivered,
            },
        )
    )

    if not contact.clicked:
        return

    if contact.product_family and contact.offer_id:
        state.offers.append(
            Offer(
                offer_id=contact.offer_id,
                client_id=state.client_id,
                product_id=product_id or "",
                product_code=None,
                product_family=contact.product_family,
                created_at=ts,
                campaign_code=contact.campaign_code,
                channel=contact.channel,
            )
        )


def _on_block_expired(sim, state: ClientState, ts: datetime, payload: dict) -> None:
    """
    Долгая блокировка снимается сама: банк перевыпускает карту
    или возвращает её в строй, и это видно в ленте.
    """

    card = state.cards.get(payload["card_id"])

    if card is None or card.status != CARD_BLOCKED:
        return

    unblock_card(state, ts, card, "block_expired")


_HANDLERS["card_block_expired"] = _on_block_expired
_HANDLERS["session"] = _on_session
_HANDLERS["communication"] = _on_communication


__all__ = ["unblock_card"]
