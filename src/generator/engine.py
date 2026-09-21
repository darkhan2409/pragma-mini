from __future__ import annotations

from datetime import datetime, timedelta

from . import params as params_module
from .behaviour import adoption as adoption_module
from .behaviour import communications as comm_module
from .behaviour import habits as habits_module
from .behaviour import merchants as merchant_choice
from .behaviour import needs as needs_module
from .behaviour import sessions as session_module
from . import config
from .finance import cards as card_rules
from .finance import loans as loan_rules
from .finance.entities import (
    CARD_BLOCKED,
)
from .finance.ledger import COUNTERPART_BANK, COUNTERPART_GOVERNMENT
from .life import calendar as cal
from .life import household as household_module
from .life import lifecycle as lifecycle_module
from .life import stress as stress_module
from .observe import defects as defect_module
from .rng import (
    COMPONENT_CHANNEL,
    COMPONENT_CONTENT,
    COMPONENT_OUTCOME,
    COMPONENT_TIME,
    NS_ADOPTION,
    NS_CARD_BLOCK,
    NS_LEDGER,
    NS_PURCHASE_SOURCE,
    NS_QR,
    NS_INBOUND,
    NS_SUPPORT_CAUSE,
    NS_TRANSFER,
    day_rng,
    event_rng,
    keyed_rng,
)
from .simulate import (
    Action,
    ClientState,
    CommunityResult,
    CommunitySimulation,
    _money,
    _transfer_id,
    in_window,
)
from .world.dictionaries import (
    MCC_CASH,
    MCC_SALARY,
    MCC_TRANSFER,
)


# ============================================================
# ДВИЖОК ДНЯ
# ============================================================
#
# Все клиенты сообщества живут в одной очереди: действия дня
# собираются вместе и исполняются по времени. Внутрибанковский
# перевод поэтому доходит до получателя сразу и влияет на его
# последующие решения.
#
# Деньги двигаются только через ledger: у каждой успешной
# операции есть проводка, у каждой проводки две стороны, и
# balance_after продолжает предыдущий balance_after.
# ============================================================


def run_community(community_id: int, ordinals: tuple) -> CommunityResult:

    sim = CommunitySimulation(community_id, ordinals)

    for state in sim.clients.values():

        sim._prehistory(state)
        state.state = lifecycle_module.initial_state(state.persona, config.HISTORY_START)

        # Клиент, пришедший до окна наблюдения, известен банку с
        # первого его дня: первая версия профиля описывает то, с
        # чем он вошёл в окно. Клиент, зарегистрированный внутри
        # окна, получает первую версию в момент регистрации.
        if state.persona.relationship_start < config.HISTORY_START:
            _update_profile(state, config.HISTORY_START, moment=config.HISTORY_START)

    day = config.HISTORY_START

    while day < config.HISTORY_END:

        sim.queue = []

        for ordinal in sorted(sim.clients):
            sim.queue.extend(_plan_day(sim, sim.clients[ordinal], day))

        sim.queue.sort(key=_action_order)

        index = 0

        while index < len(sim.queue):

            action = sim.queue[index]
            index += 1

            _execute(sim, action)

            if sim.queue_changed:
                # Возврат, назначенный только что исполненной
                # покупкой, встаёт в очередь по своему времени:
                # деньги возвращаются до следующих решений дня,
                # а не после всей симуляции.
                sim.queue[index:] = sorted(sim.queue[index:], key=_action_order)
                sim.queue_changed = False

        sim.queue = []

        if (day + timedelta(days=1)).month != day.month:
            for ordinal in sorted(sim.clients):
                _month_end(sim, sim.clients[ordinal], day)

        day += timedelta(days=1)

    return _finish(sim)


# Возврат назначается уже во время дня, поэтому своего номера
# в плане у него нет: внутри одной секунды он идёт последним.
REFUND_ORDER = 10_000

# Канал операции, которую клиент сделал сам, но не в сессии
# приложения.
#
# app в денежном событии означает ровно одно: операция прошла
# внутри сессии, у неё есть session_id и парная запись
# app_operation. Всё остальное, что человек делает удалённо,
# наблюдается как операция без присутствия — ecom. Иначе в
# выгрузке появлялись строки с каналом app у клиента, который
# приложение не ставил.
CHANNEL_REMOTE = "ecom"


# ============================================================
# ПЛАН ДНЯ
# ============================================================


def _action_order(action: Action) -> tuple:
    return (action.ts, action.ordinal, action.order)


def _plan_day(sim: CommunitySimulation, state: ClientState, day: datetime) -> list:

    persona = state.persona

    if day < persona.relationship_start.replace(hour=0, minute=0, second=0, microsecond=0):
        return []

    actions: list[Action] = []
    order = 0

    def add(ts: datetime, kind: str, payload: dict) -> None:
        nonlocal order
        order += 1
        if config.HISTORY_START <= ts < config.HISTORY_END:
            actions.append(Action(ts=ts, ordinal=state.ordinal, order=order, kind=kind, payload=payload))

    silenced = lifecycle_module.silenced_streams(state.pauses, day)

    stress = stress_module.level_at(state.stress_episodes, day)

    month = cal.month_start(day)

    debt = loan_rules.debt_service(tuple(state.loans.values()), day)

    budget = household_module.budget_for_month(
        persona, month, state.income_streams, state.stress_episodes, debt
    )

    factor = household_module.spending_factor(budget, persona)

    # Деньги месяца не бесконечны: потратив их раньше срока,
    # клиент покупает реже и дешевле, а не упирается в отказы.
    factor *= household_module.budget_pressure(budget, state.month_purchases, day)
    factor *= household_module.funds_pressure(state.ledger.payment_capacity(day), budget)

    # После зарплаты тратят охотнее, перед ней придерживают.
    last_payday = None
    next_payday = None

    for payout in state.payouts:
        if payout.kind not in ("salary", "pension"):
            continue
        if payout.ts <= day:
            last_payday = payout.ts if last_payday is None or payout.ts > last_payday else last_payday
        elif next_payday is None or payout.ts < next_payday:
            next_payday = payout.ts

    factor *= cal.payday_factor(day, last_payday, next_payday)
    factor *= cal.month_factor(day)

    # --- регистрация клиента внутри окна ---

    if persona.registered_in_window and day.date() == persona.relationship_start.date():
        add(persona.relationship_start, "registration", {})

    # --- доход ---

    for payout in state.payouts:
        if payout.ts.date() == day.date():
            add(payout.ts, "income", {"payout": payout})

    # --- счета ---

    for index, bill in enumerate(state.habits.bills):

        if bill.valid_from > day or (bill.valid_to and bill.valid_to <= day):
            continue

        if cal.day_in_month(day, bill.day_of_month).date() != day.date():
            continue

        if "bills" in silenced:
            # Счёт никуда не делся, но оплачен мимо этого банка.
            state.note(day, "hidden_purchase", "bill_outside_bank", {"kind": bill.kind})
            continue

        add(
            day.replace(hour=int(9 + index % 10), minute=int((index * 7) % 60)),
            "bill",
            {"bill": bill, "index": index},
        )

    # --- подписки ---

    for index, subscription in enumerate(state.habits.subscriptions):

        if not habits_module.subscription_active(subscription, day):
            continue

        if "bills" in silenced or "purchases" in silenced:
            continue

        if cal.day_in_month(day, subscription.day_of_month).date() != day.date():
            continue

        add(
            day.replace(hour=int(2 + index % 5), minute=int((index * 13) % 60)),
            "subscription",
            {"subscription": subscription, "index": index},
        )

    # --- покупки ---

    for index, intent in enumerate(
        needs_module.daily_intents(persona, state.habits, day, state.state, factor, silenced)
    ):
        add(intent.ts, "purchase",
            {"intent": intent, "index": index, "budget": budget,
             "factor": factor, "stress": stress})

    # --- наличные ---

    if needs_module.cash_need(persona, day, silenced):
        rng = keyed_rng(NS_LEDGER, state.ordinal, day.toordinal(), 7)
        ts = day.replace(hour=int(rng.integers(9, 21)), minute=int(rng.integers(0, 60)))
        add(ts, "cash_withdrawal", {"budget": budget})

    # --- внесение наличных ---

    if "cash" not in silenced and state.ledger.balance(state.ledger.cash_id) > persona.true_income * 0.4:
        rng = keyed_rng(NS_LEDGER, state.ordinal, day.toordinal(), 13)
        if rng.random() < 0.05:
            add(day.replace(hour=int(rng.integers(10, 20)), minute=int(rng.integers(0, 60))),
                "cash_deposit", {})

    # --- переводы ---

    if "transfers" not in silenced:

        active = [
            item
            for item in sim.graph.active(state.ordinal, day)
            if item.relation_type != "employer"
        ]

        planned = sum(item.typical_frequency for item in active)

        budget_per_month = (
            params_module.active().activity.transfers_per_month[persona.activity_mode]
            * params_module.active().activity.state_factor.get(state.state, 1.0)
            * persona.visible_share
            * 1.4
            # Общительный клиент переводит деньги чаще.
            * (
                0.6
                + params_module.active().traits.sociality_transfer_factor
                * persona.trait("sociality", day)
            )
        )

        scale = (budget_per_month / planned) if planned > 0 else 0.0

        for index, relation in enumerate(active):

            rate = relation.typical_frequency * scale / 30.0

            if relation.relation_type == "landlord":
                if cal.day_in_month(day, min(28, persona.salary_day + 2)).date() != day.date():
                    continue
                rate = 1.0

            rng = event_rng(NS_TRANSFER, state.ordinal, day.toordinal(), index, COMPONENT_CONTENT)

            if rng.random() >= rate:
                continue

            ts = day.replace(hour=int(rng.integers(9, 22)), minute=int(rng.integers(0, 60)))

            add(ts, "transfer", {"relation": relation, "index": index})

    # --- входящие переводы извне ---
    #
    # Внутрибанковские приходы порождает исходящий план другого
    # клиента. Здесь только внешние отправители: родня, друзья,
    # постоянные контрагенты.

    for index, relation in enumerate(sim.graph.active(state.ordinal, day)):

        if relation.inbound_frequency <= 0.0:
            continue

        if relation.counterpart.client_ordinal is not None:
            continue

        rng = event_rng(NS_INBOUND, state.ordinal, day.toordinal(), index, COMPONENT_TIME)

        if rng.random() >= relation.inbound_frequency / 30.0:
            continue

        moment = day.replace(
            hour=int(rng.integers(8, 22)),
            minute=int(rng.integers(0, 60)),
            second=0,
        )

        add(moment, "inbound_transfer", {"relation": relation, "index": index})

    # --- сессии приложения ---

    app_adopted = state.app_adopted_at is not None and day >= state.app_adopted_at

    context = session_module.SessionContext(
        due_bills=tuple(
            bill.kind
            for bill in state.habits.bills
            if bill.valid_from <= day and cal.day_in_month(day, bill.day_of_month) >= day - timedelta(days=6)
            and cal.day_in_month(day, bill.day_of_month) <= day
        ),
        card_blocked=any(card.is_blocked_at(day) for card in state.cards.values()),
        recent_offer_family=state.offers[-1].product_family if state.offers else None,
        has_loan=state.has_open_loan(),
        has_deposit=state.assets() > 0,
        has_card=any(card.usable_at(day) for card in state.cards.values()),
        accounts=len(state.ledger.visible_accounts(day)),
        dpd=state.worst_dpd(),
        salary_just_arrived=any(
            payout.ts.date() == (day - timedelta(days=1)).date() for payout in state.payouts
        ),
        recent_failure=state.recent_failure_at is not None
        and (day - state.recent_failure_at).days <= 3,
        fraud_alert=state.fraud_alert_at is not None and (day - state.fraud_alert_at).days <= 3,
    )

    for index, session in enumerate(
        session_module.plan_sessions(
            persona, day, state.state, silenced, app_adopted, context, state.stress_episodes
        )
    ):
        add(session.started_at, "session", {"session": session, "index": index})

    # --- просрочка и ближайший платёж на сегодня ---
    #
    # worst_dpd обновляется вечерней проверкой кредита, поэтому
    # планировщик видел бы вчерашнее состояние.

    live_dpd = 0
    days_to_due = None

    for loan in state.loans.values():

        if loan.closed:
            continue

        live_dpd = max(live_dpd, loan_rules.days_past_due(loan, day))

        upcoming = loan.next_due(day)

        if upcoming is not None:
            remaining = (upcoming.due_date - day).days
            if days_to_due is None or remaining < days_to_due:
                days_to_due = remaining

    # --- коммуникации ---

    if state.consent_at is not None and day >= state.consent_at:

        candidates = adoption_module.candidates(
            persona,
            day,
            state.held_codes(day),
            state.held_counts(day),
            state.assets(),
            app_adopted,
            state.open_loan_count(),
            len(state.open_contracts(day)),
            state.income_months_at(day),
            stress,
        )

        families = frozenset(item.view.family for item in candidates)

        contacts = comm_module.contacts_for_day(
            persona=persona,
            day=day,
            consented=True,
            app_adopted=app_adopted,
            fatigue=state.comm_fatigue,
            state_factor=params_module.active().activity.state_factor.get(state.state, 1.0),
            owned_families=state.owned_families(day),
            candidate_families=families,
            dpd=live_dpd,
            in_pause=lifecycle_module.pause_at(state.pauses, day) is not None,
            stress=stress,
            pending_notice=state.pending_notice,
            fraud_alert=state.fraud_alert_at is not None and (day - state.fraud_alert_at).days <= 5,
            days_to_due=days_to_due,
        )

        for index, contact in enumerate(contacts):
            add(contact.ts, "communication", {"contact": contact, "index": index, "candidates": candidates})

    # --- кредитное обслуживание ---

    for contract_id, loan in list(state.loans.items()):

        if loan.closed:
            continue

        item = loan_rules.due_today(loan, day)

        if item is not None:
            add(day.replace(hour=0, minute=0, second=0), "installment_due",
                {"contract_id": contract_id, "installment": item})

        # Клиент платит сам: в срок или в льготные дни. Решение
        # по каждому взносу принято заранее и от порядка
        # исполнения дня не зависит.
        #
        # Смотреть надо на взнос ЭТОГО периода, а не на самый
        # старый неоплаченный: иначе отставший клиент больше
        # никогда не получит намерения заплатить, потому что
        # дата старого взноса давно прошла, и просрочка станет
        # вечной.
        if not loan.autopay:

            grace = params_module.active().products.grace_days_before_missed

            for item in loan.schedule:

                if item.outstanding <= 0:
                    continue

                elapsed = (day - item.due_date).days

                if elapsed < 0 or elapsed > grace:
                    continue

                plan = _payment_plan(state, loan, item)

                if plan["will_pay"] and plan["offset"] == elapsed:
                    add(
                        day.replace(hour=plan["hour"], minute=plan["minute"], second=0),
                        "loan_payment_intent",
                        {"contract_id": contract_id},
                    )
                    break

        add(day.replace(hour=23, minute=30), "loan_check", {"contract_id": contract_id})

    # --- органический интерес к продукту ---

    if day >= persona.relationship_start and "purchases" not in silenced:

        # Момент интереса разыгрывается, а не назначается. Раньше
        # здесь стояло ровно 20:15, и ВСЕ заявки выгрузки падали
        # в одну минуту суток — шаблон, которого в жизни нет.
        pick = keyed_rng(NS_ADOPTION, state.ordinal, day.toordinal(), 3)

        add(
            day.replace(
                hour=int(pick.integers(9, 23)),
                minute=int(pick.integers(0, 60)),
                second=int(pick.integers(0, 60)),
            ),
            "adoption",
            {"stress": stress, "app": app_adopted},
        )

    # --- крупная покупка как жизненное событие ---
    #
    # Раньше событие только запускало стресс, а самой покупки
    # не порождало: «крупная покупка» ничего не покупала.

    for index, event in enumerate(state.life_events):

        if event.kind != "big_purchase" or event.ts.date() != day.date():
            continue

        if "purchases" in silenced:
            continue

        category = str(event.payload.get("category") or "home_goods")

        moment = day.replace(
            hour=int(event.ts.hour) or 15, minute=int(event.ts.minute), second=0
        )

        add(
            moment,
            "purchase",
            {
                "intent": needs_module.Intent(
                    ts=moment, category=category, zone="other", from_routine=False
                ),
                "index": 900 + index,
                "budget": budget,
                "factor": factor * float(event.payload.get("amount_factor") or 1.0),
                "stress": stress,
            },
        )

    # --- мошеннические эпизоды ---

    for index, episode in enumerate(state.fraud_episodes):

        # Шаги, которые делает сам клиент, в паузе не случаются:
        # молчащий клиент не покупает за границей и не переводит
        # деньги «службе безопасности». Чужие руки паузой не
        # ограничены.
        if episode.kind == "false_positive" and "purchases" in silenced:
            continue

        if episode.kind == "social_engineering" and "transfers" in silenced:
            continue

        for position, step in enumerate(episode.steps):
            if step.ts.date() == day.date():
                add(step.ts, "fraud_step", {"episode": episode, "step": step,
                                            "index": index, "position": position})

    # --- клиент блокирует свою карту ---
    #
    # Странное списание в выписке, поездка, карта не нашлась в
    # кармане. Чаще это временная заморозка, которую клиент сам
    # же и снимает; реже карта потеряна или скомпрометирована, и
    # тогда размораживать нечего, нужен перевыпуск.

    if day >= persona.relationship_start and "purchases" not in silenced:

        usable = [
            card
            for card in state.cards.values()
            if card.usable_at(day) and card.closed_at is None
        ]

        if usable:

            pick = day_rng(NS_CARD_BLOCK, state.ordinal, day.toordinal())

            products = params_module.active().products

            if pick.random() < products.card_block_client_share_per_year / 365.25:

                card = usable[int(pick.integers(0, len(usable)))]

                add(
                    day.replace(hour=int(pick.integers(8, 22)), minute=int(pick.integers(0, 60))),
                    "card_block_request",
                    {
                        "card_id": card.card_id,
                        "lost": bool(pick.random() < products.card_block_lost_share),
                    },
                )

    # --- истёкшая блокировка карты ---

    for card in state.cards.values():
        if (
            card.status == CARD_BLOCKED
            and card.blocked_until is not None
            and card.blocked_until.date() == day.date()
        ):
            add(card.blocked_until, "card_block_expired", {"card_id": card.card_id})

    # --- обращение в поддержку ---

    # Поводов может быть несколько сразу. Раньше цепочка elif
    # прятала заблокированную карту за просрочкой, и в данных
    # оставалась почти одна тема.
    causes: list[str] = []

    if state.recent_failure_at is not None and (day - state.recent_failure_at).days <= 2:
        causes.append("failed_operation")

    # Повод обратиться даёт СВЕЖАЯ блокировка. Раньше условие
    # смотрело «заблокирована сейчас», и навсегда заблокированная
    # карта делала бы эту тему поводом до конца истории.
    if any(
        item.blocked_at is not None
        and item.closed_at is None
        and 0 <= (day - item.blocked_at).days <= 7
        and item.is_blocked_at(day)
        for item in state.cards.values()
    ):
        causes.append("card_blocked")

    if live_dpd >= 30:
        causes.append("delinquency")
    elif any(
        event.event_type == "installment_missed" and (day - event.event_time).days <= 3
        for event in state.events[-40:]
    ):
        causes.append("missed_installment")

    if causes:

        pick = keyed_rng(NS_SUPPORT_CAUSE, state.ordinal, day.toordinal())

        cause = causes[int(pick.integers(0, len(causes)))]

        last = state.support_last_by_cause.get(cause)

        cooldown = params_module.active().activity.support_cooldown_days

        if last is None or (day - last).days >= cooldown:
            add(day.replace(hour=13, minute=20), "support_check", {"cause": cause, "stress": stress})

    # --- просроченные счета к оплате ---

    if state.open_bills:
        add(day.replace(hour=21, minute=5), "bill_sweep", {})

    # --- назначенные ранее возвраты ---

    for key in sorted(moment for moment in state.pending_refunds if moment <= day.toordinal()):
        for plan in state.pending_refunds.pop(key):
            add(plan["ts"], "refund", {"plan": plan})

    # --- банк узнал об изменении профиля ---

    for index, event in enumerate(state.life_events):
        if event.known_to_bank_at is not None and event.known_to_bank_at.date() == day.date():
            add(event.known_to_bank_at, "profile_change", {"event": event, "index": index})

    return actions


# ============================================================
# ИСПОЛНЕНИЕ
# ============================================================


def _execute(sim: CommunitySimulation, action: Action) -> None:
    """
    Одно действие плана.

    Вид без обработчика это ошибка сборки, а не пустой день:
    молчаливый пропуск отнимал бы у клиента целый механизм
    поведения, и данные выглядели бы правдоподобно.
    """

    state = sim.clients[action.ordinal]

    handler = _HANDLERS.get(action.kind)

    if handler is None:
        raise KeyError(f"нет обработчика для действия {action.kind!r}")

    handler(sim, state, action.ts, action.payload)


def _touch_client(state: ClientState, ts: datetime) -> None:
    """
    Клиентское действие: оно и определяет паузы и возвращения.
    """

    previous = state.last_client_event

    if previous is not None and (ts - previous).days >= 45:
        state.returned_flag = True
        state.note(ts, "pause_end", "return", {"silence_days": (ts - previous).days})

    state.last_client_event = ts


# --- деньги ---------------------------------------------------


def _pick_spending_account(state: ClientState, ts: datetime, sources: list, stress: float):
    """
    Каким счётом платят. Под стрессом кредитный лимит
    поднимается в очереди источников.
    """

    if len(sources) < 2 or stress <= 0.0:
        return sources[0]

    credit = next((item for item in sources if item.kind == "credit_card"), None)

    if credit is None or credit is sources[0]:
        return sources[0]

    rise = params_module.active().stress.utilization_rise

    rng = event_rng(NS_PURCHASE_SOURCE, state.ordinal, ts.toordinal(), int(ts.hour), COMPONENT_CHANNEL)

    return credit if rng.random() < min(0.9, rise * stress) else sources[0]


def _register_card_debt(
    state: ClientState, account_id: str, amount: int, ts, is_cash: bool, cause_event_id: str
) -> None:
    """
    Трата по кредитной карте становится долгом: покупка идёт в
    рассрочку, наличные копят проценты.
    """

    if amount <= 0:
        return

    account = state.ledger.accounts.get(account_id)

    if account is None or account.kind != "credit_card":
        return

    credit = state.card_credits.get(account.contract_id)

    if credit is None or credit.closed:
        return

    if is_cash:
        card_rules.add_cash(credit, amount)
    else:
        card_rules.add_purchase(credit, amount, cal.month_index(ts), cause_event_id)


def _release_card_debt(
    state: ClientState, account_id: str, amount: int, cause_event_id: str | None
) -> None:
    """
    Возврат по кредитной карте снимает долг ТОЙ покупки, которую
    вернули, а не просто кладёт деньги на счёт.

    Раньше покупка ставила части рассрочки в график, а её полный
    возврат их не убирал: клиент оставался должен банку за то,
    чего не покупал. Потом возврат снимал долг без разбора и
    добирался до наличного: возврат уже выплаченной покупки
    гасил снятые наличные. Возврат без причины долга не снимает:
    непонятно, чьего.
    """

    if not cause_event_id:
        return

    account = state.ledger.accounts.get(account_id)

    if account is None or account.kind != "credit_card":
        return

    credit = state.card_credits.get(account.contract_id)

    if credit is None or credit.closed:
        return

    card_rules.reverse_purchase(credit, int(amount), cause_event_id)


def _emit_money(
    state: ClientState,
    ts: datetime,
    event_type: str,
    account_id: str | None,
    amount: int,
    direction: str,
    counterpart_account: str,
    payload: dict,
    status: str = "approved",
    post: bool = True,
):
    """
    Одна денежная операция: проводка и событие с balance_after.
    Отклонённая операция проводки не создаёт.

    post=False — вторая нога перевода между своими счетами:
    событие с остатком есть, проводки нет, её сделала первая нога.
    """

    account = state.ledger.get(account_id) if account_id else None

    # Одобренное списание не может увести счёт в минус. Проверка
    # стоит ЗДЕСЬ, а не у каждого вызывающего: путей списания
    # полтора десятка, счёт выбирается заранее, а исполняется
    # операция позже — за это время соседние события дня успевают
    # снять деньги, и остаток уходил ниже нуля у обычной карты.
    #
    # available знает про кредитный лимит: по кредитной карте
    # минус разрешён ровно до него, по дебетовому счёту лимита
    # нет и минуса быть не может.
    #
    # Нехватка денег превращается в ОТКАЗ, а не в тихий пропуск:
    # попытка была, и банк её видел. Проводки у отказа нет, так
    # что ни остаток, ни ledger не меняются.
    if (
        status == "approved"
        and post
        and direction == "debit"
        and account is not None
        and account.available < int(amount)
    ):
        status = "declined"
        payload = dict(payload)
        payload.setdefault("decline_reason", "insufficient_funds")

    body = dict(payload)
    body["amount"] = int(amount)
    body["direction"] = direction
    body["status"] = status
    body["account_id"] = account_id
    body.setdefault("currency", "KZT")

    # Покупка за рубежом прошла в чужой валюте, а на счёт легла
    # в тенге по курсу.
    country = body.get("merchant_country")

    if country and country != "KZ" and body.get("original_amount") is None:

        fx = params_module.active().amounts

        code = fx.country_currency.get(country)

        rate = fx.fx_rates.get(code) if code else None

        if rate:
            body["original_currency"] = code
            body["original_amount"] = int(round(int(amount) / rate))

    event = state.factory.make(event_type, ts, body)

    # За концом окна проводки нет. Строку туда всё равно не
    # записать, а остаток она бы изменила — и следующий срез
    # увидел бы деньги, которых выгрузка не показывает. До
    # НАЧАЛА окна проводка, наоборот, нужна: там мир жил.
    if status == "approved" and account is not None and ts < config.HISTORY_END:

        # Вторая нога перевода между своими счетами денег не
        # двигает: проводка первой ноги уже изменила оба остатка.
        # Повторная проводка удваивала пополнение вклада в ledger,
        # снимала с карты вдвое больше и считала проценты с
        # удвоенного остатка. RAW этого не показывал: balance_after
        # там пересчитывается по ленте, страдала только симуляция.
        if post:
            if direction == "debit":
                state.ledger.post(ts, event.event_id, account_id, counterpart_account, int(amount))
            else:
                state.ledger.post(ts, event.event_id, counterpart_account, account_id, int(amount))

        event.payload["balance_after"] = account.balance

        # Трата по кредитной карте это долг, а не просто минус
        # на счёте: она встаёт в рассрочку или копит проценты.
        if direction == "debit" and event_type not in ("loan_payment", "fee_charge"):
            _register_card_debt(
                state,
                account_id,
                int(amount),
                ts,
                event_type in ("cash_withdrawal", "transfer_out", "p2p_out"),
                event.event_id,
            )

        # Возврат, отмена и chargeback идут обратным ходом: долг
        # той же покупки уменьшается на вернувшуюся сумму.
        if direction == "credit" and event_type in ("refund", "reversal", "chargeback"):
            _release_card_debt(state, account_id, int(amount), event.payload.get("cause_event_id"))

    return state.emit(event)


def _decline(state: ClientState, ts: datetime, event_type: str, account_id: str | None,
             amount: int, direction: str, payload: dict, reason: str):

    body = dict(payload)
    body["decline_reason"] = reason

    return _emit_money(
        state, ts, event_type, account_id, amount, direction,
        "external:none", body, status="declined",
    )


# --- обработчики ----------------------------------------------


def _on_registration(sim, state: ClientState, ts: datetime, payload: dict) -> None:

    view = sim._pick_product(state, "debit_card", ts)

    if view is None:
        return

    sim._open_contract(state, view, ts.replace(hour=12), None, None)

    _touch_client(state, ts)

    # Профиль появляется вместе с клиентом, а не в конце месяца.
    _update_profile(state, ts, moment=ts)

    state.note(ts, "state_transition", lifecycle_module.STATE_ONBOARDING, {"cause": "registration"})


def _on_income(sim, state: ClientState, ts: datetime, payload: dict) -> None:

    payout = payload["payout"]

    amount = int(payout.amount)

    if amount <= 0:
        return

    counterpart = f"employer:{payout.payer}"

    if payout.landing == "cash":
        state.ledger.post(ts, "hidden", counterpart, state.ledger.cash_id, amount)
        state.note(ts, "income_event", "cash", {"amount": amount, "kind": payout.kind})
        return

    if payout.landing == "other_bank":
        state.ledger.post(ts, "hidden", counterpart, state.ledger.other_bank_id, amount)
        state.note(ts, "income_event", "other_bank", {"amount": amount, "kind": payout.kind})
        return

    account = state.primary_card_account(ts)

    if account is None:
        state.ledger.post(ts, "hidden", counterpart, state.ledger.cash_id, amount)
        return

    event_type = {
        "pension": "pension_credit",
        "salary": "salary_credit",
    }.get(payout.kind, "other_income_credit")

    _emit_money(
        state, ts, event_type, account.account_id, amount, "credit", counterpart,
        {
            "channel": "system",
            "mcc": MCC_SALARY,
            "counterparty": payout.payer if payout.kind != "salary" else "Employer",
            "reason": payout.outcome,
            "merchant_country": "KZ",
        },
    )

    state.note(ts, "income_event", payout.outcome, {"amount": amount, "kind": payout.kind})


def _on_bill(sim, state: ClientState, ts: datetime, payload: dict) -> None:

    bill = payload["bill"]

    rng = event_rng(NS_LEDGER, state.ordinal, ts.toordinal(), payload["index"], COMPONENT_CONTENT)

    amount = habits_module.bill_amount(bill, ts, state.persona.region, rng)

    silenced = lifecycle_module.silenced_streams(state.pauses, ts)

    if "purchases" in silenced and not bill.autopay:
        return

    account = None

    for candidate in state.ledger.payment_sources(ts, amount):
        account = candidate
        break

    choice = merchant_choice.choose_outlet(
        state.persona, state.habits, bill.category, ts, rng, online_hint=True
    )

    merchant = choice.outlet if choice else None

    body = {
        # Счёт оплачен без сессии приложения, значит и каналом
        # app он быть не может: у операции в приложении есть
        # session_id и своя запись app_operation. Самостоятельная
        # оплата онлайн наблюдается как ecom, автоплатёж — как
        # действие самого банка.
        "channel": "ecom" if not bill.autopay else "system",
        "merchant_id": merchant.merchant_id if merchant else None,
        "outlet_id": merchant.outlet_id if merchant else None,
        "merchant_name": merchant.merchant_name if merchant else None,
        "mcc": merchant.mcc if merchant else None,
        "merchant_city": merchant.settlement if merchant else None,
        "merchant_country": "KZ",
        "is_online": bool(merchant.is_online) if merchant is not None else True,
        "is_subscription": False,
        "reason": f"bill_{bill.kind}",
    }

    if not bill.autopay:
        # Счёт остаётся к оплате: клиент заплатит его в приложении
        # либо, не успев, мимо банка у срока.
        state.open_bills.append(
            {
                "kind": bill.kind,
                "category": bill.category,
                "amount": amount,
                "due": ts,
                "deadline": ts + timedelta(days=10),
                "body": body,
            }
        )
        return

    if account is None:

        # Автоплатёж чаще всего просто не за что списывать, и
        # счёт закрывается мимо банка. Наблюдаемая неудачная
        # попытка списания случается реже.
        fail_rng = keyed_rng(NS_LEDGER, state.ordinal, ts.toordinal(), 71)

        if fail_rng.random() < params_module.active().activity.autopay_attempt_share:
            _decline(state, ts, "bill_payment", None, amount, "debit", body,
                     "insufficient_funds")

        state.note(ts, "hidden_purchase", "bill_unpaid_in_bank", {"amount": amount, "kind": bill.kind})
        return

    card = state.usable_card(account.account_id, ts)

    body["card_id"] = card.card_id if card else None

    paid = _emit_money(
        state, ts, "bill_payment", account.account_id, amount, "debit",
        f"merchant:{merchant.outlet_id}" if merchant else COUNTERPART_GOVERNMENT,
        body,
    )

    # Отклонённое списание счёт не закрывает: он остался неоплачен.
    if paid.payload.get("status") != "approved":
        state.note(ts, "hidden_purchase", "bill_unpaid_in_bank",
                   {"amount": amount, "kind": bill.kind})


def _on_subscription(sim, state: ClientState, ts: datetime, payload: dict) -> None:

    subscription = payload["subscription"]

    amount = habits_module.subscription_amount(subscription, ts)

    account = None

    for candidate in state.ledger.payment_sources(ts, amount):
        account = candidate
        break

    outlet = None

    for item in sim._outlets_by_id(state, "subscription", ts):
        if item.outlet_id == subscription.outlet_id:
            outlet = item
            break

    body = {
        "channel": "ecom",
        "merchant_id": outlet.merchant_id if outlet else None,
        "outlet_id": subscription.outlet_id,
        "merchant_name": outlet.merchant_name if outlet else None,
        "mcc": outlet.mcc if outlet else "5815",
        "merchant_city": outlet.settlement if outlet else None,
        "merchant_country": "KZ",
        "is_online": True,
        "is_subscription": True,
        "reason": "subscription",
    }

    if account is None:

        fail_rng = keyed_rng(NS_LEDGER, state.ordinal, ts.toordinal(), 72)

        if fail_rng.random() < params_module.active().activity.autopay_attempt_share:
            _decline(state, ts, "purchase", None, amount, "debit", body,
                     "insufficient_funds")
            return

        state.note(ts, "hidden_purchase", "subscription_outside_bank", {"amount": amount})
        return

    body["card_id"] = state.usable_card(account.account_id, ts).card_id if state.usable_card(account.account_id, ts) else None

    event = _emit_money(
        state, ts, "purchase", account.account_id, amount, "debit",
        f"merchant:{subscription.outlet_id}", body,
    )

    _schedule_refunds(sim, state, event)
    state.month_purchases += amount


def _on_purchase(sim, state: ClientState, ts: datetime, payload: dict) -> None:

    intent = payload["intent"]
    budget = payload["budget"]

    persona = state.persona

    rng = event_rng(NS_LEDGER, state.ordinal, ts.toordinal(), payload["index"] + 50, COMPONENT_CONTENT)

    # Фактор трат посчитан планировщиком дня и уже учитывает
    # бюджет месяца, остаток на счёте и зарплатный цикл.
    factor = payload["factor"]

    travel = None
    foreign = None

    vacation = None

    for event in state.life_events:
        if event.kind != "vacation":
            continue
        if event.ts <= ts < event.ts + timedelta(days=int(event.payload.get("days", 0))):
            vacation = event
            break

    if vacation is not None and vacation.payload.get("abroad"):
        foreign = str(
            rng.weighted(params_module.active().geography.foreign_countries)
        )

    choice = merchant_choice.choose_outlet(
        persona, state.habits, intent.category, ts, rng,
        travel_settlement=travel, foreign_country=foreign,
    )

    if choice is None:
        return

    amount = merchant_choice.purchase_amount(
        persona, intent.category, choice.outlet, ts, factor, rng
    )

    sources = state.ledger.payment_sources(ts, amount)

    # Оплата по QR это тот же поход в магазин, но другой канал.
    channel = choice.channel

    if channel == "pos" and state.app_adopted_at is not None and ts >= state.app_adopted_at:

        activity = params_module.active().activity

        if persona.trait("digital_affinity", ts) >= activity.qr_min_digital_affinity:

            qr_rng = event_rng(NS_QR, state.ordinal, ts.toordinal(), payload["index"], COMPONENT_CHANNEL)

            if qr_rng.random() < activity.qr_share_of_pos:
                channel = "qr"

    body = {
        "channel": channel,
        "merchant_id": choice.outlet.merchant_id,
        "outlet_id": choice.outlet.outlet_id,
        "merchant_name": choice.outlet.merchant_name,
        "mcc": choice.outlet.mcc,
        "merchant_city": choice.outlet.settlement or None,
        "merchant_country": choice.outlet.country,
        "is_online": choice.outlet.is_online,
        "is_subscription": False,
        "reason": "routine" if intent.from_routine else "purchase",
    }

    if not sources:

        # Денег на счёте нет. Чаще всего банк этого даже не
        # видит: клиент платит наличными, деньгами в другом
        # банке или откладывает покупку. Наблюдаемый отказ
        # редок, и после пары отказов за день клиент перестаёт
        # пробовать.
        settings = params_module.active().activity

        hidden = state.ledger.hidden_sources(amount)

        if hidden and rng.random() < settings.hidden_purchase_share:
            state.ledger.post(ts, "hidden", hidden[0].account_id, f"merchant:{choice.outlet.outlet_id}", amount)
            state.note(ts, "hidden_purchase", intent.category, {"amount": amount})
            return

        if rng.random() < settings.decline_attempt_share and state.may_decline(ts):
            _decline(state, ts, "purchase", None, amount, "debit", body,
                     "insufficient_funds")
            _touch_client(state, ts)
            return

        state.note(ts, "hidden_purchase", intent.category,
                   {"amount": amount, "reason": "postponed"})
        return

    # В трудный период чаще расплачиваются кредитным лимитом,
    # а не своими деньгами.
    account = _pick_spending_account(state, ts, sources, payload.get("stress", 0.0))

    card = state.usable_card(account.account_id, ts)

    if not choice.outlet.is_online and card is None:

        blocked = any(
            item.account_id == account.account_id and item.is_blocked_at(ts)
            for item in state.cards.values()
        )

        if blocked:
            body["card_id"] = next(
                (item.card_id for item in state.cards.values() if item.account_id == account.account_id),
                None,
            )
            _decline(state, ts, "purchase", account.account_id, amount, "debit", body,
                     "card_blocked")
            _touch_client(state, ts)
            return

    body["card_id"] = card.card_id if card else None

    event = _emit_money(
        state, ts, "purchase", account.account_id, amount, "debit",
        f"merchant:{choice.outlet.outlet_id}", body,
    )

    _schedule_refunds(sim, state, event)
    state.month_purchases += amount

    _touch_client(state, ts)

    # --- кешбэк по тарифу версии договора ---

    contract = state.contracts.get(account.contract_id) if account.contract_id else None

    if contract is not None and contract.terms:

        months = max(0, cal.month_index(ts) - cal.month_index(contract.opened_at))

        value = card_rules.cashback_amount(
            contract.terms, intent.category, amount, account.balance, state.assets(), months
        )

        cap = card_rules.cashback_cap(contract.terms)

        if cap:
            value = max(0, min(value, cap - state.monthly_cashback))

        if value > 0:
            # Кешбэк начисляется не за каждую покупку, а один раз
            # в месяц: у периодического начисления нет отдельного
            # события-причины.
            state.monthly_cashback += value
            key = (contract.contract_id, account.account_id)
            state.pending_cashback[key] = state.pending_cashback.get(key, 0) + value


def _schedule_refunds(sim, state: ClientState, event) -> None:
    """
    Возврат и отмена назначаются в момент покупки.

    Раньше они рождались после всей симуляции, и вернувшиеся
    деньги не влияли ни на одно решение клиента: он продолжал
    жить так, будто покупка не отменялась. Теперь возврат
    встаёт в очередь своего дня и доходит до счёта вовремя.
    """

    for plan in defect_module.plan_refunds([event]):

        moment = plan["ts"]

        if not (config.HISTORY_START <= moment < config.HISTORY_END):
            continue

        if moment.toordinal() == event.event_time.toordinal():
            sim.schedule(
                Action(
                    ts=moment,
                    ordinal=state.ordinal,
                    order=REFUND_ORDER,
                    kind="refund",
                    payload={"plan": plan},
                )
            )
            continue

        state.pending_refunds.setdefault(moment.toordinal(), []).append(plan)


def _on_refund(sim, state: ClientState, ts: datetime, payload: dict) -> None:
    """
    Возврат или отмена: ссылается на исходную операцию и не
    превышает её сумму.
    """

    plan = payload["plan"]

    cause = plan["cause"]

    account_id = cause.payload.get("account_id")

    if account_id is None:
        return

    body = {
        "channel": "system",
        "card_id": cause.payload.get("card_id"),
        "merchant_id": cause.payload.get("merchant_id"),
        "outlet_id": cause.payload.get("outlet_id"),
        "merchant_name": cause.payload.get("merchant_name"),
        "mcc": cause.payload.get("mcc"),
        "merchant_city": cause.payload.get("merchant_city"),
        "merchant_country": cause.payload.get("merchant_country"),
        "cause_event_id": cause.event_id,
        "reason": plan["kind"],
        "is_online": cause.payload.get("is_online"),
        "is_subscription": False,
    }

    _emit_money(
        state,
        ts,
        plan["kind"],
        account_id,
        int(plan["amount"]),
        "credit",
        f"merchant:{cause.payload.get('outlet_id')}"
        if cause.payload.get("outlet_id")
        else "external:merchant",
        body,
    )


def _on_cash(sim, state: ClientState, ts: datetime, payload: dict) -> None:

    budget = payload["budget"]

    rng = keyed_rng(NS_LEDGER, state.ordinal, ts.toordinal(), 11)

    low, high = params_module.active().activity.cash_withdrawal_share_of_income

    amount = _money(max(2_000, budget.income * rng.uniform(low, high)))

    amount = int(round(amount / 1_000) * 1_000)

    # В банкомате снимают то, что есть, а не задуманную сумму.
    capacity = state.ledger.payment_capacity(ts)

    if amount > capacity > 0:
        trimmed = int(capacity / 1_000) * 1_000
        if trimmed >= 2_000:
            amount = trimmed

    sources = state.ledger.payment_sources(ts, amount)

    body = {
        "channel": "atm",
        "mcc": MCC_CASH,
        "merchant_country": "KZ",
        "is_online": False,
        "reason": "cash_need",
    }

    if not sources:

        settings = params_module.active().activity

        if rng.random() < settings.decline_attempt_share and state.may_decline(ts):
            _decline(state, ts, "cash_withdrawal", None, amount, "debit", body,
                     "insufficient_funds")
            return

        state.note(ts, "hidden_purchase", "cash_need", {"amount": amount, "reason": "postponed"})
        return

    account = sources[0]

    card = state.usable_card(account.account_id, ts)

    body["card_id"] = card.card_id if card else None

    _emit_money(
        state, ts, "cash_withdrawal", account.account_id, amount, "debit",
        state.ledger.cash_id, body,
    )

    _touch_client(state, ts)

    contract = state.contracts.get(account.contract_id) if account.contract_id else None

    if contract is not None:

        fee = card_rules.withdrawal_fee(
            contract.terms, amount, state.monthly_atm, state.monthly_atm_count
        )

        state.monthly_atm += amount
        state.monthly_atm_count += 1

        if fee > 0 and state.ledger.can_debit(account.account_id, fee):
            _emit_money(
                state, ts + timedelta(seconds=5), "fee_charge", account.account_id, fee, "debit",
                COUNTERPART_BANK,
                {
                    "channel": "system",
                    "contract_id": contract.contract_id,
                    "reason": "atm_withdrawal_fee",
                    "accrual_period": ts.strftime("%Y-%m"),
                    "merchant_country": "KZ",
                },
            )


def _on_transfer(sim, state: ClientState, ts: datetime, payload: dict) -> None:

    relation = payload["relation"]

    settings = params_module.active().relationships

    rng = event_rng(NS_TRANSFER, state.ordinal, ts.toordinal(), payload["index"] + 200, COMPONENT_OUTCOME)

    amount = int(rng.integers(max(1_000, relation.typical_amount_low),
                              max(2_000, relation.typical_amount_high)))

    amount = int(round(amount / 100) * 100)

    counterpart = relation.counterpart

    internal = counterpart.client_ordinal is not None and counterpart.client_ordinal in sim.clients

    # Человек переводит то, что у него есть: привычная сумма
    # уменьшается до возможной, а не упирается в отказ. Отказом
    # заканчивается только попытка при пустом счёте.
    capacity = state.ledger.payment_capacity(ts)

    if amount > capacity > 0:
        trimmed = int(round(capacity * rng.uniform(0.35, 0.95) / 100) * 100)
        if trimmed >= 500:
            amount = trimmed

    sources = state.ledger.payment_sources(ts, amount)

    if not sources:

        outcome = rng.weighted(settings.transfer_shortfall)

        if outcome == "topup_from_other_bank":

            hidden = state.ledger.accounts[state.ledger.other_bank_id]

            if hidden.balance >= amount:

                account = state.primary_card_account(ts)

                if account is not None:
                    _emit_money(
                        state, ts - timedelta(minutes=3), "transfer_in", account.account_id,
                        amount, "credit", state.ledger.other_bank_id,
                        {
                            # Деньги пришли из другого банка: этот
                            # банк их только зачислил.
                            "channel": "system",
                            "counterparty": "Own account",
                            "reason": "topup_before_transfer",
                            "mcc": MCC_TRANSFER,
                            "merchant_country": "KZ",
                        },
                    )
                    sources = state.ledger.payment_sources(ts, amount)

        if not sources and outcome == "reduce_amount":
            reduced = int(amount * rng.uniform(*settings.reduce_amount_factor))
            reduced = max(500, int(round(reduced / 100) * 100))
            sources = state.ledger.payment_sources(ts, reduced)
            if sources:
                amount = reduced

        if not sources:

            if outcome in ("client_cancels", "topup_from_other_bank", "reduce_amount"):
                # Попытка не удалась и до банка не дошла.
                state.note(ts, "transfer_intent", "cancelled", {"amount": amount})
                return

            if not state.may_decline(ts):
                state.note(ts, "transfer_intent", "abandoned", {"amount": amount})
                return

            _decline(
                state, ts, "p2p_out" if internal else "transfer_out", None, amount, "debit",
                {
                    "channel": CHANNEL_REMOTE,
                    "counterparty": counterpart.masked_name,
                    "mcc": MCC_TRANSFER,
                    "merchant_country": "KZ",
                    "reason": "transfer",
                },
                "insufficient_funds",
            )
            _touch_client(state, ts)
            return

    account = sources[0]

    transfer_id = _transfer_id(state.client_id, ts, payload["index"])

    body = {
        # Плановый перевод сессии не принадлежит: записи
        # app_operation и session_id у него нет, поэтому и канал
        # не app.
        "channel": CHANNEL_REMOTE,
        "counterparty": counterpart.masked_name,
        "mcc": MCC_TRANSFER,
        "merchant_country": "KZ",
        "reason": "transfer",
        "card_id": state.usable_card(account.account_id, ts).card_id
        if state.usable_card(account.account_id, ts)
        else None,
    }

    if internal:

        other = sim.clients[counterpart.client_ordinal]

        target = other.primary_card_account(ts)

        if target is None:
            internal = False

    # Обе ноги перевода или ни одной. Зачисление получателю
    # датируется секундой позже, и на самом краю окна оно уже не
    # попадает в выгрузку: у перевода осталась бы одна сторона, а
    # деньги ушли бы в никуда.
    if internal and not in_window(ts + timedelta(seconds=1)):
        internal = False

    if internal:

        _emit_money(
            state, ts, "p2p_out", account.account_id, amount, "debit",
            target.account_id, dict(body, transfer_id=transfer_id),
        )

        # Деньги доходят немедленно и влияют на решения получателя.
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

    else:

        destination = (
            state.ledger.other_bank_id
            if relation.relation_type == "own_account_other_bank"
            else f"external:{counterpart.counterpart_id}"
        )

        _emit_money(
            state, ts, "transfer_out", account.account_id, amount, "debit",
            destination, dict(body, transfer_id=transfer_id),
        )

    _touch_client(state, ts)

    contract = state.contracts.get(account.contract_id) if account.contract_id else None

    if contract is not None:

        fee = card_rules.transfer_fee(contract.terms, amount, state.monthly_transfer)

        state.monthly_transfer += amount

        if fee > 0 and state.ledger.can_debit(account.account_id, fee):
            _emit_money(
                state, ts + timedelta(seconds=4), "fee_charge", account.account_id, fee, "debit",
                COUNTERPART_BANK,
                {
                    "channel": "system",
                    "contract_id": contract.contract_id,
                    "reason": "transfer_fee",
                    "accrual_period": ts.strftime("%Y-%m"),
                    "merchant_country": "KZ",
                },
            )


def graph_counterpart_name(state: ClientState) -> str:
    from .world.relationships import masked_name

    return masked_name(state.client_id)


_HANDLERS: dict = {}


def _on_inbound_transfer(sim, state: ClientState, ts: datetime, payload: dict) -> None:
    """
    Деньги пришли клиенту со стороны: помощь родных, возврат
    долга, расчёт постоянного контрагента.
    """

    relation = payload["relation"]

    account = state.primary_card_account(ts)

    if account is None:
        return

    settings = params_module.active().relationships

    rng = event_rng(
        NS_INBOUND, state.ordinal, ts.toordinal(), payload["index"], COMPONENT_CONTENT
    )

    low, high = settings.inbound_amount_share_of_income

    amount = int(state.persona.true_income * rng.uniform(low, high))

    amount = max(1_000, int(round(amount / 100) * 100))

    _emit_money(
        state,
        ts,
        "transfer_in",
        account.account_id,
        amount,
        "credit",
        f"external:{relation.counterpart.counterpart_id}",
        {
            "channel": "system",
            "counterparty": relation.counterpart.masked_name,
            "mcc": MCC_TRANSFER,
            "merchant_country": "KZ",
            "reason": "inbound",
        },
    )


def _on_cash_deposit(sim, state: ClientState, ts: datetime, payload: dict) -> None:
    """
    Наличные возвращаются на счёт: скрытый мир и наблюдаемый
    связаны проводкой, а не появлением денег из ниоткуда.
    """

    rng = keyed_rng(NS_LEDGER, state.ordinal, ts.toordinal(), 17)

    cash = state.ledger.accounts[state.ledger.cash_id]

    amount = int(round(cash.balance * rng.uniform(0.35, 0.9) / 1_000) * 1_000)

    if amount < 1_000:
        return

    account = state.primary_card_account(ts)

    if account is None:
        return

    card = state.usable_card(account.account_id, ts)

    _emit_money(
        state, ts, "cash_deposit", account.account_id, amount, "credit",
        state.ledger.cash_id,
        {
            "channel": "atm",
            "card_id": card.card_id if card else None,
            "mcc": MCC_CASH,
            "merchant_country": "KZ",
            "is_online": False,
            "reason": "cash_deposit",
        },
    )

    _touch_client(state, ts)


_HANDLERS.update(
    {
        "cash_deposit": _on_cash_deposit,
        "registration": _on_registration,
        "income": _on_income,
        "bill": _on_bill,
        "subscription": _on_subscription,
        "purchase": _on_purchase,
        "refund": _on_refund,
        "cash_withdrawal": _on_cash,
        "transfer": _on_transfer,
        "inbound_transfer": _on_inbound_transfer,
    }
)


# Обработчики остальных доменов регистрируются в своих модулях.
from .engine_app import unblock_card  # noqa: E402,F401
from .engine_credit import close_loan, payment_plan as _payment_plan, repay_loan  # noqa: E402,F401
from .engine_products import _emit_case  # noqa: E402,F401
from .engine_month import (  # noqa: E402
    _update_profile,
    finish as _finish,
    month_end as _month_end,
)


__all__ = ["run_community"]
