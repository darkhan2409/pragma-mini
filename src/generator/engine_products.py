from __future__ import annotations

from datetime import datetime, timedelta

from . import params as params_module
from .behaviour import adoption as adoption_module
from .behaviour import support as support_module
from .behaviour import fraud as fraud_behaviour
from . import config
from .engine import _HANDLERS, _decline, _emit_money, _touch_client
from .engine_app import unblock_card
from .engine_credit import emit_product_closed
from .finance import cards as card_rules
from .finance import deposits as deposit_rules
from .finance import loans as loan_rules
from .finance.entities import (
    CARD_ACTIVE,
    CONTRACT_CLOSED,
    Application,
    Card,
    LoanState,
)
from .finance.ledger import COUNTERPART_BANK
from .life import calendar as cal
from .rng import (
    COMPONENT_CONTENT,
    COMPONENT_OUTCOME,
    NS_ADOPTION,
    NS_CARD_BLOCK,
    NS_FRAUD,
    NS_FRAUD_MATERIAL,
    NS_PROFILE,
    NS_SUPPORT,
    event_rng,
    keyed_rng,
    stable_hash,
)
from .simulate import ClientState, _application_id
from .world import geography, merchants as merchant_catalog, products as product_catalog
from .world.dictionaries import FUNNEL_SCREENS, MCC_CASH, MCC_TRANSFER
from .world.relationships import masked_name


# ============================================================
# ПРОДУКТЫ, АНТИФРОД И ПРОФИЛЬ
# ============================================================
#
# Распространение продукта:
#
#   eligibility -> предложение -> заявка -> решение
#   -> активация -> первое использование
#
# Версия продукта фиксируется на дате подписания договора и
# живёт с ним. Новая версия касается действующего договора
# только по applies_to, а переход на другой продукт это
# отдельное событие миграции.
# ============================================================


CREDIT_FAMILIES = ("cash_loan", "credit_card", "refinance", "installment")

# Семейства, у которых договор это кредит с графиком.
LOAN_FAMILIES = ("cash_loan", "refinance", "installment")


def _age_limit(persona, ts, income, dpd, loans) -> bool:
    rules = params_module.active().products.bank_rules
    age = persona.age_at(ts)
    return age < int(rules.get("min_age", 18)) or age > int(rules.get("max_age", 79))


def _income_not_confirmed(persona, ts, income, dpd, loans) -> bool:
    threshold = params_module.active().products.low_income_threshold
    return persona.income_type in ("unemployed", "student") or income < threshold


REJECT_CASCADE = (
    ("age_limit", _age_limit),
    ("income_not_confirmed", _income_not_confirmed),
    ("existing_debt", lambda persona, ts, income, dpd, loans: loans >= 2),
)


def _open_loans(state: ClientState) -> tuple:
    return tuple(item for item in state.loans.values() if not item.closed)


def _prospective_payment(state: ClientState, candidate, ts: datetime, amount, term) -> int:
    """
    Во сколько обойдётся клиенту новый договор в месяц.
    """

    settings = params_module.active().products

    family = candidate.view.family

    if amount is None:
        return 0

    if family == "credit_card":
        return int(amount * settings.credit_card_payment_share_of_limit)

    if family not in ("cash_loan", "refinance", "installment"):
        return 0

    terms = candidate.version.terms

    rate = terms.get("rate")

    if rate is None:
        by_term = terms.get("rate_by_term") or {}
        rate = by_term.get(str(term)) or by_term.get(term) or 0.28

    return int(loan_rules.annuity_payment(int(amount), float(rate), int(term or 12)))


def _debt_service_fits(state: ClientState, candidate, ts: datetime, amount, term) -> bool:
    """
    Долговая нагрузка после нового договора.

    Рефинансирование закрывает старые кредиты, поэтому их
    платежи из нагрузки вычитаются.
    """

    settings = params_module.active().products

    ratio = float(settings.bank_rules.get("max_debt_service_ratio", 0.5))

    income = state.bank_income()

    open_loans = _open_loans(state)

    existing = loan_rules.debt_service(open_loans, ts)

    if candidate.view.family == "refinance":
        existing = 0

    return (existing + _prospective_payment(state, candidate, ts, amount, term)) <= ratio * income


def _approval(
    state: ClientState,
    candidate,
    ts: datetime,
    stress: float,
    rng,
    amount=None,
    term=None,
) -> tuple:

    settings = params_module.active().products

    persona = state.persona

    family = candidate.view.family

    # Долговая нагрузка это ЖЁСТКОЕ правило, а не ярлык после
    # отказа. Розыгрыш при этом не тратится: решение
    # детерминировано.
    if family in CREDIT_FAMILIES and not _debt_service_fits(state, candidate, ts, amount, term):
        return False, "debt_service_ratio"

    if family == "refinance" and state.primary_card_account(ts) is None:
        return False, "documents_invalid"

    base = settings.approval_base.get(family, 0.9)

    if family in CREDIT_FAMILIES:

        base += settings.approval_income_factor * min(1.0, state.bank_income() / 700_000)
        base += settings.approval_discipline_factor * (persona.trait("financial_discipline", ts) - 0.5)
        base -= settings.approval_stress_penalty * stress

        worst = state.worst_dpd()

        if worst > 0:
            base -= settings.approval_dpd_penalty * min(1.0, worst / 60.0)

        if _open_loans(state):
            base -= settings.approval_existing_loan_penalty

    low, high = settings.approval_bounds

    probability = max(low, min(high, base))

    approved = rng.random() < probability

    if approved:
        return True, None

    open_loans = len(_open_loans(state))

    for reason, rule in REJECT_CASCADE:
        if rule(persona, ts, state.bank_income(), state.worst_dpd(), open_loans):
            return False, reason

    if stress > 0.55:
        return False, "scoring_declined"

    if rng.random() < settings.blacklist_share:
        return False, "blacklist"

    return False, rng.weighted(settings.reject_reason_weights)


def _funnel(state: ClientState, application: Application, ts: datetime, approved: bool,
            reject_reason: str | None, rng) -> datetime:
    """
    Воронка заявки в приложении: просмотр, форма, проверка,
    решение.
    """

    moment = ts

    stages = ["view", "application", "kyc", "approved" if approved else "rejected"]

    for index, stage in enumerate(stages):

        if index:
            moment = moment + timedelta(seconds=int(rng.integers(30, 900)))

        state.emit(
            state.factory.make(
                "app_screen",
                moment,
                {
                    "firebase_screen": FUNNEL_SCREENS[stage],
                    "product_id": application.product_id,
                    "funnel_stage": stage,
                    "reject_reason": reject_reason if stage == "rejected" else None,
                },
            )
        )

    return moment


# Часы, в которые канал подачи вообще работает.
#
# Приложение, сайт, банкомат и терминал доступны круглосуточно;
# отделение, почта и партнёрская касса — нет. Заявка в отделение
# в полночь это не редкий случай, а невозможный.
CHANNEL_HOURS: dict[str, tuple[int, int]] = {
    "app": (0, 24),
    "web": (0, 24),
    "atm": (0, 24),
    "terminal": (0, 24),
    "call_center": (8, 22),
    "partner_pos": (10, 21),
    "branch": (9, 19),
    "micro_office": (9, 19),
    "qazpost": (9, 18),
}


def _channel_moment(ts: datetime, channel: str, rng) -> datetime | None:
    """
    Момент подачи внутри рабочих часов канала.

    Сдвиг только ВПЕРЁД, и это главное правило здесь. Решение о
    заявке принято по состоянию клиента на исходный момент: по
    его остаткам, договорам и просрочке на 22:00. Перенос заявки
    на 10:00 того же дня датировал бы её задним числом — решением,
    принятым по данным, которых в ту минуту ещё не было.

    Канал уже закрылся — заявки сегодня не будет вовсе. Интерес к
    продукту разыгрывается каждый день заново, поэтому клиент
    просто дойдёт до отделения в другой раз.
    """

    low, high = CHANNEL_HOURS.get(channel, (9, 19))

    if (low, high) == (0, 24) or low <= ts.hour < high:
        return ts

    if ts.hour >= high:
        return None

    return ts.replace(
        hour=low,
        minute=int(rng.integers(0, 60)),
        second=int(rng.integers(0, 60)),
    )


def _application_payload(application: Application, **extra) -> dict:

    body = {
        "application_id": application.application_id,
        "product_id": application.product_id,
        "offer_id": application.offer_id,
        "channel": application.channel,
        "requested_amount": application.requested_amount,
        "requested_term": application.requested_term,
        "decision": None,
        "reject_reason": None,
        "approved_amount": None,
        "approved_term": None,
    }

    body.update(extra)

    return body


def _on_adoption(sim, state: ClientState, ts: datetime, payload: dict) -> None:

    settings = params_module.active().products

    persona = state.persona

    stress = payload["stress"]
    app_adopted = payload["app"]

    rng = event_rng(NS_ADOPTION, state.ordinal, ts.toordinal(), 0, COMPONENT_CONTENT)

    _migrate_products(sim, state, ts, rng, voluntary=True)

    pool = adoption_module.candidates(
        persona,
        ts,
        state.held_codes(ts),
        state.held_counts(ts),
        state.assets(),
        app_adopted,
        state.open_loan_count(),
        len(state.open_contracts(ts)),
        state.income_months_at(ts),
        stress,
    )

    if not pool:
        return

    recent_offer = None

    for offer in reversed(state.offers):
        if (ts - offer.created_at).days <= 14:
            recent_offer = offer
            break

    candidate = None

    if recent_offer is not None:
        candidate = next((item for item in pool if item.view.family == recent_offer.product_family), None)

    from_offer = candidate is not None

    if candidate is None:
        candidate = adoption_module.pick(pool, rng)

    if candidate is None:
        return

    probability = adoption_module.application_probability(
        persona, candidate, ts, from_offer, stress,
        total_weight=sum(item.weight for item in pool),
    )

    if rng.random() >= probability:
        return

    # Заявки не подаются каждый день подряд: после отказа или
    # оформления клиент выдерживает паузу.
    cooldown = settings.application_cooldown_days

    recent = max(
        (item.submitted_at for item in state.applications.values()),
        default=None,
    )

    if recent is not None and (ts - recent).days < cooldown:
        return

    # --- заявка ---

    channels = tuple(candidate.version.channels) or ("branch",)

    channel = str(rng.choice(list(channels)))

    if channel == "app" and not app_adopted:

        # Приложения у клиента нет. Отделение сюда не подставляется:
        # продукт может продаваться только онлайн, и заявка в
        # отделении по нему невозможна. Берётся другой канал ИЗ
        # РАЗРЕШЁННЫХ продуктом, а если такого нет — заявки нет.
        instead = [name for name in channels if name != "app"]

        if not instead:
            return

        channel = str(rng.choice(instead))

    # Дальше момент подачи принадлежит каналу, а не плану дня.
    moment = _channel_moment(ts, channel, rng)

    if moment is None:
        return

    ts = moment

    amount, term = sim._contract_terms(state, candidate.view, ts, rng)

    # Класть нечего: заявки на вклад без денег не бывает.
    if amount is None and candidate.view.family in ("deposit", "deposit_certificate"):
        return

    # Сумма выбрана только сейчас, поэтому порог договора
    # проверяется здесь, а не при отборе кандидатов.
    minimum = int((candidate.version.eligibility or {}).get("min_amount", 0))

    if minimum and (amount is None or amount < minimum):
        return

    application = Application(
        application_id=_application_id(state.client_id, ts, len(state.applications)),
        client_id=state.client_id,
        product_id=candidate.view.record.product_id,
        product_code=candidate.view.code,
        product_family=candidate.view.family,
        product_version=candidate.version.product_version,
        channel=channel,
        submitted_at=ts,
        requested_amount=amount,
        requested_term=term,
        offer_id=recent_offer.offer_id if from_offer and recent_offer else None,
    )

    state.applications[application.application_id] = application

    state.emit(
        state.factory.make(
            "application_submitted",
            ts,
            _application_payload(application),
        )
    )

    _touch_client(state, ts)

    approved, reason = _approval(state, candidate, ts, stress, rng, amount, term)

    low, high = settings.decision_delay_seconds.get(channel, (60, 3600))

    decision_ts = ts + timedelta(seconds=int(rng.integers(low, high)))

    if channel == "app" and app_adopted:
        decision_ts = _funnel(state, application, ts, approved, reason, rng)

    if decision_ts >= config.HISTORY_END:
        return

    application.decision = "approved" if approved else "rejected"
    application.decided_at = decision_ts
    application.reject_reason = reason
    application.approved_amount = amount if approved else None
    application.approved_term = term if approved else None

    state.emit(
        state.factory.make(
            "application_decision",
            decision_ts,
            _application_payload(
                application,
                decision=application.decision,
                reject_reason=reason,
                approved_amount=application.approved_amount,
                approved_term=application.approved_term,
            ),
        )
    )

    if not approved:
        return

    open_ts = decision_ts + timedelta(seconds=int(rng.integers(*settings.disbursement_delay_seconds)))

    if open_ts >= config.HISTORY_END:
        return

    contract = sim._open_contract(
        state,
        candidate.view,
        open_ts,
        amount,
        term,
        offer_id=application.offer_id,
        application_id=application.application_id,
    )

    _activate_product(sim, state, contract, candidate.view, open_ts, rng)


def _close_refinanced(state: ClientState, ts: datetime, new_contract_id: str) -> None:
    """
    Закрывает кредиты, ради которых бралось рефинансирование.
    """

    from .engine_credit import close_loan

    targets = [
        item
        for item in state.loans.values()
        if not item.closed and item.contract_id != new_contract_id
    ]

    for position, loan in enumerate(targets):
        close_loan(
            state,
            ts + timedelta(minutes=10 + position),
            loan,
            early=True,
            reason="refinanced",
        )


def _gather_on_card(state: ClientState, ts: datetime, amount: int) -> bool:
    """
    Клиент собирает сумму вклада на карте: недостающее приходит
    наличными через банкомат или переводом из другого банка.
    Случайности здесь нет — решение открыть вклад уже принято с
    оглядкой на все деньги клиента.
    """

    account = state.primary_card_account(ts)

    if account is None:
        return False

    shortfall = max(0, amount - state.ledger.available_at(account.account_id, ts))

    if shortfall <= 0:
        return True

    sources = state.ledger.hidden_sources(shortfall)

    if not sources:
        return False

    hidden = sources[0]

    from_cash = hidden.account_id == state.ledger.cash_id

    _emit_money(
        state,
        ts,
        "cash_deposit" if from_cash else "transfer_in",
        account.account_id,
        shortfall,
        "credit",
        hidden.account_id,
        {
            # Наличные вносит банкомат, перевод приходит из
            # другого банка. Сессии приложения тут нет ни в том,
            # ни в другом случае.
            "channel": "atm" if from_cash else "system",
            "counterparty": "Own account",
            "reason": "cash_deposit" if from_cash else "transfer",
            "mcc": MCC_CASH if from_cash else MCC_TRANSFER,
            "merchant_country": "KZ",
        },
    )

    return True


def _cancel_unfunded(state: ClientState, ts: datetime, contract) -> None:
    """
    Договор вклада, под который не нашлось денег, аннулируется
    сразу. Открытый вклад с нулевым остатком не зарабатывает и не
    заканчивается, а в ленте выглядел бы живым продуктом.
    """

    moment = ts + timedelta(minutes=3)

    contract.status = CONTRACT_CLOSED
    contract.closed_at = moment

    account = state.ledger.get(contract.account_id) if contract.account_id else None

    if account is not None:
        account.closed_at = moment

    emit_product_closed(state, moment, contract, "not_funded")


def _activate_product(
    sim, state: ClientState, contract, view, ts: datetime, rng,
    previous: LoanState | None = None,
) -> None:
    """
    Первое действие по новому договору: выдача кредита,
    зачисление на депозит, оплата страховки.

    previous — кредит, долг которого принял этот договор. Денег
    по такому договору не выдаётся, а просрочка прежнего графика
    переходит в новый.
    """

    family = contract.product_family

    if family in LOAN_FAMILIES:

        if not _can_activate(state, family, contract.amount_or_limit, ts):
            return

        account = state.primary_card_account(ts)

        if family != "installment" and previous is None:
            _emit_money(
                state,
                ts + timedelta(minutes=5),
                "loan_disbursement",
                account.account_id,
                int(contract.amount_or_limit),
                "credit",
                f"loan:{contract.contract_id}",
                {
                    "channel": "system",
                    "contract_id": contract.contract_id,
                    "reason": "disbursement",
                    "merchant_country": "KZ",
                },
            )

        # Рефинансирование не добавляет ещё один долг к прежним:
        # выданные деньги гасят их и закрывают договоры.
        if family == "refinance":
            _close_refinanced(state, ts, contract.contract_id)

        loan = loan_rules.open_loan(
            contract.contract_id,
            int(contract.amount_or_limit),
            float(contract.rate or 0.28),
            int(contract.term or 12),
            ts,
            autopay=rng.random() < params_module.active().products.autopay_share,
        )

        if previous is not None:
            loan_rules.carry_arrears(loan, previous)

        state.loans[contract.contract_id] = loan

        # Ближайший взнос — первый ещё не наступивший: перенесённые
        # просроченные взносы стоят в графике раньше него.
        upcoming = next((item for item in loan.schedule if item.status == "scheduled"), None)

        state.emit(
            state.factory.make(
                "schedule_created",
                ts + timedelta(minutes=6),
                {
                    "contract_id": contract.contract_id,
                    "installment_no": len(loan.schedule),
                    "amount_due": upcoming.amount if upcoming else None,
                    "amount_paid": None,
                    "principal_outstanding": loan.principal_outstanding,
                    "days_past_due": loan.dpd,
                    "due_date": upcoming.due_date.date().isoformat() if upcoming else None,
                    "reason": "annuity",
                },
            )
        )

        return

    if family in ("deposit", "deposit_certificate"):

        amount = int(contract.amount_or_limit or 0)

        def sources_for(value: int) -> list:
            return [
                item
                for item in state.ledger.payment_sources(ts, value)
                if item.account_id != contract.account_id
            ]

        sources = sources_for(amount) if amount > 0 else []

        # На одном счёте суммы нет: клиент собирает её на карте —
        # наличными или переводом из другого банка.
        if amount > 0 and not sources and _gather_on_card(state, ts + timedelta(minutes=1), amount):
            sources = sources_for(amount)

        if not sources or amount <= 0:
            # Вклад без денег не живёт. Раньше договор оставался
            # открытым без остатка, процентов и срока; теперь он
            # аннулируется сразу, и это видно в ленте.
            _cancel_unfunded(state, ts, contract)
            return

        from .engine_app import _own_transfer

        if not _own_transfer(
            state,
            ts + timedelta(minutes=2),
            "deposit_topup",
            sources[0].account_id,
            contract.account_id,
            amount,
            contract.contract_id,
            "initial_deposit",
        ):
            # Денег на вклад так и не положили: договор без
            # остатка не живёт, и аннулируется он здесь же.
            _cancel_unfunded(state, ts, contract)
            return

        terms = contract.terms

        state.deposits[contract.contract_id] = deposit_rules.open_deposit(
            contract_id=contract.contract_id,
            account_id=contract.account_id,
            amount=amount,
            rate=float(contract.rate or terms.get("rate") or 0.14),
            opened_at=ts,
            term_months=int(contract.term or 12),
            topup=bool(terms.get("topup", False)),
            withdrawal=bool(terms.get("withdrawal", False)),
            capitalisation=str(terms.get("capitalisation", "daily")),
        )

        return

    if family == "credit_card":

        # Карта рассрочки: покупки делятся на части, наличные
        # копят проценты. Без этого карта была бесплатным
        # кредитом без конца.
        if contract.account_id is not None:
            state.card_credits[contract.contract_id] = card_rules.open_credit(
                contract, contract.terms or {}
            )

        return

    if family == "insurance":

        amount = int(contract.amount_or_limit or 0)

        sources = state.ledger.payment_sources(ts, amount)

        if not sources or amount <= 0:
            return

        _emit_money(
            state,
            ts + timedelta(minutes=3),
            "purchase",
            sources[0].account_id,
            amount,
            "debit",
            COUNTERPART_BANK,
            {
                # Регулярная премия списывается банком по договору.
                "channel": "system",
                "contract_id": contract.contract_id,
                "reason": "insurance_premium",
                "mcc": "6300",
                "merchant_country": "KZ",
                "is_online": True,
                "is_subscription": False,
            },
        )


def _can_activate(state: ClientState, family: str, amount, ts: datetime) -> bool:
    """
    Можно ли завести кредит по договору: деньги выдаются и
    списываются через основную карту, без неё и без суммы
    кредита нет.
    """

    if family not in LOAN_FAMILIES:
        return True

    return amount is not None and state.primary_card_account(ts) is not None


def _on_tariff_check(sim, state: ClientState, ts: datetime, payload: dict) -> None:

    _apply_new_versions(state, ts)

    # Принудительная миграция это решение банка, как и смена
    # тарифа: от активности клиента она не зависит.
    rng = event_rng(NS_ADOPTION, state.ordinal, ts.toordinal(), 1, COMPONENT_CONTENT)

    _migrate_products(sim, state, ts, rng, voluntary=False)


def _apply_new_versions(state: ClientState, ts: datetime) -> None:
    """
    Новая версия продукта касается действующего договора только
    по applies_to и только после уведомления.
    """

    catalog = product_catalog.catalog()

    for contract in list(state.contracts.values()):

        if not contract.is_open_at(ts) or not catalog.has(contract.product_code):
            continue

        view = catalog.view(contract.product_code)

        for version in view.later_versions(contract.opened_at):

            if version.applies_to != "existing_from_date":
                continue

            if (version.product_version, version.tariff_version) <= (
                contract.product_version, contract.tariff_version
            ):
                continue

            notice = version.notice_days or params_module.active().products.adoption.get(
                "notice_days_default", 30
            )

            effective = view.version_start(version) + timedelta(days=int(notice))

            if ts.date() != effective.date():
                continue

            terms_changed = version.product_version != contract.product_version

            event_type = "contract_terms_changed" if terms_changed else "product_repriced"

            contract.product_version = version.product_version
            contract.tariff_version = version.tariff_version
            contract.terms = dict(version.terms)

            state.pending_notice = True

            state.emit(
                state.factory.make(
                    event_type,
                    ts.replace(hour=0, minute=5),
                    {
                        "product_id": contract.product_id,
                        "contract_id": contract.contract_id,
                        "account_id": contract.account_id,
                        "card_id": contract.card_id,
                        "amount_or_limit": contract.amount_or_limit,
                        "term": contract.term,
                        "rate": contract.rate,
                        "reason": "tariff_update" if not terms_changed else "terms_update",
                    },
                )
            )


# ============================================================
# МОШЕННИЧЕСТВО
# ============================================================


def _fraud_purchase(state: ClientState, account, card, ts, amount, episode, step, rng):
    """
    Мошенническая покупка выглядит как обычная: реальная точка,
    её MCC и её имя. Постоянная метка вроде UNKNOWN MERCHANT
    была бы готовым признаком для модели.
    """

    from .behaviour import merchants as merchant_choice

    settings = params_module.active().fraud

    if card is None:
        return None, "card"

    pick_rng = event_rng(
        NS_FRAUD_MATERIAL, state.ordinal, ts.toordinal(), int(step.kind == "strike"), COMPONENT_CONTENT
    )

    category = str(pick_rng.weighted(settings.material_categories))

    country = _foreign_country(state, pick_rng) if step.foreign else None

    choice = merchant_choice.choose_outlet(
        state.persona,
        state.habits,
        category,
        ts,
        pick_rng,
        online_hint=step.online,
        foreign_country=country,
    )

    if choice is None:
        return None, "card"

    outlet = choice.outlet

    body = {
        "channel": choice.channel,
        "card_id": card.card_id,
        **merchant_catalog.payload_fields(outlet),
        "is_online": outlet.is_online,
        "is_subscription": False,
        "reason": "purchase",
    }

    # Денег на счёте не хватило. Это ОТКАЗ, а не отсутствие
    # попытки: банк видит неудачную авторизацию и реагирует на
    # неё так же. Раньше эпизод на этом месте исчезал целиком —
    # ни срабатывания антифрода, ни обращения клиента.
    if not state.ledger.payment_sources(ts, amount):
        return (
            _decline(
                state, ts, "purchase", account.account_id, amount, "debit",
                body, "insufficient_funds",
            ),
            "card",
        )

    event = _emit_money(
        state,
        ts,
        "purchase",
        account.account_id,
        amount,
        "debit",
        merchant_catalog.counterpart(outlet),
        body,
    )

    return event, "card"


def _fraud_transfer(state: ClientState, account, ts, amount, episode, step, rng):
    """
    Подозрительный перевод и социальная инженерия это перевод,
    а не покупка. Захват доступа вдобавок начинается со входа
    с нового устройства.
    """

    # Объект проверки — сама операция. Отдельный объект для
    # захвата доступа называл бы вид эпизода прямо в поле.
    subject = "transfer"

    if episode.kind == "account_takeover" and step.kind == "strike":
        _emit_takeover_login(state, ts - timedelta(minutes=4))

    counterpart = f"cp_fraud_{stable_hash(state.client_id, ts.toordinal()) % 10 ** 8:08d}"

    body = {
        # Мошенническая операция сессии клиента не принадлежит:
        # ни app_operation, ни session_id у неё нет.
        "channel": "ecom",
        # Имя в том же виде, что у любого другого получателя:
        # особый формат выдавал бы мошенническую ногу сам по себе.
        "counterparty": masked_name(counterpart),
        "mcc": MCC_TRANSFER,
        "merchant_country": "KZ",
        "reason": "transfer",
    }

    # Социальную инженерию переводит сам клиент, захват доступа
    # делают чужие руки.
    # Денег не хватило — это отказ, а не отсутствие попытки.
    if not state.ledger.payment_sources(ts, amount):
        return (
            _decline(
                state, ts, "transfer_out", account.account_id, amount, "debit",
                body, "insufficient_funds",
            ),
            subject,
        )

    event = _emit_money(
        state,
        ts,
        "transfer_out",
        account.account_id,
        amount,
        "debit",
        f"external:{counterpart}",
        body,
    )

    return event, subject


def _emit_takeover_login(state: ClientState, ts: datetime) -> None:
    """
    Вход с нового устройства перед захватом доступа.
    """

    if ts < config.HISTORY_START or ts >= config.HISTORY_END:
        return

    if state.app_adopted_at is None or ts < state.app_adopted_at:
        return

    session_id = f"ses_{stable_hash('takeover', state.client_id, ts.toordinal()) % 10 ** 12:012d}"

    state.emit(
        state.factory.make(
            "app_operation",
            ts,
            {
                "domain": "auth",
                "operation": "login",
                "status": "success",
                "amount": None,
                "error_code": None,
                "device_new": True,
                # Вход открывает сессию, как любой другой вход:
                # запись без session_id выделялась бы из ленты.
                "session_id": session_id,
                "contract_id": None,
            },
        )
    )


def _foreign_country(state: ClientState, rng) -> str:
    return str(rng.weighted(params_module.active().geography.foreign_countries))


def _on_fraud_step(sim, state: ClientState, ts: datetime, payload: dict) -> None:

    settings = params_module.active().fraud

    episode = payload["episode"]
    step = payload["step"]

    rng = event_rng(NS_FRAUD, state.ordinal, ts.toordinal(), payload["position"], COMPONENT_CONTENT)

    account = state.primary_card_account(ts)

    if account is None:
        return

    card = state.usable_card(account.account_id, ts)

    if step.kind == "probe":
        amount = int(step.amount_hint)
    else:
        limit = max(50_000, account.balance + account.credit_limit)
        amount = int(max(5_000, limit * step.amount_hint))

    amount = int(round(amount / 100) * 100)

    as_transfer = episode.kind in settings.transfer_kinds

    if as_transfer:
        event, subject = _fraud_transfer(state, account, ts, amount, episode, step, rng)
    else:
        event, subject = _fraud_purchase(state, account, card, ts, amount, episode, step, rng)

    if event is None:
        return

    # Мошенническая операция не попадает в список кандидатов на
    # обычный возврат: её оспаривают через chargeback, и два
    # возврата по одной операции превысили бы её сумму.

    if step.kind != "strike" and episode.kind != "false_positive":
        return

    if not episode.detected:
        return

    alert_ts = ts + timedelta(minutes=episode.detect_delay_minutes)

    if alert_ts >= config.HISTORY_END:
        return

    # Тревожность считается от доли месячного дохода, а не от
    # подсказки шага: у пробной покупки подсказка это сумма в
    # тенге, и доля всегда упиралась бы в единицу. Доход тот,
    # что знает банк.
    share = amount / state.bank_income()

    alert_rng = event_rng(NS_FRAUD, state.ordinal, ts.toordinal(), payload["position"], COMPONENT_OUTCOME)

    state.emit(
        state.factory.make(
            "fraud_alert",
            alert_ts,
            {
                "subject": subject,
                "card_id": card.card_id if card and subject == "card" else None,
                "account_id": account.account_id,
                "score_band": fraud_behaviour.score_band(step.foreign, share, alert_rng),
                "rule_code": fraud_behaviour.rule_code(episode.kind, alert_rng),
            },
        )
    )

    state.fraud_alert_at = alert_ts

    decision = episode.decision

    decision_ts = alert_ts + timedelta(minutes=int(rng.integers(1, 60)))

    state.emit(
        state.factory.make(
            "fraud_decision",
            decision_ts,
            {
                "subject": subject,
                "card_id": card.card_id if card and subject == "card" else None,
                "account_id": account.account_id,
                "decision": decision,
                "resolution": episode.client_response if episode.client_response != "no_response" else None,
            },
        )
    )

    # Карта скомпрометирована, если операцию делали чужие руки
    # и клиент её своей не признал. Такую карту размораживать
    # нельзя: её место занимает перевыпущенная.
    compromised = episode.kind != "false_positive" and episode.client_response != "confirmed_by_client"

    # Блокировать нечего, если карта не при чём: подозрительный
    # перевод банк останавливает без карты.
    #
    # Второй удар того же эпизода приходит раньше, чем банк
    # успевает решить по первому, и карта к моменту решения уже
    # заблокирована. Блокировать её снова банк не станет: одна
    # карта — одна блокировка, и перевыпуск тоже один.
    blocked = (
        decision == "block"
        and card is not None
        and not card.is_blocked_at(decision_ts)
    )

    if blocked:

        card_rules.block(card, decision_ts, "fraud_suspicion", permanent=compromised)

        state.emit(
            state.factory.make(
                "card_blocked",
                decision_ts + timedelta(seconds=10),
                dict(
                    state.card_facts(card),
                    # В момент блокировки банк знает только
                    # подозрение: скомпрометирована ли карта, решит
                    # разбор, и его исход выгрузка покажет позже.
                    reason="fraud_suspicion",
                ),
            )
        )

    # --- реакция клиента ---
    #
    # Обращение и оспаривание следуют из того, что клиент увидел
    # у себя чужую операцию. Решение банка тут ни при чём: карту
    # он мог и не заблокировать, а деньги всё равно ушли.
    #
    # Ложное срабатывание это своя же покупка: жаловаться не на
    # что, пока банк не заблокировал карту.
    #
    # Эпизод из нескольких ударов даёт одно обращение и один
    # возврат, а не по одному на каждый удар.

    episode_key = (episode.kind, episode.start)

    if (
        episode.opens_case
        and episode_key not in state.fraud_disputes
        and (episode.kind != "false_positive" or blocked)
    ):

        state.fraud_disputes.add(episode_key)

        case = support_module.open_case(
            state.persona, "fraud_alert" if episode.kind != "false_positive" else "card_blocked",
            decision_ts + timedelta(hours=int(rng.integers(1, 20))),
            len(state.cases),
        )

        _emit_case(state, case)

        # Возврат по оспариванию бывает только там, где спорят
        # с торговой точкой. Перевод, сделанный руками клиента
        # под давлением, так не возвращают.
        #
        # И возвращают только то, что действительно списали:
        # у отклонённой операции денег не забирали, возвращать
        # нечего.
        if (
            episode.chargeback
            and episode.kind in settings.chargeback_kinds
            and event.payload.get("status") == "approved"
        ):

            back_ts = case.resolved_at + timedelta(days=int(rng.integers(*settings.chargeback_delay_days)))

            if back_ts < config.HISTORY_END:
                _emit_money(
                    state, back_ts, "chargeback", account.account_id, amount, "credit",
                    f"merchant:{event.payload.get('merchant_id')}"
                    if event.payload.get("merchant_id")
                    else "external:merchant",
                    {
                        "channel": "system",
                        "card_id": event.payload.get("card_id"),
                        **{
                            name: event.payload.get(name)
                            for name in merchant_catalog.MERCHANT_PAYLOAD_FIELDS
                        },
                        "reason": "dispute_resolved",
                    },
                )

    # --- судьба заблокированной карты ---

    if not blocked:
        return

    # Скомпрометированную карту не размораживают ни при каких
    # условиях. Обычно банк выдаёт новую, а прежняя остаётся
    # закрытой; если клиент новой не захотел, карта так и
    # остаётся заблокированной навсегда.
    if compromised:

        if episode.reissue:

            reissue_ts = decision_ts + timedelta(days=int(rng.integers(*settings.reissue_delay_days)))

            if reissue_ts < config.HISTORY_END:
                _reissue_card(state, card, reissue_ts)

        return

    unblock_ts = decision_ts + timedelta(hours=int(rng.integers(*settings.unblock_delay_hours)))

    if unblock_ts < config.HISTORY_END:
        unblock_card(state, unblock_ts, card, "confirmed_by_client")


def _reissue_card(state: ClientState, card, ts: datetime, reason: str = "fraud_reissue") -> None:

    contract = state.contracts.get(card.contract_id)

    card.status = "closed"
    card.closed_at = ts

    fresh = Card(
        card_id=f"crd_{stable_hash('reissue', card.card_id, ts.toordinal()) % 10 ** 12:012d}",
        account_id=card.account_id,
        client_id=card.client_id,
        contract_id=card.contract_id,
        product_code=card.product_code,
        issued_at=ts,
        activated_at=ts + timedelta(days=1),
        status=CARD_ACTIVE,
        reissued_from=card.card_id,
        expires_at=cal.add_months(
            ts, 12 * int(params_module.active().products.card_expiry_years)
        ),
    )

    state.cards[fresh.card_id] = fresh

    if contract is not None:
        contract.card_id = fresh.card_id

    state.emit(
        state.factory.make(
            "card_reissued",
            ts,
            dict(state.card_facts(card, fresh.card_id), reason=reason),
        )
    )


def _emit_case(state: ClientState, case) -> None:

    state.cases.append(case)

    body = {
        "case_id": case.case_id,
        "channel": case.channel,
        "topic": case.topic,
        "status": "open",
        "resolution": None,
    }

    state.emit(
        state.factory.make(
            "case_opened",
            case.opened_at,
            body,
        )
    )

    if case.updated_at is not None and case.updated_at < config.HISTORY_END:
        state.emit(
            state.factory.make(
                "case_updated",
                case.updated_at,
                dict(body, status="in_progress"),
            )
        )

    if case.resolved_at < config.HISTORY_END:
        state.emit(
            state.factory.make(
                "case_resolved",
                case.resolved_at,
                dict(body, status="resolved", resolution=case.resolution),
            )
        )


# ============================================================
# ПРОФИЛЬ
# ============================================================


# «Стереть поле» это НЕ «ничего не менять». Сторож нужен, потому
# что None означает «событие этого поля не касается»: без него
# потеря работы не могла убрать ни отрасль, ни день выплаты.
CLEAR = object()


def _on_profile_change(sim, state: ClientState, ts: datetime, payload: dict) -> None:

    event = payload["event"]

    fields = event.changes_profile

    if not fields:
        return

    for name in fields:

        old = state.profile_values.get(name)

        new = _new_profile_value(state, name, event)

        if new is None:
            continue

        if new is CLEAR:
            if old is None:
                continue
            new = None
        elif str(new) == str(old):
            continue

        state.profile_values[name] = new

        state.emit(
            state.factory.make(
                "profile_change",
                ts,
                {
                    "field_name": name,
                    "old_value": None if old is None else str(old),
                    "new_value": None if new is None else str(new),
                    "change_source": "client" if event.kind in ("move", "wedding", "divorce") else "application",
                    "confirmed": event.confirmed,
                },
            )
        )


def _new_profile_value(state: ClientState, name: str, event):

    payload = event.payload

    # Потеря работы меняет анкету только там, где она оборвала
    # зарплату: у кого зарплатного потока не было, деньги идут
    # прежние, и «безработный» в анкете противоречил бы ленте.
    if event.kind == "job_loss" and not any(
        item.kind == "salary" and item.valid_to == event.ts
        for item in state.income_streams
    ):
        return None

    if name == "region":
        return payload.get("region")

    if name == "city":

        target = payload.get("settlement")

        if not target:
            return None

        # Переезд в сельскую местность города клиенту не даёт:
        # корзина области это не населённый пункт, и прежний
        # город при этом перестаёт быть верным.
        return geography.by_name(target).public_name or CLEAR

    if name == "children":
        # Банк узнаёт о рождениях со своими задержками, и порядок
        # сообщений может не совпасть с порядком рождений. Каждое
        # сообщение добавляет одного ребёнка к тому, что уже в
        # анкете, а не ставит итоговое число: иначе анкета
        # прыгала бы N -> N+2 -> N+1.
        current = state.profile_values.get("children")
        return int(state.persona.children if current is None else current) + 1

    if name == "family_status":
        return payload.get("family_status_after")

    if name == "declared_income":
        current = state.profile_values.get("declared_income") or state.persona.declared_income
        factor = payload.get("factor") or payload.get("income_factor") or 1.0
        return int(round(current * float(factor) / 5000) * 5000)

    if name == "income_type":

        if event.kind == "job_loss":
            return "unemployed"

        # Новая работа возвращает занятость тому, кто её терял.
        # Остальным тип дохода менять незачем: самозанятый не
        # становится наёмным оттого, что сменил заказчика.
        #
        # Но занятость объявляется только если деньги её
        # подтверждают: зарплатный поток к этому моменту
        # действительно открыт. Иначе анкета сообщала бы о работе,
        # которой в ленте нет — ровно та несостыковка, из-за
        # которой безработный оставался безработным навсегда,
        # только с другой стороны.
        if state.profile_values.get("income_type") != "unemployed":
            return state.profile_values.get("income_type")

        moment = event.known_to_bank_at or event.ts

        working = any(
            item.kind == "salary" and item.active_at(moment)
            for item in state.income_streams
        )

        return "employed" if working else "unemployed"

    if name == "industry":

        if event.kind == "job_loss":
            return CLEAR

        # Отрасль есть только у наёмной занятости. Самозанятому,
        # предпринимателю и пенсионеру её не приписывают: в анкете
        # это поле места работы, а не рода занятий.
        settings = params_module.active().population

        if state.profile_values.get("income_type") not in settings.industry_income_types:
            return CLEAR

        # Новая работа это новая отрасль. Возврат прежнего
        # значения делал эффект события пустым.
        choice = keyed_rng(
            NS_PROFILE, state.ordinal, int(event.ts.toordinal()), 1
        ).weighted(settings.industry_weights)
        return str(choice)

    if name == "income_day":

        if event.kind == "job_loss":
            return CLEAR

        low, high = params_module.active().income.salary_day_range
        return int(
            keyed_rng(NS_PROFILE, state.ordinal, int(event.ts.toordinal()), 2).integers(low, high)
        )

    return None


def _on_support_check(sim, state: ClientState, ts: datetime, payload: dict) -> None:
    """
    Обращение связано с причиной, а решение имеет последствие.
    """

    cause = payload["cause"]

    rng = event_rng(NS_SUPPORT, state.ordinal, ts.toordinal(), len(state.cases), COMPONENT_CONTENT)

    probability = support_module.contact_probability(state.persona, cause, ts, payload["stress"])

    if rng.random() >= probability:
        return

    state.support_last_by_cause[cause] = ts

    # Повод обращения известен симуляции и остаётся её знанием:
    # ссылки на событие-причину в выгрузке нет, и искать его
    # здесь больше незачем.
    case = support_module.open_case(state.persona, cause, ts, len(state.cases))

    _emit_case(state, case)

    _touch_client(state, ts)

    if case.resolution == "card_unblocked":
        # Поддержка снимает временную заморозку. Утраченную или
        # скомпрометированную карту не размораживает никто: у
        # такой блокировки нет срока, и она не заканчивается.
        card = next(
            (
                item
                for item in state.cards.values()
                if item.releasable()
            ),
            None,
        )
        if card is not None and case.resolved_at < config.HISTORY_END:
            unblock_card(state, case.resolved_at, card, "support_resolution")

    if case.resolution == "record_corrected":
        state.pending_notice = True


def _migrate_products(sim, state: ClientState, ts: datetime, rng, voluntary: bool) -> None:
    """
    Переход на продукт-преемник: добровольный по предложению,
    принудительный по уведомлению.

    voluntary=True — только добровольные переходы: это решение
    клиента, и оно случается в его активный день. Иначе — все
    остальные политики: их решает банк, каждый день.
    """

    from .engine_credit import close_loan

    for view, target, policy in adoption_module.migration_targets(ts, state.held_codes(ts)):

        if (policy == "voluntary") != voluntary:
            continue

        contract = next(
            (
                item
                for item in state.open_contracts(ts)
                if item.product_code == view.code
            ),
            None,
        )

        if contract is None:
            continue

        if policy == "voluntary":
            chance = 0.0015 * (0.5 + 1.5 * state.persona.trait("digital_affinity", ts))
        elif policy == "forced_with_notice":
            chance = 1.0 if ts >= view.version_start(view.version_at(ts)) else 0.0
        else:
            chance = 0.02 / 365.0

        if rng.random() >= chance:
            continue

        amount, term = sim._contract_terms(state, target, ts, rng)

        # Преемник-вклад без денег не открывается: прежний договор
        # остаётся.
        if amount is None and target.family in ("deposit", "deposit_certificate"):
            continue

        loan = state.loans.get(contract.contract_id)

        if loan is not None and loan.closed:
            loan = None

        if loan is not None:
            # Новый график строится только на непросроченном теле:
            # просроченные взносы переедут в него сами, со своими
            # сроками, и тело не посчитается дважды.
            amount = max(0, loan.principal_outstanding - loan_rules.overdue_principal(loan))

        # Условия активации проверяются ДО закрытия прежнего
        # договора: иначе долг закрылся бы, а новый кредит не
        # завёлся, и деньги клиента пропали бы вместе с ним.
        if not _can_activate(state, target.family, amount, ts + timedelta(minutes=12)):
            continue

        if loan is not None:

            # Долг переходит в договор-преемник, а не выдаётся
            # заново: новые деньги клиенту не приходят, прежний
            # график закрывается без платежа.
            loan.principal_outstanding = 0
            close_loan(state, ts, loan, early=False, reason="migrated_to_successor")

        else:
            contract.status = CONTRACT_CLOSED
            contract.closed_at = ts
            emit_product_closed(state, ts, contract, "migrated_to_successor")

        fresh = sim._open_contract(
            state,
            target,
            ts + timedelta(minutes=10),
            amount,
            term,
            previous_product_id=contract.product_id,
            migration_reason="successor_offer" if policy == "voluntary" else "forced_migration",
        )

        _activate_product(
            sim, state, fresh, target, ts + timedelta(minutes=12), rng,
            previous=loan,
        )

        return


_HANDLERS["support_check"] = _on_support_check
_HANDLERS["adoption"] = _on_adoption
_HANDLERS["tariff_check"] = _on_tariff_check
def _on_card_block_request(sim, state: ClientState, ts: datetime, payload: dict) -> None:
    """
    Клиент блокирует карту сам.

    Временная заморозка и утрата это разные истории. Заморозку
    клиент ставит на всякий случай и снимает сам; утраченную или
    скомпрометированную карту не размораживают никогда, её место
    занимает перевыпущенная.
    """

    card = state.cards.get(payload["card_id"])

    if card is None or not card.usable_at(ts):
        return

    settings = params_module.active().products

    lost = bool(payload["lost"])

    reason = "lost_or_stolen" if lost else "client_freeze"

    rng = event_rng(NS_CARD_BLOCK, state.ordinal, ts.toordinal(), 0, COMPONENT_OUTCOME)

    days = None if lost else int(rng.integers(*settings.card_freeze_days))

    card_rules.block(card, ts, reason, days=days, permanent=lost)

    state.emit(
        state.factory.make(
            "card_blocked",
            ts,
            dict(state.card_facts(card), reason=reason),
        )
    )

    if lost:

        reissue_ts = ts + timedelta(days=int(rng.integers(*settings.card_lost_reissue_delay_days)))

        if reissue_ts < config.HISTORY_END:
            _reissue_card(state, card, reissue_ts, reason="lost_or_stolen")

        return

    # Заморозку клиент чаще снимает сам, не дожидаясь срока.
    # Остальные ждут, и её снимет истёкший срок блокировки.
    if rng.random() >= settings.card_freeze_self_unblock_share:
        return

    unblock_ts = ts + timedelta(
        hours=int(rng.integers(2, max(3, 24 * max(1, days or 1))))
    )

    if unblock_ts < config.HISTORY_END and card.blocked_until is not None and unblock_ts < card.blocked_until:
        unblock_card(state, unblock_ts, card, "client_request")


_HANDLERS["fraud_step"] = _on_fraud_step
_HANDLERS["card_block_request"] = _on_card_block_request
_HANDLERS["profile_change"] = _on_profile_change


__all__ = ["_emit_case"]
