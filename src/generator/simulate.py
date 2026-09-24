from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from . import params as params_module
from .behaviour import adoption as adoption_module
from .behaviour import habits as habits_module
from . import config
from .config import (
    CONTRACT_TERMS_KEYS,
    REGISTRY_START,
)
from .finance import deposits as deposit_rules
from .finance import loans as loan_rules
from .finance.entities import (
    ACCOUNT_CARD,
    ACCOUNT_CREDIT_CARD,
    ACCOUNT_DEPOSIT,
    CARD_ACTIVE,
    CONTRACT_CLOSED,
    Account,
    Card,
    Contract,
)
from .finance.ledger import Ledger
from .life import calendar as cal
from .life import events as life_events
from .life import fraud as fraud_plan
from .life import income as income_module
from .life import lifecycle as lifecycle_module
from .life import stress as stress_module
from .life.persona import Persona, app_adoption, consent_date, draw_persona
from .life.traits import event_shift
from .observe.envelope import Event, EventFactory
from .rng import (
    NS_PREHISTORY,
    NS_ADOPTION,
    NS_CARD,
    NS_LEDGER,
    keyed_rng,
    stable_hash,
)
from .world import geography, products as product_catalog, relationships as graph_module


# ============================================================
# СИМУЛЯЦИЯ СООБЩЕСТВА
# ============================================================
#
# Сообщество это единица симуляции: связанные клиенты живут в
# ОДНОЙ очереди событий, поэтому внутрибанковский перевод
# доходит до получателя и влияет на его последующие решения.
#
# Порядок внутри дня задаётся временем действия, а не номером
# клиента: решение принимается на состояние своей секунды.
# ============================================================


@dataclass
class ClientState:
    persona: Persona
    factory: EventFactory
    ledger: Ledger
    events: list = field(default_factory=list)
    profile_known: bool = False

    life_events: tuple = ()
    stress_episodes: tuple = ()
    pauses: tuple = ()
    fraud_episodes: tuple = ()
    income_streams: tuple = ()
    payouts: tuple = ()
    habits: object = None
    traits: object = None

    contracts: dict = field(default_factory=dict)
    cards: dict = field(default_factory=dict)
    loans: dict = field(default_factory=dict)
    deposits: dict = field(default_factory=dict)
    card_credits: dict = field(default_factory=dict)
    applications: dict = field(default_factory=dict)
    offers: list = field(default_factory=list)

    app_adopted_at: datetime | None = None
    consent_at: datetime | None = None

    state: str = lifecycle_module.STATE_PROSPECT
    last_client_event: datetime | None = None
    last_state_change: datetime | None = None
    comm_fatigue: int = 0
    recent_failure_at: datetime | None = None
    decline_day: int = 0
    decline_count: int = 0
    fraud_alert_at: datetime | None = None
    pending_notice: bool = False
    returned_flag: bool = False

    monthly_atm: int = 0
    monthly_atm_count: int = 0
    monthly_transfer: int = 0
    monthly_cashback: int = 0
    month_purchases: int = 0
    pending_cashback: dict = field(default_factory=dict)
    month_key: int = 0

    profile_values: dict = field(default_factory=dict)
    # Возвраты и отмены, назначенные покупкой на будущие дни:
    # ключ это порядковый номер дня исполнения.
    pending_refunds: dict = field(default_factory=dict)
    open_bills: list = field(default_factory=list)
    cases: list = field(default_factory=list)
    support_last_by_cause: dict = field(default_factory=dict)
    fraud_disputes: set = field(default_factory=set)
    closed_at: datetime | None = None

    # --------------------------------------------------------

    @property
    def ordinal(self) -> int:
        return self.persona.client_ordinal

    @property
    def client_id(self) -> str:
        return self.persona.client_id

    def emit(self, event: Event) -> Event:
        """
        Записывает событие в ленту клиента.

        Три условия, и все про НАБЛЮДЕНИЕ, а не про симуляцию:

          событие раньше начала окна выгрузка не показывает
          вовсе: банк отдаёт окно, а не всю жизнь клиента;

          событие на границе выгрузки или позже в неё не попадает:
          выгрузка сделана в этот момент, и того, что случилось
          после, банк ещё не знает. Проверка нужна именно здесь:
          отдельные места прибавляют к моменту минуты и секунды
          (решение по заявке, вторая нога перевода, подтверждение
          операции) и легко переступают границу;

          система банка, которой в тот день ещё не существовало,
          записать ничего не могла.

        Объект всё равно возвращается: деньги по нему двигались,
        договор открылся, цепочка остатков осталась целой. Просто
        наблюдения этой строки у банка нет.
        """

        if not in_window(event.event_time):
            return event

        if event.event_time >= config.SOURCE_AVAILABILITY[event.source]:
            self.events.append(event)

        return event

    def may_decline(self, ts: datetime) -> bool:
        """
        Разрешён ли ещё один наблюдаемый отказ в этот день.

        Клиент, у которого не хватило денег, не повторяет
        попытку десять раз подряд: после пары отказов он
        перестаёт пробовать через этот банк.
        """

        limit = params_module.active().activity.max_declines_per_day

        day = ts.toordinal()

        if day != self.decline_day:
            self.decline_day = day
            self.decline_count = 0

        if self.decline_count >= limit:
            return False

        self.decline_count += 1

        return True

    def has_open_loan(self) -> bool:
        """
        Есть ли действующий кредит. Закрытый договор не делает
        клиента заёмщиком.
        """

        return any(not item.closed for item in self.loans.values())

    def income_months_at(self, ts: datetime) -> int:
        """
        Сколько месяцев подряд у клиента идёт доход, который банк
        может подтвердить.

        Считается по ДЕЙСТВУЮЩИМ потокам дохода, а не по сроку
        отношений с банком. Человек мог обслуживаться здесь пять
        лет и устроиться на работу вчера — для условия
        income_months это ноль месяцев, а не шестьдесят.

        Стаж отсчитывается от последнего РАЗРЫВА, и разрывом
        считается не только начало потока, но и его конец. Иначе
        достаточно было бы сдавать квартиру десять лет, чтобы
        вчерашняя потеря работы прошла незамеченной: самый старый
        поток продолжался бы, и стаж остался бы десятилетним при
        доходе, упавшем в несколько раз.

        Так что берётся позднейшее из двух: начало непрерывной
        части нынешнего дохода и момент, когда клиент последний
        раз лишился подтверждаемого потока.
        """

        confirmable = [
            item
            for item in self.income_streams
            if item.kind in income_module.CONFIRMABLE_KINDS
        ]

        starts = [item.valid_from for item in confirmable if item.active_at(ts)]

        if not starts:
            return 0

        ended = [
            item.valid_to
            for item in confirmable
            if item.valid_to is not None and item.valid_to <= ts
        ]

        start = min(starts)

        if ended:
            start = max(start, max(ended))

        months = (ts.year - start.year) * 12 + (ts.month - start.month)

        if ts.day < start.day:
            months -= 1

        return max(0, months)

    def open_loan_count(self) -> int:
        """
        Сколько кредитов действует прямо сейчас.

        Условиям каталога нужен именно счёт, а не признак:
        рефинансирование требует двух кредитов, и «хотя бы один»
        этому условию не отвечает.
        """

        return sum(1 for item in self.loans.values() if not item.closed)

    def held_codes(self, ts: datetime) -> frozenset:
        return frozenset(
            contract.product_code
            for contract in self.contracts.values()
            if contract.is_open_at(ts)
        )

    def held_counts(self, ts: datetime) -> dict:
        counts: dict[str, int] = {}
        for contract in self.contracts.values():
            if contract.is_open_at(ts):
                counts[contract.product_code] = counts.get(contract.product_code, 0) + 1
        return counts

    def open_contracts(self, ts: datetime) -> list:
        return [item for item in self.contracts.values() if item.is_open_at(ts)]

    def owned_families(self, ts: datetime) -> frozenset:
        return frozenset(item.product_family for item in self.open_contracts(ts))

    def assets(self) -> int:
        return sum(
            account.balance
            for account in self.ledger.accounts.values()
            if account.kind == ACCOUNT_DEPOSIT
        )

    def primary_card_account(self, ts: datetime) -> Account | None:
        best = None
        for account in self.ledger.accounts.values():
            if account.kind != ACCOUNT_CARD or not account.is_open_at(ts):
                continue
            if best is None or account.balance > best.balance:
                best = account
        return best

    def usable_card(self, account_id: str, ts: datetime) -> Card | None:
        for card in self.cards.values():
            if card.account_id == account_id and card.usable_at(ts):
                return card
        return None

    def bank_income(self) -> int:
        """
        Доход, который видит банк: заявленный в анкете с учётом
        сообщённых изменений. Настоящий доход банку неизвестен,
        поэтому ни сумма, ни одобрение от него не зависят.
        """

        return max(1, int(self.profile_values.get("declared_income") or self.persona.declared_income))

    def worst_dpd(self) -> int:
        return max((state.dpd for state in self.loans.values() if not state.closed), default=0)

    def card_facts(self, card, card_id: str | None = None) -> dict:
        """
        Продуктовая часть payload карточного события.

        Раньше каждое из четырёх мест писало её заново и, не найдя
        договора, подставляло правдоподобные числа: версию 1 и
        семейство debit_card. Выдуманное значение в выгрузке
        неотличимо от настоящего, поэтому здесь его нет: карта без
        договора это поломка симуляции, и она обязана быть видна.
        """

        contract = self.contracts.get(card.contract_id)

        if contract is None:
            raise KeyError(
                f"карта {card.card_id} ссылается на договор {card.contract_id}, "
                f"которого нет у клиента {self.client_id}"
            )

        return {
            "product_id": contract.product_id,
            "contract_id": contract.contract_id,
            "account_id": card.account_id,
            "card_id": card_id or card.card_id,
        }


@dataclass
class Action:
    ts: datetime
    ordinal: int
    order: int
    kind: str
    payload: dict


@dataclass
class CommunityResult:
    events: list
    profile_rows: list


# ============================================================
# ПОМОЩНИКИ
# ============================================================


def _money(value: float) -> int:
    return int(round(value))


def _account_id(client_id: str, kind: str, index: int) -> str:
    return f"acc_{stable_hash('account', client_id, kind, index) % 10 ** 12:012d}"


def _contract_id(client_id: str, code: str, index: int) -> str:
    return f"ctr_{stable_hash('contract', client_id, code, index) % 10 ** 12:012d}"


def _card_id(client_id: str, contract_id: str, index: int) -> str:
    return f"crd_{stable_hash('card', client_id, contract_id, index) % 10 ** 12:012d}"


def _application_id(client_id: str, ts: datetime, index: int) -> str:
    return f"app_{stable_hash('application', client_id, ts.toordinal(), index) % 10 ** 12:012d}"


def in_window(ts: datetime) -> bool:
    """
    Момент попадает в окно выгрузки.

    Границы не симметричны по смыслу, и это важно:

      до начала окна мир ЖИЛ — деньги двигались, договоры
      открывались, просто банк этого в выгрузку не положил;

      на конце окна и позже мира ещё НЕТ — выгрузка снята в этот
      момент, и ничего после него случиться не успело.

    Поэтому проводка до начала окна делается, а после конца —
    нет: иначе остаток менялся бы от события, которого не было.
    """

    return config.HISTORY_START <= ts < config.HISTORY_END


def _transfer_id(client_id: str, ts: datetime, index: int, scope: str = "transfer") -> str:
    """
    Идентификатор перевода: общий у обеих его ног.

    scope разводит независимые источники переводов. Запланированный
    на день перевод и перевод из сессии приложения нумеруются
    каждый по-своему, и без разных областей их номера столкнулись
    бы, склеив два разных перевода в один.
    """

    return f"trf_{stable_hash(scope, client_id, ts.toordinal(), index) % 10 ** 12:012d}"


class CommunitySimulation:
    """
    Одно сообщество от начала до конца окна наблюдения.
    """

    def __init__(self, community_id: int, ordinals: tuple) -> None:

        self.community_id = community_id
        self.settings = params_module.active()

        self.personas = {ordinal: draw_persona(ordinal) for ordinal in ordinals}

        self.graph = graph_module.build_graph(community_id, ordinals, self.personas)

        self.clients: dict[int, ClientState] = {}

        for ordinal in ordinals:
            self.clients[ordinal] = self._prepare(self.personas[ordinal])

        self.by_client_id = {state.client_id: state for state in self.clients.values()}

        # Очередь текущего дня. Она открыта во время исполнения:
        # действие, порождённое другим действием этого же дня,
        # встаёт в неё по своему времени.
        self.queue: list = []
        self.queue_changed = False

    def schedule(self, action: Action) -> None:
        self.queue.append(action)
        self.queue_changed = True

    # --------------------------------------------------------
    # ПОДГОТОВКА
    # --------------------------------------------------------

    def _prepare(self, persona: Persona) -> ClientState:

        state = ClientState(
            persona=persona,
            factory=EventFactory(persona.client_id),
            ledger=Ledger(persona.client_id),
        )

        state.life_events = life_events.plan_events(persona)
        state.stress_episodes = stress_module.plan_episodes(persona, state.life_events)
        state.pauses = lifecycle_module.plan_pauses(persona, state.life_events)
        state.fraud_episodes = fraud_plan.plan_episodes(persona, state.life_events)
        state.income_streams = income_module.build_streams(persona, state.life_events)

        state.payouts = tuple(
            sorted(
                income_module.payouts(persona, state.income_streams, state.stress_episodes)
                + income_module.vacation_payouts(persona, state.income_streams, state.life_events),
                key=lambda item: item.ts,
            )
        )

        state.habits = habits_module.build_habits(persona, state.life_events)

        traits = persona.traits

        for event in state.life_events:
            shift = event_shift(event.kind, event.ts)
            if shift is not None:
                traits = traits.with_shift(shift)

        state.traits = traits

        state.app_adopted_at = app_adoption(persona.client_ordinal)
        state.consent_at = consent_date(persona.client_ordinal)

        state.profile_values = self._initial_profile(persona)

        return state

    def _initial_profile(self, persona: Persona) -> dict:

        return {
            "age": persona.age_at(config.HISTORY_START),
            "gender": persona.gender,
            "family_status": persona.family_status,
            "children": persona.children,
            "education": persona.education,
            "region": persona.region,
            # Сельская корзина области названием города не
            # становится: у села имени в генераторе нет.
            "city": geography.by_name(persona.settlement).public_name,
            "housing_type": persona.housing_type,
            "pensioner": persona.is_pensioner_at(config.HISTORY_START),
            "income_type": persona.income_type,
            "declared_income": persona.declared_income,
            "industry": persona.industry,
            "income_day": persona.income_day,
            "relationship_months": persona.relationship_months_at(config.HISTORY_START),
            "contracts_count": 0,
            "active_contracts": 0,
            "holds_credit_card": False,
            "holds_debit_card": False,
            "holds_deposit": False,
            "credit_limit": None,
            "credit_utilization": None,
        }

    # --------------------------------------------------------
    # ПРОДУКТЫ
    # --------------------------------------------------------

    def _pick_product(self, state: ClientState, family: str, ts: datetime):

        catalog = product_catalog.catalog()

        pool = [
            view
            for view in catalog.by_family(family)
            if view.sellable_at(ts) and not view.record.is_synthetic
        ]

        if not pool:
            pool = [view for view in catalog.by_family(family) if view.serviced_at(ts)]

        if not pool:
            return None

        rng = keyed_rng(NS_ADOPTION, state.ordinal, ts.toordinal(), stable_hash(family) % 997)

        weights = []

        for view in pool:
            weight = 1.0
            if view.status_at(ts) == product_catalog.STATUS_ACTIVE:
                weight *= 3.0
            eligibility = view.version_at(ts).eligibility or {}
            if eligibility.get("min_assets", 0) > state.assets():
                weight *= 0.05
            if eligibility.get("requires_pension") and not state.persona.is_pensioner_at(ts):
                weight *= 0.02
            weights.append(weight)

        return pool[int(rng.choice(len(pool), p=weights))]

    def _open_contract(
        self,
        state: ClientState,
        view,
        ts: datetime,
        amount: int | None,
        term: int | None,
        offer_id: str | None = None,
        application_id: str | None = None,
        previous_product_id: str | None = None,
        migration_reason: str | None = None,
        emit_events: bool = True,
    ) -> Contract:
        """
        Открывает договор на версии продукта, действующей на дату
        подписания, и заводит нужные счёт и карту.
        """

        version = view.version_at(ts)

        index = len(state.contracts) + 1

        contract_id = _contract_id(state.client_id, view.code, index)

        family = view.family

        rate = version.terms.get("rate")

        if rate is None and "rate_by_term" in version.terms and term:
            rate = version.terms["rate_by_term"].get(term) or version.terms["rate_by_term"].get(str(term))

        contract = Contract(
            contract_id=contract_id,
            client_id=state.client_id,
            product_id=view.record.product_id,
            product_code=view.code,
            product_family=family,
            product_version=version.product_version,
            tariff_version=version.tariff_version,
            opened_at=ts,
            amount_or_limit=amount,
            term=term,
            rate=float(rate) if rate is not None else None,
            offer_id=offer_id,
            previous_product_id=previous_product_id,
            application_id=application_id,
            terms=dict(version.terms),
        )

        account_kind = {
            "debit_card": ACCOUNT_CARD,
            "credit_card": ACCOUNT_CREDIT_CARD,
            "deposit": ACCOUNT_DEPOSIT,
            "deposit_certificate": ACCOUNT_DEPOSIT,
        }.get(family)

        account = None

        if account_kind is not None:

            account = state.ledger.add_account(
                Account(
                    account_id=_account_id(state.client_id, account_kind, index),
                    client_id=state.client_id,
                    kind=account_kind,
                    opened_at=ts,
                    contract_id=contract_id,
                    product_code=view.code,
                    credit_limit=int(amount or 0) if family == "credit_card" else 0,
                )
            )

            contract.account_id = account.account_id

        card = None

        if family in ("debit_card", "credit_card"):

            card = Card(
                card_id=_card_id(state.client_id, contract_id, index),
                account_id=contract.account_id,
                client_id=state.client_id,
                contract_id=contract_id,
                product_code=view.code,
                issued_at=ts,
                expires_at=cal.add_months(
                    ts, 12 * int(self.settings.products.card_expiry_years)
                ),
            )

            state.cards[card.card_id] = card
            contract.card_id = card.card_id

        state.contracts[contract_id] = contract

        if not emit_events:
            if card is not None:
                card.status = CARD_ACTIVE
                card.activated_at = ts
            return contract

        # Время договора точное: витрина больше его не теряет.
        registry_ts = ts

        payload = {
            "product_id": contract.product_id,
            "contract_id": contract_id,
            "account_id": contract.account_id,
            "card_id": contract.card_id,
            "offer_id": offer_id,
            # Договор прямо называет заявку, по которой открыт.
            # Раньше связь читалась только по reason и требовала
            # догадки, а заявок у клиента может быть несколько.
            "application_id": application_id,
            "previous_product_id": previous_product_id,
            "migration_reason": migration_reason,
            "amount_or_limit": amount,
            "term": term,
            "rate": contract.rate,
            "reason": "application_approved" if application_id else "opened",
        }

        if account is not None:
            state.emit(
                state.factory.make(
                    "account_opened",
                    registry_ts,
                    payload,
                )
            )

        state.emit(
            state.factory.make(
                "product_migrated" if migration_reason else "product_opened",
                registry_ts,
                payload,
            )
        )

        if card is not None:

            rng = keyed_rng(NS_CARD, state.ordinal, ts.toordinal(), index)

            low, high = self.settings.products.card_activation_delay_days

            activation = ts + timedelta(days=int(rng.integers(low, high + 1)), hours=int(rng.integers(1, 20)))

            if activation >= config.PLANNING_END:
                activation = ts

            card.activated_at = activation
            card.status = CARD_ACTIVE

            # Активация карты условий не назначает: сумма, срок
            # и ставка остались в событии открытия договора.
            state.emit(
                state.factory.make(
                    "card_activated",
                    activation,
                    {
                        name: value
                        for name, value in payload.items()
                        if name not in CONTRACT_TERMS_KEYS
                    },
                )
            )

        return contract

    # --------------------------------------------------------
    # ПРЕДЫСТОРИЯ
    # --------------------------------------------------------

    def _prehistory(self, state: ClientState) -> None:
        """
        Договоры, открытые до окна наблюдения. Прошлые проводки
        не выдумываются: с чем клиент вошёл в окно, видно по
        первому же остатку его ленты.
        """

        persona = state.persona

        if persona.relationship_start >= config.HISTORY_START:
            return

        rng = keyed_rng(NS_LEDGER, state.ordinal, 1)

        # Первая дебетовая карта в день прихода в банк.
        view = self._pick_product(state, "debit_card", persona.relationship_start)

        if view is not None:
            self._open_contract(
                state,
                view,
                persona.relationship_start.replace(hour=11, minute=30),
                amount=None,
                term=None,
                emit_events=persona.relationship_start >= REGISTRY_START,
            )

        span = max(1, (config.HISTORY_START - persona.relationship_start).days)

        products = self.settings.products

        for index, (family, rule) in enumerate(products.prehistory_penetration.items()):

            base, slope, trait = rule

            probability = base + slope * persona.trait(trait)

            # Отдельный поток: добавление правил предыстории не
            # должно сдвинуть начальные остатки, которые
            # разыгрываются из того же rng ниже.
            family_rng = keyed_rng(NS_PREHISTORY, persona.client_ordinal, index)

            if family_rng.random() >= min(products.prehistory_max_probability, probability):
                continue

            offset = int(family_rng.integers(0, span))

            ts = persona.relationship_start + timedelta(
                days=offset, hours=int(family_rng.integers(9, 19))
            )

            if ts >= config.HISTORY_START:
                continue

            item = self._pick_product(state, family, ts)

            if item is None:
                continue

            amount, term = self._contract_terms(state, item, ts, family_rng, affordable=False)

            # Правила банка действовали и до окна наблюдения:
            # ни лишней карты сверх лимита, ни кредита сверх
            # долговой нагрузки.
            if not adoption_module.eligible(
                persona,
                item,
                item.version_at(ts),
                ts,
                state.held_codes(ts),
                state.held_counts(ts),
                state.assets(),
                True,
                state.open_loan_count(),
                len(state.open_contracts(ts)),
                state.income_months_at(ts),
                amount,
            ):
                continue

            if family in ("cash_loan", "installment") and not self._prehistory_debt_fits(
                state, ts, family, amount, term, item
            ):
                continue

            contract = self._open_contract(
                state, item, ts, amount, term, emit_events=ts >= REGISTRY_START
            )

            # График строится сразу, чтобы следующий договор
            # предыстории видел уже принятую нагрузку.
            self._register_prehistory_loan(state, contract, family_rng)

        # Начальные остатки.
        opening_cash = _money(persona.true_income * rng.uniform(0.05, 0.45))
        opening_other = _money(persona.true_income * rng.uniform(0.2, 1.8) * (1.0 - persona.visible_share))

        state.ledger.accounts[state.ledger.cash_id].balance = opening_cash
        state.ledger.accounts[state.ledger.other_bank_id].balance = opening_other

        card_account = state.primary_card_account(config.HISTORY_START)

        if card_account is not None:
            card_account.balance = _money(
                persona.true_income * persona.visible_share * rng.uniform(0.1, 0.9)
            )

        for account in state.ledger.accounts.values():
            if account.kind == ACCOUNT_DEPOSIT:
                account.balance = _money(persona.true_income * rng.uniform(1.0, 8.0))

        # Вклад предыстории живёт так же, как открытый в окне: у
        # него есть состояние, ставка и срок. Раньше состояния не
        # было, и такой вклад не зарабатывал процентов, не
        # заканчивался и не закрывался досрочно. Срок, вышедший до
        # окна, считается пролонгированным: ближайшее окончание —
        # первое внутри окна.
        for contract in state.contracts.values():

            if contract.product_family not in ("deposit", "deposit_certificate"):
                continue

            account = state.ledger.get(contract.account_id) if contract.account_id else None

            if account is None or account.balance <= 0 or not contract.is_open_at(config.HISTORY_START):
                continue

            terms = contract.terms
            term = int(contract.term or 12)

            deposit = deposit_rules.open_deposit(
                contract_id=contract.contract_id,
                account_id=contract.account_id,
                amount=account.balance,
                rate=float(contract.rate or terms.get("rate") or 0.14),
                opened_at=contract.opened_at,
                term_months=term,
                topup=bool(terms.get("topup", False)),
                withdrawal=bool(terms.get("withdrawal", False)),
                capitalisation=str(terms.get("capitalisation", "daily")),
            )

            while deposit.matures_at <= config.HISTORY_START:
                deposit.matures_at = cal.add_months(deposit.matures_at, term)
                contract.renewals += 1

            state.deposits[contract.contract_id] = deposit

        # Остаток, с которым клиент вошёл в окно: он не результат
        # наблюдавшихся проводок, а начальное условие ленты.
        for account in state.ledger.accounts.values():
            account.opening_balance = account.balance

    def _outlets_by_id(self, state: ClientState, category: str, ts: datetime) -> tuple:
        from .world import merchants as catalog

        era = state.habits.era_at(ts)

        return catalog.outlets_of(era.settlement, category)

    def _prehistory_debt_fits(
        self, state: ClientState, ts, family: str, amount, term, item
    ) -> bool:
        """
        Долговая нагрузка проверяется и до окна наблюдения.
        """

        if amount is None or term is None:
            return True

        products = self.settings.products

        ratio = float(products.bank_rules.get("max_debt_service_ratio", 0.5))

        income = state.bank_income()

        open_loans = tuple(loan for loan in state.loans.values() if not loan.closed)

        existing = loan_rules.debt_service(open_loans, ts)

        rate = self._rate_for(item.version_at(ts).terms, term)

        payment = loan_rules.annuity_payment(int(amount), rate, int(term))

        return (existing + payment) <= ratio * income

    def _register_prehistory_loan(self, state: ClientState, contract, rng) -> None:
        """
        Кредит предыстории: график, исполненные до окна платежи
        и остаток долга на начало наблюдения.
        """

        if contract is None:
            return

        if contract.product_family not in ("cash_loan", "refinance", "installment"):
            return

        if contract.amount_or_limit is None or contract.term is None:
            return

        loan = loan_rules.open_loan(
            contract.contract_id,
            int(contract.amount_or_limit),
            float(contract.rate or 0.28),
            int(contract.term),
            contract.opened_at,
            autopay=rng.random() < self.settings.products.autopay_share,
        )

        for item in loan.schedule:
            if item.due_date < config.HISTORY_START:
                loan_rules.apply_payment(loan, item, item.amount, item.due_date)

        if loan.principal_outstanding <= 0:
            contract.status = CONTRACT_CLOSED
            contract.closed_at = max(
                contract.opened_at,
                cal.add_months(contract.opened_at, int(contract.term)),
            )
            return

        state.loans[contract.contract_id] = loan

    def _rate_for(self, terms: dict, term) -> float:
        """
        Ставка версии договора: прямая или по сроку.
        """

        rate = terms.get("rate")

        if rate is not None:
            return float(rate)

        by_term = terms.get("rate_by_term") or {}

        value = by_term.get(str(term)) or by_term.get(term)

        return float(value) if value is not None else 0.28

    def _fit_to_debt_service(
        self,
        state: ClientState,
        ts: datetime,
        family: str,
        amount: int,
        term,
        terms: dict,
        floor: int = 0,
    ) -> int:
        """
        Сумма урезается так, чтобы платёж вместе с уже
        имеющимися обязательствами укладывался в долговую
        нагрузку банка.

        Если даже минимальная сумма не влезает, она остаётся
        минимальной, а отказ выносит решение по заявке.
        """

        products = self.settings.products

        ratio = float(products.bank_rules.get("max_debt_service_ratio", 0.5))

        income = state.bank_income()

        open_loans = tuple(item for item in state.loans.values() if not item.closed)

        existing = 0 if family == "refinance" else loan_rules.debt_service(open_loans, ts)

        if family == "credit_card":
            share = products.credit_card_payment_share_of_limit
            capacity = int(ratio * income) - existing
            allowed = int(capacity / share) if share > 0 else amount
        else:
            allowed = loan_rules.max_amount_for_dsr(
                income, existing, self._rate_for(terms, term), int(term or 12), ratio
            )

        return max(floor, min(amount, allowed)) if allowed > 0 else max(floor, min(amount, floor))

    def _refinance_amount(
        self,
        state: ClientState,
        ts: datetime,
        drawn: int,
        low: int,
        high: int,
        rng,
    ) -> int:
        """
        Рефинансирование гасит имеющиеся долги и добирает
        немного наличных сверху.
        """

        outstanding = sum(
            loan_rules.payoff_amount(item)
            for item in state.loans.values()
            if not item.closed
        )

        if outstanding <= 0:
            return drawn

        topup = rng.uniform(*self.settings.products.refinance_cash_topup_share)

        return int(min(high, max(low, outstanding * (1.0 + topup))))

    def _contract_terms(
        self, state: ClientState, view, ts: datetime, rng, affordable: bool = True
    ) -> tuple:
        """
        Сумма и срок будущего договора.

        affordable: сумма вклада ограничена тем, что клиент может
        собрать на одном счёте прямо сейчас. Предыстория передаёт
        False: её остатки разыгрываются позже и отдельно.
        """

        terms = view.version_at(ts).terms

        income = state.bank_income()

        family = view.family

        if family == "debit_card":
            return None, None

        products = self.settings.products

        multiples = products.loan_amount_income_multiple

        if family == "credit_card":
            low = int(terms.get("limit_min", 20_000))
            high = int(terms.get("limit_max", 2_000_000))
            limit = int(min(high, max(low, income * rng.uniform(*multiples["credit_card"]))))
            limit = self._fit_to_debt_service(state, ts, family, limit, None, terms)
            return int(round(limit / 10_000) * 10_000), int(terms.get("installment_months", 0)) or None

        if family in ("cash_loan", "refinance"):
            low = int(terms.get("amount_min", 10_000))
            high = int(terms.get("amount_max", 9_500_000))
            amount = int(min(high, max(low, income * rng.uniform(*multiples[family]))))
            term = int(rng.choice(list(products.loan_term_options), p=list(products.loan_term_weights)))
            term = max(int(terms.get("term_min", 6)), min(int(terms.get("term_max", 60)), term))

            if family == "refinance":
                amount = self._refinance_amount(state, ts, amount, low, high, rng)

            amount = self._fit_to_debt_service(state, ts, family, amount, term, terms, floor=low)
            return int(round(amount / 1_000) * 1_000), term

        if family == "installment":
            low = int(terms.get("amount_min", 10_000))
            high = int(terms.get("amount_max", 1_500_000))
            amount = int(min(high, max(low, income * rng.uniform(*multiples["installment"]))))
            options = list(terms.get("term_options", (6, 12, 24)))
            term = int(rng.choice(options))
            amount = self._fit_to_debt_service(state, ts, family, amount, term, terms, floor=low)
            return int(round(amount / 1_000) * 1_000), term

        if family in ("deposit", "deposit_certificate"):
            minimum = int(terms.get("min_amount", 1_000))
            free = max(minimum, state.ledger.total_visible_balance() + state.ledger.hidden_funds())
            amount = int(max(minimum, free * rng.uniform(*self.settings.products.deposit_open_share_of_free_cash)))
            options = list(terms.get("term_options", (12,)))
            term = int(rng.choice(options))

            if affordable:
                # Вклад кладут с ОДНОГО счёта, подтянув недостающее
                # наличными или переводом из другого банка. Сумма не
                # больше того, что клиент способен собрать сейчас;
                # меньше минимума — вклада не будет. Раньше сумма
                # считалась от всех денег сразу, финансирование
                # срывалось, и договор висел открытым без остатка.
                reachable = state.ledger.payment_capacity(ts) + max(
                    0,
                    state.ledger.balance(state.ledger.cash_id),
                    state.ledger.balance(state.ledger.other_bank_id),
                )
                amount = int(min(amount, reachable) // 1_000 * 1_000)
                if amount < minimum:
                    return None, term
                return amount, term

            return int(round(amount / 1_000) * 1_000), term

        if family == "insurance":
            price = int(terms.get("price_annual", 12_000))
            return price, int(terms.get("term_months", 12))

        if family == "bonds":
            rate = self.settings.amounts.fx_rates.get("USD", 500.0)
            amount = int(int(terms.get("min_amount_usd", 1_000)) * rate)
            return amount, int(terms.get("term_max_months", 12))

        return None, None


__all__ = [
    "Action",
    "ClientState",
    "CommunityResult",
    "CommunitySimulation",
]
