from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, timedelta

import pyarrow.parquet as pq
import pytest

from src.generator import emit
from src.generator import params as params_module
from src.generator.life import calendar as cal
from src.generator.world import products as product_catalog


# ============================================================
# ИДЕЯ
# ============================================================
#
# Инварианты реализма из аудита 2026-11-05
# (audit/2026-11-05-generator-realism): то, что у реального банка
# не бывает никогда, не бывает и в выгрузке. Числовые полосы
# (доли, частоты) проверяет checks/scorecard.py на scratch-мире;
# здесь — только правила без исключений.
# ============================================================


START = datetime(2024, 1, 1)
END = datetime(2025, 7, 1)


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict:

    out = tmp_path_factory.mktemp("realism") / "raw"

    emit.generate_dataset(
        total_clients=48, out_dir=out, seed=93, world_seed=42, history_start=START,
        history_end=END, registration_end=END, workers=1, community_size=4, quiet=True,
    )

    table = pq.read_table(out / "events.parquet").to_pylist()

    events = []
    for row in table:
        payload = json.loads(row["payload"])
        payload["client_id"] = row["client_id"]
        payload["ts"] = datetime.fromisoformat(row["event_time"]).replace(tzinfo=None)
        events.append(payload)

    profile = pq.read_table(out / "profile.parquet").to_pylist()

    transitions = pq.read_table(out / "truth" / "transitions.parquet").to_pylist()

    return {"events": events, "profile": profile, "transitions": transitions}


def _registered(row) -> datetime:
    return min(item["event_time"] for item in row["lifelong"] if item["type"] == "bank_registered").replace(tzinfo=None)


def test_every_client_from_before_the_window_has_an_account(world) -> None:
    """Давний клиент входит в окно со счётом (C2: до 2018 карт в каталоге не было)."""

    with_account = {item["client_id"] for item in world["events"] if item.get("account_id")}

    old = [row["client_id"] for row in world["profile"] if _registered(row) < START]

    assert old, "в мире нет давних клиентов — проверять нечего"
    assert set(old) <= with_account


def test_transfer_fields_do_not_reveal_the_kind_of_transfer(world) -> None:
    """У каждого перевода — счёт и номер операции, прошёл он или нет (C4)."""

    transfers = [item for item in world["events"] if item["type"] in ("transfer_out", "p2p_out")]

    assert transfers
    for item in transfers:
        assert item.get("account_id"), item
        assert item.get("transfer_id"), item


def test_declines_name_the_account_they_tried(world) -> None:
    """Отказ — попытка списать со счёта: без счёта отказа не бывает (money-20)."""

    declined = [item for item in world["events"] if item.get("status") == "declined"]

    assert declined
    assert all(item.get("account_id") for item in declined)


def test_no_contract_stays_open_without_money(world) -> None:
    """Открытый кредит выдан или аннулирован; полис оплачен или аннулирован (M10)."""

    params_module.activate(params_module.load(None))
    family = {view.record.product_id: view.family for view in product_catalog.catalog().views.values()}

    opened = {item["contract_id"]: item for item in world["events"] if item["type"] == "product_opened"}
    funded = {item["contract_id"] for item in world["events"]
              if item["type"] in ("loan_disbursement", "schedule_created", "product_closed")
              or (item["type"] in ("purchase", "transfer_out") and item.get("contract_id")
                  and item.get("reason") in ("insurance_premium", "bond_purchase"))}

    late = END.replace(day=1)
    checked = 0
    for contract_id, item in opened.items():
        if family.get(item["product_id"]) not in ("cash_loan", "installment", "refinance", "insurance", "bonds"):
            continue
        if item["ts"] >= late:
            continue
        checked += 1
        assert contract_id in funded, item

    assert checked


def test_closed_account_keeps_no_money(world) -> None:
    """Закрытие счёта не уносит денег: последний остаток до закрытия — ноль (M11)."""

    balance: dict[str, int] = {}
    lost = []
    for item in world["events"]:
        if item.get("balance_after") is not None and item.get("account_id"):
            balance[item["account_id"]] = item["balance_after"]
        if item["type"] == "product_closed" and item.get("account_id"):
            if balance.get(item["account_id"], 0) > 0:
                lost.append(item)

    assert not lost


def test_card_arrears_are_cleared_only_by_the_full_minimum(world) -> None:
    """Просрочку карты снимает только полный минимальный платёж (money-03)."""

    last_paid: dict[str, dict] = {}
    for item in world["events"]:
        if item["type"] == "installment_paid" and item.get("installment_no") is None:
            last_paid[item["contract_id"]] = item
        if item["type"] == "arrears_cleared" and item["contract_id"] in last_paid:
            paid = last_paid[item["contract_id"]]
            assert paid["amount_paid"] >= paid["amount_due"], (paid, item)


def test_loan_rates_come_from_the_catalogue_range(world) -> None:
    """Ставка кредита записана и лежит в диапазоне каталога (M12)."""

    rates = [item["rate"] for item in world["events"]
             if item["type"] == "product_opened" and item.get("rate") is not None]

    assert rates
    assert all(0.0 <= rate <= 0.46 for rate in rates)


def test_loan_delinquency_marks_are_reset_after_cure(world) -> None:
    """Погашенная просрочка кредита даёт arrears_cleared, и повторная начинается с dpd_1 (M13)."""

    marks: dict[str, list] = defaultdict(list)
    for item in world["events"]:
        if item["type"] == "delinquency_registered":
            marks[item["contract_id"]].append(item["reason"])
        if item["type"] == "arrears_cleared":
            marks[item["contract_id"]].append("cleared")

    for sequence in marks.values():
        for before, after in zip(sequence, sequence[1:]):
            if before == "cleared":
                assert after in ("dpd_1", "delinquency", "cleared"), sequence


@pytest.mark.parametrize(
    ("day", "off"),
    [
        (date(2024, 3, 25), True),   # Наурыз 23.03.2024 — суббота, перенос на понедельник
        (date(2024, 5, 8), True),    # перенос постановлением
        (date(2024, 5, 4), False),   # рабочая суббота
        (date(2024, 12, 17), False),  # 17 декабря с 2022 года — рабочий день
        (date(2025, 1, 3), True),
        (date(2025, 1, 5), False),
        (date(2025, 3, 10), True),   # 8 марта — суббота
        (date(2025, 3, 25), True),   # Наурыз 21–23.03.2025 — пятница–воскресенье
        (date(2025, 9, 1), True),    # 30 августа — суббота
        (date(2026, 8, 31), False),  # с 01.07.2026 30 августа не праздник
        (date(2025, 6, 9), False),   # Курбан айт 06.06.2025 — пятница, религиозный не переносится
    ],
)
def test_production_calendar(day: date, off: bool) -> None:
    params_module.activate(params_module.load(None))
    assert cal.is_day_off(datetime(day.year, day.month, day.day, 12)) is off


# ============================================================
# ВОЛНА 2: ИДЕНТИЧНОСТЬ МИРА
# ============================================================


def test_a_trip_abroad_happens_in_one_country(world) -> None:
    """Поездка — одна страна: покупки за границей подряд (разрыв до 3 дней) в одной стране (C1)."""

    foreign = sorted(
        (item for item in world["events"]
         if item["type"] == "purchase" and item.get("merchant_country") not in (None, "KZ")),
        key=lambda item: (item["client_id"], item["ts"]),
    )

    # Мошенничество по карте идёт из своей страны эпизода, не из
    # страны поездки клиента: покупки рядом с эпизодом (truth) не
    # сравниваются.
    fraud: dict[str, list] = defaultdict(list)
    for row in world["transitions"]:
        if row["component"] == "fraud":
            fraud[row["client_id"]].append(row["time"].replace(tzinfo=None) + (datetime(2000, 1, 1, 5) - datetime(2000, 1, 1)))

    def near_fraud(item) -> bool:
        return any(abs((item["ts"] - moment).days) <= 3 for moment in fraud.get(item["client_id"], ()))

    for before, after in zip(foreign, foreign[1:]):
        if before["client_id"] != after["client_id"] or (after["ts"] - before["ts"]).days > 3:
            continue
        if near_fraud(before) or near_fraud(after):
            continue
        assert before["merchant_country"] == after["merchant_country"], (before, after)


def test_a_bill_has_one_provider(world) -> None:
    """Оператор связи у клиента один за всё окно (C1, context-11)."""

    providers: dict[str, set] = defaultdict(set)
    for item in world["events"]:
        if item["type"] == "bill_payment" and item.get("merchant_category") == "telecom" and item.get("merchant_id"):
            providers[item["client_id"]].add(item["merchant_id"])

    assert providers
    assert all(len(found) == 1 for found in providers.values()), providers


def test_salary_names_the_employer_and_comes_as_one_registry(world) -> None:
    """Зарплата названа работодателем; у коллег в один день — одна пачка реестра (C1, M8)."""

    salaries = [item for item in world["events"] if item["type"] == "salary_credit"]

    assert salaries
    assert all(item["counterparty"] not in ("Employer", None) for item in salaries)

    # По одной выплате клиента на работодателя и день: у одного
    # клиента в день бывает и аванс, и доплата.
    first: dict[tuple, datetime] = {}
    for item in salaries:
        key = (item["counterparty"], item["ts"].date(), item["client_id"])
        first[key] = min(first.get(key, item["ts"]), item["ts"])

    batches: dict[tuple, list] = defaultdict(list)
    for (payer, day, _), moment in first.items():
        batches[(payer, day)].append(moment)

    for moments in batches.values():
        assert (max(moments) - min(moments)).total_seconds() <= 240, moments

    assert all(cal.is_business_day(item["ts"]) for item in salaries if item.get("reason") != "partial_topup")


def test_fixed_prices_are_fixed(world) -> None:
    """Проезд — по тарифу города, подписка сервиса — по его прейскуранту (M3)."""

    from src.generator.behaviour.habits import brand_key

    params_module.activate(params_module.load(None))
    tariffs = params_module.active().amounts.subscription_tariffs
    allowed = {}
    for name, value in tariffs.items():
        prices = set(value["tiers"])
        if value.get("raise"):
            prices |= {int(round(price * value["raise"][1] / 10) * 10) for price in value["tiers"]}
        allowed[brand_key(name)] = prices

    fares = set()
    for steps in params_module.active().amounts.transit_fares.values():
        fares |= {value for _, value in steps}

    for item in world["events"]:
        if item["type"] != "purchase" or item.get("status") != "approved":
            continue
        if item.get("merchant_category") == "transit":
            assert item["amount"] in fares, item
        if item.get("is_subscription") and brand_key(item.get("merchant_name")) in allowed:
            assert item["amount"] in allowed[brand_key(item["merchant_name"])], item


# ============================================================
# ВОЛНА 3: ЦЕНА ОТДЕЛЬНО ОТ ЧАСТОТЫ
# ============================================================


def test_no_price_floor_peak(world) -> None:
    """Пол суммы — не пик: покупок ровно по 100 ₸ вне проезда почти нет (C3)."""

    purchases = [item for item in world["events"]
                 if item["type"] == "purchase" and item.get("status") == "approved"
                 and item.get("merchant_category") != "transit"]

    assert purchases
    assert sum(item["amount"] == 100 for item in purchases) / len(purchases) < 0.01


def test_purchase_price_does_not_follow_the_payday_cycle() -> None:
    """Сумма покупки не зависит от дня: у функции цены нет дневного множителя (C3)."""

    import inspect

    from src.generator.behaviour import merchants as merchant_choice

    assert "spending_factor" not in inspect.signature(merchant_choice.purchase_amount).parameters


def test_exchange_rates_follow_the_national_bank() -> None:
    """Курс — среднемесячный НБК, а не константа (M7)."""

    from src.generator.params import fx

    assert fx.rate("USD", datetime(2024, 1, 10)) == pytest.approx(451.28)
    assert fx.rate("USD", datetime(2025, 1, 10)) == pytest.approx(524.67)
    assert fx.rate("EUR", datetime(2025, 9, 10)) != fx.rate("EUR", datetime(2024, 9, 10))


def test_prices_grow_with_inflation() -> None:
    """Индекс цен продуктов за 2024 г. равен ИПЦ продуктов 5.5% (M7)."""

    params_module.activate(params_module.load(None))

    assert cal.price_index("grocery", datetime(2024, 1, 15)) == pytest.approx(1.0)
    assert cal.price_index("grocery", datetime(2025, 1, 15)) == pytest.approx(1.055, rel=1e-6)
    assert cal.price_index("utilities", datetime(2025, 1, 15)) == pytest.approx(1.142, rel=1e-6)


def test_foreign_amount_has_cents_and_topups_are_not_at_midnight(world) -> None:
    """Сумма в валюте — с центами; пополнение под платёж — не в 00:00:00 (M7, M22)."""

    foreign = [item for item in world["events"] if item.get("original_amount") is not None]
    if foreign:
        assert any(float(item["original_amount"]) != int(float(item["original_amount"])) for item in foreign)

    topups = [item for item in world["events"] if item.get("reason") == "payment_topup"]
    assert all(item["ts"].time().isoformat() != "00:00:00" for item in topups)


# ============================================================
# ВОЛНА 4: АНКЕТА → ПОВЕДЕНИЕ
# ============================================================


@pytest.fixture(scope="module")
def personas():

    from src.generator import config, rng as rng_module
    from src.generator.emit import _build_params
    from src.generator.life.persona import draw_persona

    config.activate_horizon(START, END, END)
    settings = _build_params(None, None, None)
    params_module.activate(settings)
    rng_module.configure(93, settings.fingerprint(), 42)

    return [draw_persona(ordinal) for ordinal in range(1, 3001)]


def test_habits_depend_on_sex_and_age(personas) -> None:
    """Курят в основном мужчины; пожилые почти не делают ставок (C6)."""

    def share(items, name):
        return sum(getattr(item, name) for item in items) / max(1, len(items))

    men = [item for item in personas if item.gender == "M"]
    women = [item for item in personas if item.gender == "F"]
    old = [item for item in personas if item.age_at(START) >= 65]

    assert 0.25 < share(men, "smokes") < 0.45
    assert share(women, "smokes") < 0.10
    assert share(old, "gambles") < 0.05


def test_structural_zeros_and_car_spending(personas) -> None:
    """Без признака категории нет; машинные траты — у водящих (C6, context-13)."""

    from src.generator.behaviour import habits as habits_module
    from src.generator.behaviour.needs import _category_weights

    moment = datetime(2024, 6, 3, 12)

    from dataclasses import replace

    for persona in personas[:200]:
        habits = habits_module.build_habits(persona, ())
        weights = _category_weights(persona, habits, moment)
        if not persona.smokes:
            assert "tobacco" not in weights
        if not persona.gambles:
            assert "gambling" not in weights
        if persona.transport != "car" and "parking" in weights:
            driving = _category_weights(replace(persona, transport="car"), habits, moment)
            assert weights["parking"] * 20 <= driving["parking"]


def test_demography_is_consistent(personas) -> None:
    """Высшего нет до 22 лет; машина чаще у богатых; регион и пол — в доходе (C6, M14)."""

    import statistics

    for persona in personas:
        if persona.age_at(START) < 22:
            assert persona.education not in ("higher", "academic_degree"), persona

    # Внутри одного типа поселения машина чаще у богатых.
    richer = poorer = 0
    for kind in {item.settlement_type for item in personas}:
        group = sorted((item for item in personas if item.settlement_type == kind), key=lambda item: item.true_income)
        if len(group) < 100:
            continue
        third = len(group) // 3
        top = sum(item.transport == "car" for item in group[-third:])
        bottom = sum(item.transport == "car" for item in group[:third])
        richer += top > bottom
        poorer += top < bottom
    assert richer > poorer

    def median(items):
        return statistics.median(item.true_income for item in items)
    atyrau = [item for item in personas if item.region == "Atyrau" and item.income_type == "employed"]
    shymkent = [item for item in personas if item.region == "Shymkent" and item.income_type == "employed"]
    if len(atyrau) >= 10 and len(shymkent) >= 10:
        assert median(atyrau) > median(shymkent)

    workers = [item for item in personas if item.income_type == "employed"]
    assert median([item for item in workers if item.gender == "M"]) > median([item for item in workers if item.gender == "F"])

    pensioners = [item for item in personas if item.income_type == "pensioner"]
    assert 100_000 < median(pensioners) < 180_000


# ============================================================
# ВОЛНА 5: ЖИЗНЬ И ПОСЛЕДСТВИЯ (M9, M14–M17, M23)
# ============================================================


@pytest.fixture(scope="module")
def lives(personas):

    from src.generator.life import income
    from src.generator.life.events import plan_events

    result = []
    for persona in personas:
        events = plan_events(persona)
        streams = income.build_streams(persona, events)
        result.append((persona, events, streams))
    return result


def test_births_follow_the_mother(lives) -> None:
    """Рожают в 15–49 лет, второй ребёнок — не раньше чем через 15 месяцев (M14)."""

    gap = params_module.active().lifecycle.partner_age_gap
    births = 0
    for persona, events, _ in lives:
        dates = [item.ts for item in events if item.kind == "child_birth"]
        births += len(dates)
        assert len(dates) <= params_module.active().lifecycle.life_event_max["child_birth"]
        for earlier, later in zip(dates, dates[1:]):
            assert (later - earlier).days >= 456, persona
        for ts in dates:
            mother = persona.age_at(ts) - (gap if persona.gender == "M" else 0)
            assert 15 <= mother < 50, (persona.client_ordinal, mother)
    assert births > 50


def test_a_working_mother_goes_on_paid_leave(lives) -> None:
    """Декрет: зарплата встаёт за 70 дней до родов, ГФСС платит 126 дней разом и
    40% до полутора лет, после выхода — тот же работодатель (M9, M15)."""

    from src.generator.life import income

    settings = params_module.active().income
    leaves = 0
    for persona, events, streams in lives:
        for birth in (item for item in events if item.kind == "child_birth"):
            tag = birth.ts.toordinal()
            by_name = {item.stream_id.rsplit("_", 2)[-2]: item for item in streams if item.stream_id.endswith(f"_{tag}")}
            # Пособие при рождении и декрет — у матери; у отца доход не меняется.
            if persona.gender == "M":
                assert not by_name, persona.client_ordinal
                continue
            assert "birth" in by_name, persona.client_ordinal
            if "maternity" not in by_name:
                continue
            leaves += 1
            leave = birth.ts - timedelta(days=settings.maternity_days_before)
            salary = [item for item in streams if item.kind == "salary" and item.valid_to == leave]
            assert len(salary) == 1, persona.client_ordinal
            assert by_name["maternity"].payer == "state_benefit"
            care = by_name.get("childcare")
            if care is not None:
                assert care.payer == "state_benefit"
                assert care.valid_to <= birth.ts + timedelta(days=settings.childcare_days)
                # 40% дохода против 126 дней (4.1 месяца) разом.
                ratio = by_name["maternity"].base_amount / care.base_amount
                assert abs(ratio - settings.maternity_days / 30.4 / settings.childcare_share) < 0.01
            back = by_name.get("return")
            if back is not None:
                assert back.payer == salary[0].payer and back.kind == "salary"
            # Зарплаты в декрете нет.
            paid = income.payouts(persona, streams, ())
            end = back.valid_from if back is not None else datetime.max
            assert not [item for item in paid if item.kind == "salary" and leave <= item.ts < end
                        and item.outcome not in ("bonus", "annual_bonus")], persona.client_ordinal
    assert leaves > 5


def test_state_pays_on_schedule_in_full(lives) -> None:
    """Пенсия 1–15 числа по дате рождения, в срок и целиком, не ниже минимальной,
    индексация с января одна на всех; пособие по безработице — не дольше 6 месяцев (M9)."""

    from src.generator.life import income

    settings = params_module.active().income
    checked = 0
    for persona, _, streams in lives:
        for stream in streams:
            if stream.kind == "social_benefit" and "_benefit_" in stream.stream_id:
                assert (stream.valid_to - stream.valid_from).days <= 6 * 30
        pensions = [item for item in streams if item.kind == "pension"]
        if not pensions:
            continue
        paid = [item for item in income.payouts(persona, streams, ()) if item.kind == "pension"]
        by_month = defaultdict(list)
        for item in paid:
            assert item.outcome == "on_time" and item.payer == "state_pension"
            assert item.ts.day <= 15 and cal.is_business_day(item.ts)
            by_month[(item.ts.year, item.ts.month)].append(item.amount)
        for (year, month), amounts in by_month.items():
            assert len(amounts) == 1
            assert amounts[0] >= settings.state_pension_floor
        if (2024, 12) in by_month and (2025, 1) in by_month:
            ratio = by_month[(2025, 1)][0] / by_month[(2024, 12)][0]
            assert abs(ratio - (1 + settings.pension_indexation[2025])) < 1e-4
            checked += 1
    assert checked > 50


def test_collection_follows_the_debt() -> None:
    """Коллекшн пишет в дни вех просрочки; должнику с DPD > 30 не продают (M17)."""

    from src.generator.behaviour import communications as comm

    from src.generator.life.persona import draw_persona

    persona = draw_persona(7)
    day = datetime(2024, 6, 3)

    for dpd in range(0, 200):
        contacts = comm.collection_contacts(persona, day, dpd, True)
        steps = params_module.active().activity.collection_dpd_steps
        assert len(contacts) == (1 if dpd in steps else 0)
        for contact in contacts:
            assert contact.purpose == "collection" and contact.ts.second < 60
            assert contact.template == ("PR_RESTRUCTURE" if dpd >= 60 else "PR_OVERDUE_HARD" if dpd >= 30 else "PR_OVERDUE_SOFT")

    weights = comm._campaign_weights(persona, day, frozenset(), frozenset({"cash_loan", "deposit"}), 45,
                                     True, 0.5, False, False, None)
    purposes = {comm.CAMPAIGN_BY_CODE[code].purpose for code in weights}
    assert not purposes & {"offer", "winback", "collection"}

    # После частичной оплаты DPD ниже 30, но пауза в предложениях идёт.
    weights = comm._campaign_weights(persona, day, frozenset(), frozenset({"cash_loan"}), 5,
                                     False, 0.0, False, False, None, True)
    assert not {comm.CAMPAIGN_BY_CODE[code].purpose for code in weights} & {"offer", "winback"}
    weights = comm._campaign_weights(persona, day, frozenset(), frozenset({"cash_loan"}), 0,
                                     False, 0.0, False, False, None, False)
    assert "offer" in {comm.CAMPAIGN_BY_CODE[code].purpose for code in weights}


def test_life_events_leave_spending(lives) -> None:
    """Свадьба, болезнь, ремонт, учёба и рождение — с тратами своих категорий (M15)."""

    from src.generator.life.events import event_purchases

    expected = {
        "wedding": {"clothing", "restaurant"}, "illness": {"pharmacy", "medical"},
        "renovation": {"home_goods"}, "education": {"education"}, "child_birth": {"kids", "pharmacy"},
    }
    seen = defaultdict(set)
    for persona, events, _ in lives[:1500]:
        for event in events:
            if event.kind not in expected:
                continue
            plan = event_purchases(persona, (event,))
            categories = {item[0] for items in plan.values() for item in items}
            seen[event.kind] |= categories
            first = min(plan) if plan else None
            assert first is None or first >= (event.ts - timedelta(days=180)).toordinal()
    for kind, categories in expected.items():
        assert categories <= seen[kind], (kind, seen[kind])


def test_a_later_planning_horizon_keeps_the_consequences_before_it(personas, monkeypatch) -> None:
    """События сразу за горизонтом дают строки до него (билет, декрет, покупки к
    рождению): поднятый PLANNING_END не меняет ни плана, ни денег раньше прежнего."""

    from src.generator import config
    from src.generator.life import income
    from src.generator.life.events import event_purchases, plan_events

    old = config.PLANNING_END
    sample = personas[:400]

    def snapshot():
        result = []
        for persona in sample:
            events = plan_events(persona)
            streams = income.build_streams(persona, events)
            paid = [item for item in income.payouts(persona, streams, ()) if item.ts < old]
            spend = {day: items for day, items in event_purchases(persona, events).items() if day < old.toordinal()}
            result.append(([item for item in events if item.ts < old], paid, spend))
        return result

    before = snapshot()
    monkeypatch.setattr(config, "PLANNING_END", old + timedelta(days=365))
    after = snapshot()
    assert before == after


def test_a_leaving_client_plans_the_exit() -> None:
    """Уход: вывод остатка — в день ухода и только тогда; возврат с разгоном (M23)."""

    from types import SimpleNamespace

    from src.generator.behaviour import engagement

    settings = params_module.active().engagement
    day = datetime(2024, 5, 10)
    withdrawn = 0
    for ordinal in range(1, 401):
        state = SimpleNamespace(ordinal=ordinal, client_id=f"c{ordinal}", today=day,
                                engagement=engagement.Engagement(sensitivity={}, frailty=1.0, salary_threshold=0.5))
        engagement._lapse(state, day, "migration")
        plan = engagement.exit_today(state, day)
        withdrawn += plan is not None
        assert engagement.exit_today(state, day + timedelta(days=1)) is None
        assert engagement.transit(state) == state.engagement.transit
    assert 0.6 < withdrawn / 400 < 0.9

    state.engagement.regime = engagement.ACTIVE
    state.engagement.returned_at = day
    floor, days = settings.return_ramp
    assert engagement._ramp(state) == pytest.approx(floor)
    state.today = day + timedelta(days=days)
    assert engagement._ramp(state) == 1.0


# ============================================================
# ВОЛНА 6: РИТМ (C3, M1, M2, M18, M20, M21)
# ============================================================


def test_an_early_salary_closes_its_cycle() -> None:
    """Зарплата, пришедшая раньше срока, закрывает свой срок: следующую клиент
    ждёт через цикл, а не в плановый день через 1–3 дня (C3)."""

    from types import SimpleNamespace

    from src.generator import engine
    from src.generator.life.income import IncomeStream

    stream = IncomeStream("inc_1_1", "salary", "emp_1", "monthly", 10, "hcb_account", 300_000,
                          datetime(2020, 1, 1), None)
    state = SimpleNamespace(income_streams=(stream,))

    assert engine._next_payday(state, datetime(2024, 5, 9), datetime(2024, 5, 8, 11)) == datetime(2024, 6, 10)
    assert engine._next_payday(state, datetime(2024, 5, 9), datetime(2024, 4, 10, 11)) == datetime(2024, 5, 10)


def test_a_day_repeats_neither_the_cinema_nor_the_commute(world) -> None:
    """Кино — раз в день, проезд — не больше трёх раз (M1)."""

    per_day = defaultdict(int)
    for item in world["events"]:
        if item["type"] == "purchase" and item.get("status") == "approved" \
                and item.get("merchant_category") in ("cinema", "transit"):
            per_day[(item["client_id"], item["ts"].date(), item["merchant_category"])] += 1
    assert per_day
    for (_, _, category), count in per_day.items():
        assert count <= (1 if category == "cinema" else 3), category


def test_hours_follow_the_client_and_the_category() -> None:
    """Сова живёт позже жаворонка, кофе пьют утром, в бар ходят вечером (M2)."""

    def mean_hour(weights):
        return sum(hour * value for hour, value in enumerate(weights))

    day = datetime(2024, 4, 10)
    assert mean_hour(cal.hour_weights(day, False, 2)) - mean_hour(cal.hour_weights(day, False, -2)) > 2.5
    assert mean_hour(cal.hour_weights(day, False, 0, "coffee")) + 4 < mean_hour(cal.hour_weights(day, False, 0, "bars"))


def test_support_works_in_channel_hours(world) -> None:
    """Отделение принимает и закрывает обращения в 9–18, колл-центр — в 8–22;
    13:20:00 у всех больше нет (M20)."""

    hours = params_module.active().activity.support_channel_hours
    cases = [item for item in world["events"] if item["type"] in ("case_opened", "case_resolved")]
    assert cases
    for item in cases:
        low, high = hours[item["channel"]]
        assert low <= item["ts"].hour < high, item
    opened = [item["ts"] for item in cases if item["type"] == "case_opened"]
    assert sum(ts.hour == 13 and ts.minute == 20 and ts.second == 0 for ts in opened) <= 1


def test_card_statements_spread_over_the_month() -> None:
    """Выписка — в свой день договора, а не у всех карт в конце месяца (M18)."""

    from src.generator.engine_month import statement_day

    days = {statement_day(f"ctr_{index}") for index in range(400)}
    assert days == set(range(1, 29))


def test_taxes_are_paid_once_a_year_by_the_deadline(personas) -> None:
    """Налог на имущество и транспорт — раз в год, у владельца, к 1 октября (M18)."""

    from types import SimpleNamespace

    from src.generator import engine

    owners = 0
    for persona in personas[:600]:
        state = SimpleNamespace(persona=persona, ordinal=persona.client_ordinal)
        day = engine._tax_day(state, 2024)
        owner = persona.transport == "car" or persona.housing_type in ("own_apartment", "own_house")
        assert (day is None) == (not owner)
        if day is not None:
            owners += 1
            assert date(2024, 8, 1).toordinal() <= day <= date(2024, 12, 31).toordinal()
    assert owners > 100


def test_client_actions_have_seconds(world) -> None:
    """Секунда :00 — не чаще, чем у случайного момента (M21)."""

    kinds = ("purchase", "transfer_out", "p2p_out", "cash_withdrawal", "bill_payment", "cash_deposit")
    moments = [item["ts"] for item in world["events"] if item["type"] in kinds]
    assert len(moments) > 500
    assert sum(ts.second == 0 for ts in moments) / len(moments) < 0.05


# ============================================================
# ВОЛНА 7: МЕЛКИЕ НАХОДКИ АУДИТА
# ============================================================


def test_a_newcomer_profile_describes_the_day_of_arrival(personas, lives) -> None:
    """Анкета пришедшего внутри окна — на день прихода: переезд и свадьба до
    регистрации уже в ней."""

    from types import SimpleNamespace

    from src.generator import engine
    from src.generator.simulate import CommunitySimulation
    from src.generator.world import geography

    checked = 0
    for persona, events, streams in lives:
        if not persona.registered_in_window:
            continue
        before = [item for item in events if item.ts < persona.relationship_start and item.kind in ("move", "wedding")]
        if not before:
            continue
        state = SimpleNamespace(persona=persona, ordinal=persona.client_ordinal, life_events=events, income_streams=streams,
                                profile_values=CommunitySimulation._initial_profile(None, persona))
        engine._profile_at_registration(state)
        moves = [item for item in before if item.kind == "move"]
        if moves:
            expected = geography.by_name(moves[-1].payload["settlement"]).public_name
            assert state.profile_values["city"] == expected
        weddings = [item for item in before if item.kind == "wedding"]
        if weddings:
            assert state.profile_values["family_status"] == weddings[-1].payload["family_status_after"]
        checked += 1
    assert checked > 3


def test_activations_and_refunds_happen_in_the_daytime(world) -> None:
    """Возврат и отмена — не ночью, активация карты — почти всегда днём."""

    backs = [item for item in world["events"] if item["type"] in ("refund", "reversal")]
    assert backs
    assert all(item["ts"].hour >= 7 for item in backs)
    activations = [item["ts"] for item in world["events"] if item["type"] == "card_activated"]
    assert activations
    assert sum(ts.hour < 7 for ts in activations) / len(activations) < 0.05


def test_daily_limits_hold(world) -> None:
    """Покупки по карте и переводы людям за день — не выше дневного лимита."""

    products = params_module.active().products
    cards, transfers = defaultdict(int), defaultdict(int)
    for item in world["events"]:
        if item.get("status") != "approved":
            continue
        if item["type"] == "purchase" and item.get("card_id"):
            cards[(item["card_id"], item["ts"].date())] += int(item["amount"])
        elif item["type"] in ("transfer_out", "p2p_out") and item.get("counterparty") != "Own account":
            transfers[(item["client_id"], item["ts"].date())] += int(item["amount"])
    assert cards and transfers
    assert max(cards.values()) <= products.card_purchase_limit_per_day
    assert max(transfers.values()) <= products.transfer_limit_per_day


def test_an_app_transfer_belongs_to_its_session(world) -> None:
    """Перевод из приложения — внутри сессии со входом и операцией перевода."""

    sessions = defaultdict(set)
    for item in world["events"]:
        if item["type"] == "app_operation":
            sessions[item["session_id"]].add(item["operation"])
    app = [item for item in world["events"] if item["type"] in ("transfer_out", "p2p_out") and item.get("session_id")]
    assert app
    for item in app:
        operations = sessions.get(item["session_id"], set())
        assert operations & {"login", "biometry_login"}, item
        assert operations & {"transfer_phone", "transfer_card", "transfer_own", "transfer_template"}, item


def test_app_payments_have_something_to_pay(world) -> None:
    """Оплата в приложении — только по выставленному счёту и в основном проходит."""

    pays = [item for item in world["events"] if item["type"] == "app_operation" and str(item["operation"]).startswith("pay_")]
    assert pays
    assert sum(item["status"] == "success" for item in pays) / len(pays) > 0.6


def test_early_deposit_closures_happen_on_any_day(world) -> None:
    """Досрочно вклад закрывают днём в любой день, а не в 23:52 последнего дня."""

    early = [item for item in world["events"] if item["type"] == "product_closed" and item.get("reason") == "early_closure"]
    for item in early:
        assert 10 <= item["ts"].hour <= 20, item
