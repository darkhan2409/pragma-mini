from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime

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
