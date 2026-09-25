from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from src.generator import emit
from src.generator.config import EVENT_TYPES
from src.generator.profile import (
    LIFELONG_SOURCE_EVENTS,
    LIFELONG_SOURCE_FIELD,
    LIFELONG_TYPES,
    lifelong,
    utc,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Вехи анкеты и записи о работе — производные от того, что
# генератор уже прожил, а не новые даты:
#
#   bank_registered       Persona.relationship_start;
#   app_registered        app_adoption клиента, если она была;
#   first_card_activated  самая ранняя активация карты договора —
#                         перевыпуск новой активацией не считается;
#   first_loan_opened     самое раннее открытие кредита с графиком;
#   first_deposit_opened  самое раннее открытие вклада.
#
# У вех о продуктах есть ссылка source_id на саму карту или
# договор. Внутри окна по ней в ленте находится акт источника —
# строки типов-источников с тем же идентификатором и тем же
# временем, до окна — ничего. Первая активация не позже любой
# активации ленты. Все вехи и записи строго раньше as_of.
#
# Даты персоны снимаются сразу после выгрузки, в том же состоянии
# розыгрыша: так сравниваются с тем, по чему жила симуляция.
# ============================================================


CLIENTS = 8
COMMUNITY = 4

START = datetime(2024, 1, 1)
END = datetime(2024, 7, 1)


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> dict:

    from src.generator import params as params_module
    from src.generator.life.persona import app_adoption, draw_persona
    from src.generator.world import communities

    out = tmp_path_factory.mktemp("lifelong")

    emit.generate_dataset(
        total_clients=CLIENTS,
        out_dir=out,
        seed=100,
        world_seed=42,
        history_start=START,
        history_end=END,
        workers=1,
        community_size=COMMUNITY,
        quiet=True,
    )

    salaried = params_module.active().income.primary_kind_by_income_type

    personas = {}

    for ordinal in communities.client_ordinals(CLIENTS):
        persona = draw_persona(ordinal)
        personas[persona.client_id] = SimpleNamespace(
            start=persona.relationship_start,
            adopted=app_adoption(ordinal),
            salaried=salaried.get(persona.income_type, "salary") == "salary",
        )

    return {
        "profile": pq.read_table(out / "profile.parquet").to_pylist(),
        "events": pq.read_table(out / "events.parquet").to_pylist(),
        "personas": personas,
    }


def items(row: dict) -> dict[str, datetime]:
    return {item["type"]: item["event_time"] for item in row["lifelong"]}


def rows(run: dict, client_id: str, *types: str) -> list[tuple[datetime, dict]]:
    """
    События клиента этих типов: момент в UTC и payload.
    """

    import json

    out = []

    for event in run["events"]:

        payload = json.loads(event["payload"])

        if event["client_id"] == client_id and payload["type"] in types:
            out.append((datetime.fromisoformat(event["event_time"]).astimezone(timezone.utc), payload))

    return out


def tape(run: dict, client_id: str, *types: str) -> list[datetime]:
    """
    Моменты событий клиента этих типов, в UTC.
    """

    return [moment for moment, _ in rows(run, client_id, *types)]


# ============================================================
# ВЕХИ ПО ФАКТАМ ПЕРСОНЫ
# ============================================================


def test_as_of_is_the_end_of_the_window(run):

    for row in run["profile"]:
        assert row["as_of"] == utc(END), row["client_id"]


def test_bank_registered_is_the_persona_start(run):

    for row in run["profile"]:
        start = run["personas"][row["client_id"]].start
        assert items(row)["bank_registered"] == utc(start), row["client_id"]


def test_app_registered_only_where_the_app_was_opened(run):
    """
    Веха есть ровно у тех, кому приложение открылось раньше as_of,
    и дата у неё та самая.
    """

    seen = {True: 0, False: 0}

    for row in run["profile"]:

        adopted = run["personas"][row["client_id"]].adopted
        found = items(row)

        expected = adopted is not None and utc(adopted) < row["as_of"]

        seen[expected] += 1

        if expected:
            assert found["app_registered"] == utc(adopted), row["client_id"]
        else:
            assert "app_registered" not in found, row["client_id"]

    assert seen[True] and seen[False], seen


def test_milestones_are_before_as_of_unique_and_in_order(run):

    for row in run["profile"]:

        order = [
            (item["event_time"], LIFELONG_TYPES.index(item["type"])) for item in row["lifelong"]
        ]

        assert order == sorted(order), row["client_id"]
        assert all(moment < row["as_of"] for moment, _ in order), row["client_id"]
        assert len({item["type"] for item in row["lifelong"]}) == len(order), row["client_id"]


def test_bank_registration_before_the_window_survives(run):
    """
    Клиент, пришедший до начала выгрузки: событий того времени в
    ленте нет, а веха о приходе есть.
    """

    start = utc(START)

    old = [row for row in run["profile"] if items(row)["bank_registered"] < start]

    assert old, "проверка вырождена: все клиенты пришли в окне"

    for row in old:
        assert all(moment >= start for moment in tape(run, row["client_id"], *EVENT_TYPES))


def test_milestones_are_not_event_types():
    """
    Вехи — факты анкеты, а не события ленты: своих строк в ленте у
    них нет, есть только события-источники других типов.
    """

    assert not set(LIFELONG_TYPES) & set(EVENT_TYPES)

    for sources in LIFELONG_SOURCE_EVENTS.values():
        assert set(sources) <= set(EVENT_TYPES)


# ============================================================
# ВЕХИ О ПРОДУКТАХ И ЛЕНТА
# ============================================================


def test_first_card_activation_is_the_earliest_of_the_tape(run):
    """
    Никакая активация карты в ленте не раньше вехи, а веха внутри
    окна — сама одна из этих активаций.
    """

    found = 0

    for row in run["profile"]:

        milestone = items(row).get("first_card_activated")

        if milestone is None:
            continue

        found += 1

        moments = tape(run, row["client_id"], "card_activated")

        assert all(moment >= milestone for moment in moments), row["client_id"]

        if milestone >= utc(START):
            assert milestone in moments, row["client_id"]

    assert found, "проверка вырождена: карт нет ни у кого"


@pytest.mark.parametrize("kind", ["first_loan_opened", "first_deposit_opened"])
def test_first_product_inside_the_window_is_an_opening_of_the_tape(run, kind: str):

    for row in run["profile"]:

        milestone = items(row).get(kind)

        if milestone is None or milestone < utc(START):
            continue

        assert milestone in tape(run, row["client_id"], "product_opened", "product_migrated")


def test_product_milestones_link_to_their_source_act(run):
    """
    Ссылка вехи находит в ленте ровно акт своего источника: внутри
    окна — строки типов-источников с этим идентификатором, все в
    момент вехи (карта — одна активация, кредит — одно открытие,
    вклад — счёт и договор); до окна — ни одной. У прихода в банк и
    приложения ссылки нет.
    """

    inside = {kind: 0 for kind in LIFELONG_SOURCE_FIELD}

    for row in run["profile"]:

        for item in row["lifelong"]:

            kind = item["type"]

            if kind not in LIFELONG_SOURCE_FIELD:
                assert item["source_id"] is None, (row["client_id"], kind)
                continue

            assert item["source_id"], (row["client_id"], kind)

            field = LIFELONG_SOURCE_FIELD[kind]

            act = [
                (moment, payload["type"])
                for moment, payload in rows(run, row["client_id"], *LIFELONG_SOURCE_EVENTS[kind])
                if payload.get(field) == item["source_id"]
            ]

            if item["event_time"] < utc(START):
                assert act == [], (row["client_id"], kind)
                continue

            inside[kind] += 1

            assert act and all(moment == item["event_time"] for moment, _ in act), (row["client_id"], kind)

            types = sorted(name for _, name in act)

            if kind == "first_card_activated":
                assert types == ["card_activated"], (row["client_id"], types)
            elif kind == "first_loan_opened":
                assert len(types) == 1, (row["client_id"], types)
            else:
                assert types[0] == "account_opened" and len(types) == 2, (row["client_id"], types)

    assert all(inside.values()), f"проверка вырождена: {inside}"


# ============================================================
# ЗАПИСИ О РАБОТЕ
# ============================================================


def test_employment_records_are_the_banks_view(run):
    """
    У работающего по найму первая запись — с начала отношений, и
    работа началась не позже неё. У остальных записей нет. Все
    записи раньше as_of и по времени.
    """

    for row in run["profile"]:

        persona = run["personas"][row["client_id"]]
        records = row["employment"]

        if not persona.salaried:
            assert records == [], row["client_id"]
            continue

        assert records, row["client_id"]
        assert records[0]["record_time"] == utc(persona.start), row["client_id"]
        assert records[0]["start_date"] <= persona.start.date(), row["client_id"]

        moments = [item["record_time"] for item in records]

        assert moments == sorted(moments), row["client_id"]
        assert all(moment < row["as_of"] for moment in moments), row["client_id"]


def test_job_start_is_isolated_and_repeatable():
    """
    Начало первой работы разыгрывается в своём пространстве: номер
    не делит ни один другой розыгрыш, и повторный вызов даёт ту же
    дату.
    """

    from src.generator import rng
    from src.generator.life.income import job_start
    from src.generator.life.persona import draw_persona

    numbers = [value for name, value in vars(rng).items() if name.startswith("NS_")]

    assert numbers.count(rng.NS_EMPLOYMENT) == 1

    persona = draw_persona(1)

    first = job_start(persona, START)

    assert first == job_start(persona, START)
    assert first <= START.date()
    assert first >= (persona.birth_date + timedelta(days=int(18 * 365.25))).date() - timedelta(days=31)


# ============================================================
# ВЫВОД ВЕХ ИЗ СОСТОЯНИЯ
# ============================================================
#
# Подставное состояние с руками написанными договорами и картами:
# так проверяется сам выбор «первого», включая предысторию, которой
# в ленте нет.
# ============================================================


def card(activated: datetime | None, reissued_from: str | None = None):
    return SimpleNamespace(activated_at=activated, reissued_from=reissued_from)


def contract(family: str, opened: datetime):
    return SimpleNamespace(product_family=family, opened_at=opened)


def state(cards: list, contracts: list, adopted: datetime | None = None):
    """
    Карты и договоры в порядке выдачи: номер — их идентификатор.
    """

    for number, item in enumerate(cards):
        item.card_id = f"crd_{number}"

    for number, item in enumerate(contracts):
        item.contract_id = f"ctr_{number}"

    return SimpleNamespace(
        persona=SimpleNamespace(relationship_start=datetime(2017, 3, 1)),
        app_adopted_at=adopted,
        cards={item.card_id: item for item in cards},
        contracts={item.contract_id: item for item in contracts},
    )


def milestones(value):

    import src.generator.engine  # noqa: F401 — engine_month грузится через engine

    from src.generator.engine_month import milestones as derive

    return derive(value)


def test_first_card_skips_reissues_and_unactivated_cards():

    found = milestones(state(
        cards=[
            card(datetime(2024, 5, 1)),
            card(datetime(2023, 2, 1), reissued_from="c0"),
            card(None),
            card(datetime(2024, 3, 10)),
        ],
        contracts=[],
    ))

    assert found["first_card_activated"] == (datetime(2024, 3, 10), "crd_3")


def test_first_loan_counts_scheduled_credit_only():
    """
    Кредит — договор с графиком. Кредитная карта — карта, а вклад —
    вклад; ни то, ни другое первым кредитом не становится.
    """

    found = milestones(state(
        cards=[],
        contracts=[
            contract("credit_card", datetime(2019, 1, 1)),
            contract("deposit", datetime(2019, 6, 1)),
            contract("installment", datetime(2022, 4, 1)),
            contract("cash_loan", datetime(2021, 8, 1)),
            contract("refinance", datetime(2023, 1, 1)),
        ],
    ))

    assert found["first_loan_opened"] == (datetime(2021, 8, 1), "ctr_3")
    assert found["first_deposit_opened"] == (datetime(2019, 6, 1), "ctr_1")


def test_first_of_two_products_at_one_moment_is_the_one_issued_first():
    """
    Две карты активированы и два кредита открыты в один момент:
    источник вехи — выданные раньше, а не «любой того же времени».
    """

    moment = datetime(2024, 5, 1, 12)

    found = milestones(state(
        cards=[card(moment), card(moment)],
        contracts=[contract("cash_loan", moment), contract("installment", moment)],
    ))

    assert found["first_card_activated"] == (moment, "crd_0")
    assert found["first_loan_opened"] == (moment, "ctr_0")


def test_client_without_products_has_no_product_milestones():

    found = milestones(state(cards=[], contracts=[contract("debit_card", datetime(2020, 1, 1))]))

    assert found["first_loan_opened"] is None
    assert found["first_deposit_opened"] is None
    assert found["first_card_activated"] is None
    assert found["bank_registered"] == (datetime(2017, 3, 1), None)
    assert found["app_registered"] is None


def test_milestones_draw_nothing(monkeypatch):
    """
    Ни вывод вех, ни их запись не трогают случайность: любой
    розыгрыш здесь упал бы.
    """

    from src.generator import rng

    def forbidden(*args, **kwargs):
        raise AssertionError("вехи не имеют права разыгрывать")

    monkeypatch.setattr(rng.KeyedRandom, "random", forbidden)

    found = milestones(state(
        cards=[card(datetime(2024, 5, 1))],
        contracts=[contract("cash_loan", datetime(2024, 2, 1))],
        adopted=datetime(2024, 1, 20),
    ))

    written = lifelong(found, datetime(2024, 4, 1))

    assert [(item["type"], item["source_id"]) for item in written] == [
        ("bank_registered", None), ("app_registered", None), ("first_loan_opened", "ctr_0"),
    ]


def test_milestone_exactly_at_as_of_is_left_out():
    """
    Граница полуоткрытая, как у событий: веха в самый момент as_of
    снимку ещё не известна.
    """

    boundary = datetime(2024, 4, 1)
    before = boundary - timedelta(microseconds=1)

    assert lifelong({"bank_registered": (boundary, None)}, boundary) == []

    assert lifelong(
        {"bank_registered": (before, None), "first_card_activated": (before, "crd_0")}, boundary
    ) == [
        {"type": "bank_registered", "event_time": utc(before), "source_id": None},
        {"type": "first_card_activated", "event_time": utc(before), "source_id": "crd_0"},
    ]

    with pytest.raises(ValueError, match="вне контракта"):
        lifelong({"kyc_passed": (before, None)}, boundary)


def test_local_time_becomes_utc():

    assert utc(datetime(2024, 1, 1)) == datetime(2023, 12, 31, 19, tzinfo=timezone.utc)
