from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

from src.generator.config import key_catalogue
from src.preprocessing.canonical.build import build_group as build_canonical
from src.preprocessing.canonical.registry import catalogue_from_registry
from src.preprocessing.manifest import fingerprint_path
from src.preprocessing.history import CanonicalStore
from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK
from src.preprocessing.run import main as run_main
from src.preprocessing.semantic.activity import HAS_CLIENT_ACTION, INSUFFICIENT_COVERAGE, NO_RECORDS
from src.preprocessing.semantic.as_of import open_products, semantic_as_of
from src.preprocessing.semantic.chains import TIME_ORDER_AMBIGUOUS, ChainsError
from src.preprocessing.semantic.build import REGISTRY_FILE, build_group
from src.preprocessing.semantic.keys import (
    DERIVED_KEYS,
    TIMING_KEYS,
    KeysError,
    key_for,
    keys_registry,
    profile_change_keys,
    validate_keys,
)
from src.preprocessing.settings import PreprocessingConfig

from tests.prep_fixtures import MiniRaw, purchase_payload  # noqa: F401


CONFIG = PreprocessingConfig()

FULL_HORIZON = datetime(2023, 1, 1)


def _cli(*args: str) -> int:
    with pytest.raises(SystemExit) as result:
        run_main(list(args))
    return int(result.value.code)


def _store(raw_dir: Path, out_dir: Path) -> CanonicalStore:
    build_canonical(raw_dir, out_dir, CONFIG, "train")
    return CanonicalStore(out_dir, products=open_products(raw_dir))


def _client(mini: MiniRaw, client_id: str = "c1", income: int | None = 75000) -> None:
    """
    Клиент с покрытием с начала окна и версией профиля.
    """

    mini.cover_all(client_id, first_seen="2023-01-01")

    if income is not None:
        mini.profile_version(client_id, 1, "2023-01-01", declared_income=income, age=40)


# ============================================================
# СМЫСЛЫ
# ============================================================


def test_every_allowed_field_has_one_meaning_and_disputes_are_listed():
    """
    Каждое поле, прошедшее модельную проекцию, сопоставлено ровно
    одному смыслу; одинаковое имя в разных источниках разводится
    явно, а спорные объединения перечислены с причиной.
    """

    catalogue = key_catalogue()

    validate_keys(catalogue)

    registry = keys_registry(catalogue)

    # Одно имя, разные источники — разные смыслы.
    assert key_for("channel", "transactions").key == "operation_channel"
    assert key_for("channel", "communications").key == "communication_channel"
    assert key_for("status", "support").key == "case_status"
    assert key_for("decision", "antifraud").key == "fraud_decision"

    # Одно имя, один смысл — объединение объявлено явно.
    assert key_for("amount", "transactions").key == key_for("amount", "app_operations").key
    assert any(item["key"] == "transaction_amount" for item in registry["allowed_sharing"])

    # Продукт, семейство и оффер не смешаны.
    kinds = {name: item["value_kind"] for name, item in registry["keys"].items()}
    assert {"product_code", "product_family", "offer"} <= set(kinds)
    assert kinds["mcc"] == "categorical", "цифровой код это категория, а не величина"
    assert kinds["transaction_amount"] == "numeric"
    assert kinds["merchant_name"] == "text"

    disputed = {key for item in registry["ambiguous"] for key in item["keys"]}
    assert {"product_code", "product_family", "offer"} <= disputed
    assert {"operation_channel", "communication_channel"} <= disputed

    # Незнакомое поле смыслом не наделяется само собой.
    with pytest.raises(KeysError):
        key_for("внезапное_поле", "transactions")


# ============================================================
# ПРИЗНАКИ
# ============================================================


def test_ratios_keep_original_values_and_name_the_reason(tmp_path):
    """
    Отношение считается из исходных чисел, а неизвестный
    знаменатель остаётся причиной, а не нулём.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    _client(mini, "c1", income=75000)
    _client(mini, "c2", income=None)

    for client_id in ("c1", "c2"):
        mini.event(client_id, "purchase", "2023-03-05 10:00:00", payload=purchase_payload(amount=12500))

    store = _store(mini.write(), tmp_path / "canonical")

    known = semantic_as_of(store, "c1", datetime(2023, 4, 1))
    unknown = semantic_as_of(store, "c2", datetime(2023, 4, 1))

    # Значение осталось исходным: ни корзин, ни нормализации.
    assert known.events[0].values["transaction_amount"] == 12500
    assert known.profile["profile_declared_income"] == 75000

    ratio = next(item for item in known.events[0].derived if item.key == "amount_to_declared_income")

    assert ratio.value == pytest.approx(12500 / 75000, abs=1e-9)
    assert ratio.reason is None

    # Происхождение названо: событие и версия профиля.
    kinds = {item["kind"] for item in ratio.derived_from}
    assert kinds == {"event", "profile"}

    # Дохода не знаем — причина, а не ноль.
    missing = next(item for item in unknown.events[0].derived if item.key == "amount_to_declared_income")

    assert missing.value is None
    assert missing.reason == "income_unknown"


def test_dates_and_derived_features_are_declared_keys(tmp_path):
    """
    Отдельного вида «дата» нет: плановая дата остаётся в
    canonical, а модель получает число дней до неё. Расчётные и
    временные признаки объявлены такими же ключами, как
    физические поля, и проверка реестра их видит.
    """

    registry = keys_registry(key_catalogue())

    assert "date" not in registry["counts"]["by_value_kind"]
    assert "due_date" not in registry["keys"], "плановая дата словарём не кодируется"

    # Каждый расчётный и временной признак объявлен и называет,
    # из чего он получен.
    for key in (*DERIVED_KEYS.values(), *TIMING_KEYS.values()):

        declared = registry["keys"][key.key]

        assert declared["value_kind"] == "numeric"
        assert declared["unit"]
        assert declared["derived_from"], f"{key.key} не называет, из чего посчитан"

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    _client(mini, "c1", income=75000)

    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload(amount=12500))
    mini.event(
        "c1", "installment_due", "2023-03-10 09:00:00",
        payload={
            "contract_id": "ctr_1", "installment_no": 2, "amount_due": 40000,
            "amount_paid": None, "principal_outstanding": 300000, "days_past_due": 0,
            "due_date": "2023-04-10", "cause_event_id": None, "reason": "schedule",
        },
    )

    history = semantic_as_of(_store(mini.write(), tmp_path / "canonical"), "c1", datetime(2023, 4, 1))

    schedule = next(item for item in history.events if item.values["event_type"] == "installment_due")

    assert "due_date" not in schedule.values
    # Источник loans объявляет дневную точность: дни до платежа
    # считаются по датам, а не по часу, которого источник не знает.
    assert schedule.timing.days_to_due == 31.0

    # Признак доходит до модели под объявленным ключом.
    model_values = schedule.model_values()

    assert model_values["days_to_due"] == schedule.timing.days_to_due
    assert "time_precision" not in model_values, "признак качества времени модели не передаётся"

    purchase = next(item for item in history.events if item.values["event_type"] == "purchase")

    assert purchase.model_values()["amount_to_declared_income"] == pytest.approx(12500 / 75000, abs=1e-9)

    declared = set(registry["keys"])

    for item in history.events:
        assert set(item.model_values()) <= declared, "значение без объявленного ключа наружу не выходит"


def test_product_and_relation_meanings_replace_identifiers(tmp_path):
    """
    Продукт расшифрован по справочнику, а событие-следствие несёт
    смысл связи вместо cause_event_id. Сырые идентификаторы
    наружу не выходят.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    _client(mini, "c1")

    mini.product("prd_card", "CARD_GOLD", "Карта Халык Bonus", product_family="card")

    mini.event(
        "c1", "product_opened", "2023-02-01 10:00:00",
        payload={
            "product_id": "prd_card", "product_code": "CARD_GOLD", "product_version": 1,
            "tariff_version": 1, "product_family": "card", "contract_id": "ctr_1",
            "account_id": "acc_1", "card_id": "crd_1", "offer_id": None,
            "previous_product_id": None, "migration_reason": None, "amount_or_limit": 500000,
            "term": None, "rate": None, "reason": "application", "timestamp_quality": None,
        },
    )

    purchase = mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload(amount=12500))

    mini.event(
        "c1", "refund", "2023-03-07 10:00:00",
        payload=purchase_payload(amount=12500, direction="credit", reason="refund", cause_event_id=purchase),
    )

    history = semantic_as_of(_store(mini.write(), tmp_path / "canonical"), "c1", datetime(2023, 4, 1))

    opened = next(item for item in history.events if item.values["event_type"] == "product_opened")

    assert opened.values["product_name"] == "Карта Халык Bonus"
    assert opened.values["product_code"] == "CARD_GOLD"
    assert "prd_card" not in set(opened.values.values()), "идентификатор каталога наружу не выходит"

    consequence = next(item for item in history.events if item.values["event_type"] == "refund")

    assert consequence.values["related_event_type"] == "purchase"
    assert consequence.values["relation_type"] == "refund"
    assert consequence.values["days_since_related_event"] == pytest.approx(2.0, abs=1e-9)
    assert consequence.values["same_merchant"] is True

    # Связь прикреплена к следствию, а не к причине, и адресуется
    # внутренним номером логического события.
    cause = next(item for item in history.events if item.values["event_type"] == "purchase")

    assert "related_event_type" not in cause.values
    assert [item.stable_event_index for item in history.relations] == [consequence.stable_event_index]

    for item in history.events:
        assert not any(str(value).startswith("ev0") for value in item.values.values())


def test_profile_change_keeps_the_meaning_of_the_changed_field(tmp_path):
    """
    old_value и new_value наследуют смысл изменившегося поля:
    доход остаётся числом, город категорией. Одним текстовым
    ключом их объявить нельзя.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    _client(mini, "c1")

    mini.event(
        "c1", "profile_change", "2023-02-01 10:00:00",
        payload={"field_name": "declared_income", "old_value": "500000", "new_value": "650000",
                 "change_source": "application", "confirmed": True},
    )
    mini.event(
        "c1", "profile_change", "2023-02-02 10:00:00",
        payload={"field_name": "city", "old_value": "Алматы", "new_value": "Астана",
                 "change_source": "client", "confirmed": True},
    )

    history = semantic_as_of(_store(mini.write(), tmp_path / "canonical"), "c1", datetime(2023, 4, 1))

    income, city = history.events[0].values, history.events[1].values

    assert income["profile_declared_income_old"] == 500000
    assert income["profile_declared_income_new"] == 650000
    assert city["profile_city_new"] == "Астана"

    # Общего текстового ключа на все изменения больше нет.
    assert "old_value" not in income and "new_value" not in income

    registry = keys_registry(key_catalogue())

    assert registry["keys"]["profile_declared_income_new"]["value_kind"] == "numeric"
    assert registry["keys"]["profile_declared_income_new"]["unit"] == "KZT"
    assert registry["keys"]["profile_city_new"]["value_kind"] == "categorical"

    # Незнакомое поле профиля текстом не становится.
    with pytest.raises(KeysError):
        profile_change_keys("внезапное_поле")


def test_backwards_relation_gives_a_reason_not_a_negative_feature(tmp_path):
    """
    Причина не может произойти после следствия. Если порядок
    объясняется объявленной точностью, длительность связи не
    передаётся, а причина названа. Если обе записи точны, это
    противоречие данных и этап отказывается.
    """

    def build(folder: str, precision: str):

        mini = MiniRaw(tmp_path / folder, history_start=FULL_HORIZON)
        _client(mini, "c1")

        # Причина записана на два часа позже своего следствия.
        cause = mini.event(
            "c1", "product_opened", "2023-03-05 12:00:00", precision=precision,
            payload={
                "product_id": None, "product_code": "LOAN", "product_version": 1,
                "tariff_version": 1, "product_family": "loan", "contract_id": "ctr_1",
                "account_id": "acc_1", "card_id": None, "offer_id": None,
                "previous_product_id": None, "migration_reason": None, "amount_or_limit": 500000,
                "term": 12, "rate": 0.21, "reason": "application",
                "timestamp_quality": "date_only" if precision == "day" else "exact",
            },
        )

        mini.event(
            "c1", "loan_disbursement", "2023-03-05 10:00:00",
            payload=purchase_payload(amount=500000, direction="credit", reason="disbursement",
                                     cause_event_id=cause),
        )

        return _store(mini.write(), tmp_path / f"{folder}_canonical")

    # Дневная точность: порядок внутри суток источнику неизвестен.
    history = semantic_as_of(build("raw_day", "day"), "c1", datetime(2023, 4, 1))

    relation = history.relations[0]

    assert relation.reason == TIME_ORDER_AMBIGUOUS
    assert relation.days_since_related_event is None
    assert relation.observed_days < 0, "наблюдение сохраняется как есть, а не обнуляется"

    consequence = next(item for item in history.events if item.values["event_type"] == "loan_disbursement")

    assert consequence.values["related_event_type"] == "product_opened"
    assert consequence.values["relation_type"] == "caused_by"
    assert "days_since_related_event" not in consequence.model_values()

    assert any(TIME_ORDER_AMBIGUOUS in item for item in history.limitations)

    # Точное время с обеих сторон: отрицательная причинность это
    # ошибка данных, а не особенность времени.
    store = build("raw_second", "second")

    with pytest.raises(ChainsError):
        semantic_as_of(store, "c1", datetime(2023, 4, 1))


def test_day_precision_event_has_no_known_hour(tmp_path):
    """
    У записи дневной точности час суток не наблюдался.

    Признак внутренний: токеном он не становится и в значения
    события не попадает. Но следующий этап обязан знать, что
    пара hour_sin/hour_cos у такой записи подставлена выгрузкой,
    иначе полуночный пик дневных строк будет прочитан как
    поведение клиента.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    _client(mini, "c1")

    mini.event(
        "c1", "purchase", "2023-03-05 14:20:00",
        payload=purchase_payload(amount=12000),
    )

    # Тот же вид записи, но источник объявил только дату.
    mini.event(
        "c1", "purchase", "2023-03-06 00:00:00", precision="day",
        payload=purchase_payload(amount=9000, timestamp_quality="date_only"),
    )

    store = _store(mini.write(), tmp_path / "canonical")

    history = semantic_as_of(store, "c1", datetime(2023, 4, 1))

    exact, coarse = [item for item in history.events if item.values["event_type"] == "purchase"]

    assert exact.hour_known is True
    assert coarse.hour_known is False

    # Признак достоверности не пересекает границу модели.
    assert "hour_known" not in coarse.model_values()
    assert "hour_known" not in coarse.values

    # Календарь по-прежнему считается у обеих записей: правило
    # адресовано следующему этапу, а не этому.
    assert len(coarse.calendar) == len(exact.calendar) > 0

    assert coarse.as_dict()["hour_known"] is False


def test_intervals_respect_declared_precision(tmp_path):
    """
    Интервал считается в объявленной точности пары событий.

    Дневная запись несёт в поле времени час, которого источник не
    знает: два дневных события в 10:00 и 11:00 одного дня давали
    since_previous_hours = 1.0, и модель училась на часе, которого
    никто не наблюдал.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    _client(mini, "c1")

    def purchase(moment: str, amount: int, precision: str | None = None) -> None:
        mini.event("c1", "purchase", moment, precision=precision, payload=purchase_payload(amount=amount))

    # Три дневных записи: тот же день, тот же день, следующий день.
    purchase("2023-03-05 10:00:00", 1000, "day")
    purchase("2023-03-05 11:00:00", 2000, "day")
    purchase("2023-03-06 09:00:00", 3000, "day")

    # Минутная после дневной: точность пары дневная.
    purchase("2023-03-06 12:30:45", 4000, "minute")

    # Точная после минутной: усечение до минуты.
    purchase("2023-03-07 08:00:30", 5000)

    # Две точные: интервал точный, как и был.
    purchase("2023-03-07 09:30:30", 6000)

    # Минутные: секунды усечены.
    purchase("2023-03-07 10:00:20", 7000, "minute")
    purchase("2023-03-07 10:30:50", 8000, "minute")

    store = _store(mini.write(), tmp_path / "canonical")

    history = semantic_as_of(store, "c1", datetime(2023, 4, 1))

    purchases = [item for item in history.events if item.values["event_type"] == "purchase"]

    assert [item.timing.since_previous_hours for item in purchases] == [
        None, 0.0, 24.0, 0.0, 19.5, 1.5, 0.5, 0.5
    ]

    assert [item.timing.time_precision for item in purchases] == [
        "day", "day", "day", "minute", "second", "second", "minute", "minute"
    ]

    # Возраст истории дневной записи — целые сутки от начала наблюдения.
    assert purchases[1].timing.age_of_history_days == 63.0

    # Тот же тип: интервал по той же точности пары.
    assert purchases[3].timing.since_same_type_hours == 0.0


def test_empty_month_without_coverage_is_not_silence(tmp_path):
    """
    Месяц без записей становится доказанным молчанием только
    тогда, когда источники действий клиента в нём наблюдались.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)

    # Операции и приложение подключены с марта, заявки с начала.
    mini.cover("c1", "transactions", first_seen="2023-03-01")
    mini.cover("c1", "app_operations", first_seen="2023-03-01")
    mini.cover("c1", "applications", first_seen="2023-01-01")

    for source in ("profile", "product_events", "loans", "communications", "banners",
                   "app_screens", "support", "antifraud"):
        mini.cover("c1", source, first_seen="2023-01-01")

    mini.profile_version("c1", 1, "2023-01-01", declared_income=50000)

    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload(amount=100))

    store = _store(mini.write(), tmp_path / "canonical")

    history = semantic_as_of(store, "c1", datetime(2023, 6, 1))

    months = {item.month: item.state for item in history.activity}

    # Январь и февраль: операций ещё не видно, молчание недоказуемо.
    assert months["2023-01"] == INSUFFICIENT_COVERAGE
    assert months["2023-02"] == INSUFFICIENT_COVERAGE

    # Март: клиент действовал. Апрель: источники были, записей нет.
    assert months["2023-03"] == HAS_CLIENT_ACTION
    assert months["2023-04"] == NO_RECORDS

    summary = history.activity_summary

    assert summary["unknown_months"] == 2
    assert summary["known_months"] == len(history.activity) - 2
    assert summary["current_pause_months"] == 2


def test_future_events_do_not_change_earlier_semantics(tmp_path):
    """
    Добавление будущих записей не меняет ни признаков, ни исходов
    цепочек на прежнем срезе.
    """

    def build(folder: str, future: bool):

        mini = MiniRaw(tmp_path / folder, history_start=FULL_HORIZON)
        _client(mini)

        application = mini.event(
            "c1", "application_submitted", "2023-02-01 10:00:00",
            payload=_application_payload(), correlation_id="app_1", link_type="application",
        )
        assert application

        mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload(amount=12500))

        if future:
            # Решение и покупка позже среза.
            mini.event(
                "c1", "application_decision", "2023-06-01 10:00:00",
                payload=_application_payload(decision="approved"),
                correlation_id="app_1", link_type="application",
            )
            mini.event("c1", "purchase", "2023-07-01 10:00:00", payload=purchase_payload(amount=999))

        store = _store(mini.write(), tmp_path / f"{folder}_canonical")

        return semantic_as_of(store, "c1", datetime(2023, 4, 1))

    before = build("raw_a", future=False)
    after = build("raw_b", future=True)

    assert [item.values for item in before.events] == [item.values for item in after.events]
    assert [item.as_dict() for item in before.chains] == [item.as_dict() for item in after.chains]
    assert before.activity_summary["by_state"] == after.activity_summary["by_state"]

    # Цепочка без видимого решения осталась незавершённой.
    assert all(item.outcome == "in_progress" for item in before.chains)


def _application_payload(**overrides) -> dict:
    payload = {
        "product_id": "prd_loan",
        "product_code": "LOAN",
        "product_version": 1,
        "requested_amount": 500000,
        "requested_term": 12,
        "approved_amount": None,
        "approved_term": None,
        "decision": None,
        "reject_reason": None,
        "channel": "app",
        "offer_id": None,
        "application_id": "app_1",
    }
    payload.update(overrides)
    return payload


def _cover_all_from(mini, client_id: str, first_seen: str, **overrides) -> None:
    """
    Полное покрытие клиента с одной даты, кроме перечисленных
    источников.
    """

    from src.generator.config import SOURCES

    for source in SOURCES:
        if source in overrides:
            mini.cover(client_id, source, **overrides[source])
        else:
            mini.cover(client_id, source, first_seen=first_seen)


def test_closed_source_does_not_erase_earlier_months(tmp_path):
    """
    Покрытие месяца считается по датированным полям ЭТОГО месяца.

    Источник, закрывшийся в мае, ничего не отнимает у февраля:
    февраль он наблюдал. Раньше состояние на cutoff требовалось
    от всех месяцев сразу, и закрытие источника стирало
    доказанное молчание всей прошлой истории.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)

    _cover_all_from(
        mini, "c1", "2023-01-01",
        app_operations={"first_seen": "2023-01-01", "last_available_at": "2023-05-01",
                        "status": "ended", "reason": None},
    )

    mini.profile_version("c1", 1, "2023-01-01", declared_income=50000)

    mini.event("c1", "purchase", "2023-01-05 10:00:00", payload=purchase_payload(amount=100))

    store = _store(mini.write(), tmp_path / "canonical")

    history = semantic_as_of(store, "c1", datetime(2023, 7, 1))

    months = {item.month: item.state for item in history.activity}

    # Февраль и март источник наблюдал: молчание доказано.
    assert months["2023-02"] == NO_RECORDS
    assert months["2023-03"] == NO_RECORDS

    # Июнь он уже не наблюдал: сказать нечего.
    assert months["2023-06"] == INSUFFICIENT_COVERAGE


def test_source_ending_mid_month_does_not_prove_silence(tmp_path):
    """
    Месяц покрыт, только если покрыт его последний день.

    Конец покрытия сравнивался с началом месяца: источник,
    кончившийся пятнадцатого января, «наблюдал» январь целиком, и
    пустая вторая половина месяца становилась доказанным
    молчанием.
    """

    def january(last_available_at: str) -> str:

        mini = MiniRaw(tmp_path / f"raw_{last_available_at}", history_start=FULL_HORIZON)

        _cover_all_from(
            mini, "c1", "2023-01-01",
            app_operations={"first_seen": "2023-01-01", "last_available_at": last_available_at,
                            "status": "ended", "reason": None},
        )

        mini.profile_version("c1", 1, "2023-01-01", declared_income=50000)

        mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload(amount=100))

        store = _store(mini.write(), tmp_path / f"canonical_{last_available_at}")

        history = semantic_as_of(store, "c1", datetime(2023, 4, 1))

        return {item.month: item.state for item in history.activity}["2023-01"]

    # Покрытие до середины января: сказать о январе нечего.
    assert january("2023-01-15") == INSUFFICIENT_COVERAGE

    # Покрытие до последнего дня включительно: молчание доказано.
    assert january("2023-01-31") == NO_RECORDS


def test_client_without_app_can_have_proven_silence(tmp_path):
    """
    Источник, неприменимый к клиенту, не требуется для
    доказательства молчания.

    У клиента без приложения не бывает месяца, в котором
    приложение записало бы действие. Раньше вся его история
    оказывалась «неизвестной», хотя операции наблюдались.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)

    _cover_all_from(
        mini, "c1", "2023-01-01",
        app_operations={"first_seen": None, "status": "none", "reason": "client_not_onboarded"},
        app_screens={"first_seen": None, "status": "none", "reason": "client_not_onboarded"},
    )

    mini.profile_version("c1", 1, "2023-01-01", declared_income=50000)

    mini.event("c1", "purchase", "2023-01-05 10:00:00", payload=purchase_payload(amount=100))

    store = _store(mini.write(), tmp_path / "canonical")

    history = semantic_as_of(store, "c1", datetime(2023, 5, 1))

    months = {item.month: item.state for item in history.activity}

    assert months["2023-01"] == HAS_CLIENT_ACTION
    assert months["2023-02"] == NO_RECORDS
    assert months["2023-03"] == NO_RECORDS


def test_bank_records_without_coverage_are_not_known_silence(tmp_path):
    """
    Записи банка без покрытия действий не доказывают, что клиент
    молчал: источники его действий в этом месяце не наблюдались.
    Число записей при этом сохраняется.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)

    # Операции подключены только с апреля; витрина договоров — с начала.
    _cover_all_from(
        mini, "c1", "2023-01-01",
        transactions={"first_seen": "2023-04-01"},
        app_operations={"first_seen": "2023-04-01"},
        applications={"first_seen": "2023-04-01"},
    )

    mini.profile_version("c1", 1, "2023-01-01", declared_income=50000)

    # Запись банка в феврале: клиент её не делал.
    mini.event(
        "c1", "product_repriced", "2023-02-10 10:00:00", initiator="bank",
        payload={
            "product_id": "prd_card", "product_code": "CARD", "product_version": 2,
            "tariff_version": 2, "product_family": "debit_card", "contract_id": "ctr_1",
            "account_id": "acc_1", "card_id": None, "offer_id": None,
            "previous_product_id": None, "migration_reason": None, "amount_or_limit": None,
            "term": None, "rate": None, "reason": "repricing", "timestamp_quality": "exact",
        },
    )

    mini.event("c1", "purchase", "2023-04-05 10:00:00", payload=purchase_payload(amount=100))

    store = _store(mini.write(), tmp_path / "canonical")

    history = semantic_as_of(store, "c1", datetime(2023, 6, 1))

    february = next(item for item in history.activity if item.month == "2023-02")

    assert february.state == INSUFFICIENT_COVERAGE
    assert february.events == 1
    assert february.client_actions == 0


def test_exact_month_boundaries_are_not_partial(tmp_path):
    """
    Месяц неполон только тогда, когда граница наблюдения прошла
    ВНУТРИ него. Совпавшая с началом месяца граница отнимала
    настоящий месяц наблюдения.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    _client(mini, "c1")

    # Наблюдение начинается ровно первого февраля в 00:00.
    mini.event("c1", "purchase", "2023-02-01 00:00:00", payload=purchase_payload(amount=100))
    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload(amount=200))

    store = _store(mini.write(), tmp_path / "canonical")

    # Срез ровно на 00:00 первого апреля: март прожит целиком.
    history = semantic_as_of(store, "c1", datetime(2023, 4, 1))

    partial = {item.month: item.partial for item in history.activity}

    assert partial["2023-02"] is False
    assert partial["2023-03"] is False

    # А срез в середине месяца этот месяц действительно обрывает.
    inside = semantic_as_of(store, "c1", datetime(2023, 4, 15))

    partial = {item.month: item.partial for item in inside.activity}

    assert partial["2023-02"] is False
    assert partial["2023-04"] is True


def test_single_step_chain_is_in_progress(tmp_path):
    """
    Цепочка из одного видимого шага это НАЧАВШАЯСЯ цепочка, а не
    отсутствие цепочки. Раньше она исчезала, и доля незавершённых
    считалась по неполному знаменателю.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    _client(mini, "c1")

    mini.event(
        "c1", "application_submitted", "2023-03-05 10:00:00",
        payload=_application_payload(), correlation_id="app_1", link_type="application",
    )

    store = _store(mini.write(), tmp_path / "canonical")

    history = semantic_as_of(store, "c1", datetime(2023, 4, 1))

    assert len(history.chains) == 1

    chain = history.chains[0]

    assert chain.steps == 1
    assert chain.kind == "application"
    assert chain.outcome == "in_progress"


def test_partial_withdrawal_does_not_close_a_contract(tmp_path):
    """
    Снятие с вклада договор не закрывает: частичное снятие это
    обычная операция. Закрывают только явные события.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    _client(mini, "c1")

    mini.event(
        "c1", "product_opened", "2023-02-01 10:00:00",
        correlation_id="ctr_dep", link_type="contract",
        payload={
            "product_id": "prd_dep", "product_code": "DEP", "product_version": 1,
            "tariff_version": 1, "product_family": "deposit", "contract_id": "ctr_dep",
            "account_id": "acc_dep", "card_id": None, "offer_id": None,
            "previous_product_id": None, "migration_reason": None, "amount_or_limit": 1000000,
            "term": 12, "rate": 0.14, "reason": "application", "timestamp_quality": "exact",
        },
    )

    mini.event(
        "c1", "deposit_withdrawal", "2023-03-05 10:00:00",
        correlation_id="ctr_dep", link_type="contract",
        payload=purchase_payload(amount=200000, direction="debit", reason="withdrawal",
                                 contract_id="ctr_dep", account_id="acc_dep"),
    )

    store = _store(mini.write(), tmp_path / "canonical")

    history = semantic_as_of(store, "c1", datetime(2023, 4, 1))

    chain = next(item for item in history.chains if item.kind == "contract")

    assert chain.outcome == "in_progress"
    assert chain.last_event_type == "deposit_withdrawal"


def test_ratios_use_income_and_limit_of_that_moment(tmp_path):
    """
    Отношение к доходу считается по версии профиля, действовавшей
    В МОМЕНТ ОПЕРАЦИИ, а отношение к лимиту — только у кредитной
    карты.

    Иначе покупка двухлетней давности делилась бы на сегодняшний
    доход, а сумма вклада выдавалась бы за кредитный лимит.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    mini.profile_version("c1", 1, "2023-01-01", declared_income=100000, age=40)
    mini.profile_version("c1", 2, "2023-05-01", declared_income=200000, age=40)

    # Вклад: amount_or_limit это сумма размещения, не лимит.
    mini.event(
        "c1", "product_opened", "2023-02-01 10:00:00",
        correlation_id="ctr_dep", link_type="contract",
        payload={
            "product_id": "prd_dep", "product_code": "DEP", "product_version": 1,
            "tariff_version": 1, "product_family": "deposit", "contract_id": "ctr_dep",
            "account_id": "acc_dep", "card_id": None, "offer_id": None,
            "previous_product_id": None, "migration_reason": None, "amount_or_limit": 1000000,
            "term": 12, "rate": 0.14, "reason": "application", "timestamp_quality": "exact",
        },
    )

    mini.event(
        "c1", "deposit_topup", "2023-03-05 10:00:00",
        payload=purchase_payload(amount=50000, direction="credit", reason="topup",
                                 contract_id="ctr_dep", account_id="acc_dep"),
    )

    # Кредитная карта: то же поле означает лимит.
    mini.event(
        "c1", "product_opened", "2023-06-01 10:00:00",
        correlation_id="ctr_card", link_type="contract",
        payload={
            "product_id": "prd_cc", "product_code": "CC", "product_version": 1,
            "tariff_version": 1, "product_family": "credit_card", "contract_id": "ctr_card",
            "account_id": "acc_cc", "card_id": None, "offer_id": None,
            "previous_product_id": None, "migration_reason": None, "amount_or_limit": 400000,
            "term": None, "rate": 0.0, "reason": "application", "timestamp_quality": "exact",
        },
    )

    mini.event(
        "c1", "purchase", "2023-07-05 10:00:00",
        payload=purchase_payload(amount=40000, contract_id="ctr_card", account_id="acc_cc"),
    )

    store = _store(mini.write(), tmp_path / "canonical")

    history = semantic_as_of(store, "c1", datetime(2023, 9, 1))

    def derived(event, key):
        return next(item for item in event.derived if item.key == key)

    topup = next(item for item in history.events if item.values["event_type"] == "deposit_topup")

    # Март: действовала первая версия профиля, доход 100 000.
    income = derived(topup, "amount_to_declared_income")

    assert income.value == pytest.approx(50000 / 100000)
    assert any(
        item.get("kind") == "profile" and item.get("profile_version") == 1
        for item in income.derived_from
    )

    # Сумма вклада лимитом не является.
    limit = derived(topup, "amount_to_limit")

    assert limit.value is None
    assert limit.reason == "no_credit_limit"

    purchase = next(item for item in history.events if item.values["event_type"] == "purchase")

    # Июль: действовала вторая версия, доход 200 000.
    income = derived(purchase, "amount_to_declared_income")

    assert income.value == pytest.approx(40000 / 200000)
    assert any(
        item.get("kind") == "profile" and item.get("profile_version") == 2
        for item in income.derived_from
    )

    # А у кредитной карты отношение к лимиту считается.
    limit = derived(purchase, "amount_to_limit")

    assert limit.value == pytest.approx(40000 / 400000)
    assert limit.reason is None


# ============================================================
# ЭТАП
# ============================================================


def test_stage_writes_registry_and_reads_nothing_forbidden(tmp_path, capsys):
    """
    Этап пишет реестр, отчёт, пример и две диагностические
    таблицы с явным срезом в имени, не читая ни truth, ни
    токенизатор, ни модель.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    _client(mini)
    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload(amount=12500))

    raw_dir = mini.write()
    out = tmp_path / "processed"

    assert _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK
    assert _cli("canonical", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK

    forbidden = {"src.tokenizer", "src.model"}
    before = {name for name in sys.modules if name in forbidden}

    capsys.readouterr()

    assert (
        _cli("semantic", "--name", "x", "--group", "train", "--out", str(out),
             "--raw", str(raw_dir), "--cutoff", "2023-04-01T00:00:00")
        == EXIT_OK
    )

    assert {name for name in sys.modules if name in forbidden} == before

    target = out / "semantic" / "train"

    assert (target / REGISTRY_FILE).exists()
    assert (target / "semantic_report.md").exists()
    assert (target / "activity_months__2023-04-01.parquet").exists()
    assert (target / "chains__2023-04-01.parquet").exists()
    assert list((target / "examples").glob("*.md"))

    registry = json.loads((target / REGISTRY_FILE).read_text(encoding="utf-8"))

    assert registry["undeclared_keys"] == []
    assert registry["keys_used"] > 0
    assert registry["registry"]["counts"]["keys"] > 50

    entry = json.loads((out / "preprocessing_manifest.json").read_text(encoding="utf-8"))["stages"]["semantic"]
    assert entry["train"]["status"] == "ok"

    # Повторный запуск ничего не пересчитывает.
    capsys.readouterr()
    assert (
        _cli("semantic", "--name", "x", "--group", "train", "--out", str(out),
             "--raw", str(raw_dir), "--cutoff", "2023-04-01T00:00:00")
        == EXIT_OK
    )
    assert "этап пропущен" in capsys.readouterr().out


def test_semantic_never_imports_the_generator(tmp_path):
    """
    Смысловой слой описывает СВОЙ ВХОД, а не код, которым данные
    могли быть собраны. Каталог ключей читается из реестра полей
    canonical; импорт генератора означал бы, что подмена
    генератора останется незамеченной.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    _client(mini)
    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload(amount=12500))

    raw_dir = mini.write()
    out = tmp_path / "processed"

    assert _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK
    assert _cli("canonical", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK

    source = Path("src/preprocessing/semantic/build.py").read_text(encoding="utf-8")

    assert "src.generator" not in source

    # Каталог восстанавливается из реестра полей и совпадает с
    # каталогом выгрузки по составу.
    registry = json.loads(
        (out / "canonical" / "train" / "field_registry.json").read_text(encoding="utf-8")
    )

    restored = catalogue_from_registry(registry)

    catalogue = key_catalogue()

    assert set(restored) == set(catalogue)

    for event_type, info in restored.items():
        assert info["source"] == catalogue[event_type]["source"]
        assert [item["name"] for item in info["fields"]] == [
            item["name"] for item in catalogue[event_type]["fields"]
        ]


def test_catalog_change_recomputes_the_stage(tmp_path):
    """
    Справочник точек меняет РЕЗУЛЬТАТ: он даёт названия и
    категории. Не входя в отпечаток, подменённый справочник
    оставлял бы этап пропущенным, а отчёт — от прежнего каталога.
    """

    import pyarrow as pa
    import pyarrow.parquet as pq

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    _client(mini)
    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload(amount=12500))

    raw_dir = mini.write()
    out = tmp_path / "processed"

    assert _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK
    assert _cli("canonical", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK

    def run_semantic() -> str:
        assert (
            _cli("semantic", "--name", "x", "--group", "train", "--out", str(out),
                 "--raw", str(raw_dir), "--cutoff", "2023-04-01T00:00:00")
            == EXIT_OK
        )
        marker = json.loads(
            fingerprint_path(out, "semantic", "train").read_text(encoding="utf-8")
        )
        return marker["fingerprint"]

    first = run_semantic()

    # Тот же запуск без изменений: отпечаток тот же.
    assert run_semantic() == first

    # Справочник подменён — отпечаток обязан измениться.
    merchants = raw_dir / "catalog" / "merchants.parquet"

    table = pq.read_table(merchants)

    changed = table.set_column(
        table.column_names.index("merchant_name"),
        "merchant_name",
        pa.array(["Другое название"] * table.num_rows, pa.string()),
    )

    pq.write_table(changed, merchants)

    # Canonical собран из прежних файлов RAW: этап обязан
    # отказаться, а не молча пересчитаться на других данных.
    assert (
        _cli("semantic", "--name", "x", "--group", "train", "--out", str(out),
             "--raw", str(raw_dir), "--cutoff", "2023-04-01T00:00:00")
        == EXIT_BLOCKED
    )


def test_client_example_does_not_replace_stage_state(tmp_path):
    """
    Разбор одного клиента это чтение, а не сборка этапа: он не
    чистит каталог, не переписывает маркер и не подменяет запись
    в манифесте состоянием одного клиента.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    _client(mini, "c1")
    _client(mini, "c2")
    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload(amount=12500))
    mini.event("c2", "purchase", "2023-03-06 10:00:00", payload=purchase_payload(amount=4300))

    raw_dir = mini.write()
    out = tmp_path / "processed"

    assert _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK
    assert _cli("canonical", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK
    assert (
        _cli("semantic", "--name", "x", "--group", "train", "--out", str(out),
             "--raw", str(raw_dir), "--cutoff", "2023-04-01T00:00:00")
        == EXIT_OK
    )

    target = out / "semantic" / "train"

    marker_path = fingerprint_path(out, "semantic", "train")

    before_marker = marker_path.read_text(encoding="utf-8")
    before_report = (target / "semantic_report.md").read_text(encoding="utf-8")

    assert (
        _cli("semantic", "--name", "x", "--group", "train", "--out", str(out),
             "--raw", str(raw_dir), "--cutoff", "2023-04-01T00:00:00", "--client", "c2")
        == EXIT_OK
    )

    # Состояние этапа не изменилось.
    assert marker_path.read_text(encoding="utf-8") == before_marker
    assert (target / "semantic_report.md").read_text(encoding="utf-8") == before_report
    assert (target / "activity_months__2023-04-01.parquet").exists()

    # А пример клиента появился рядом.
    assert list((target / "examples").rglob("*c2*"))


def test_stage_is_reproducible(tmp_path):
    """
    Один вход и одна настройка дают тот же отчёт.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    _client(mini)
    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload(amount=12500))

    raw_dir = mini.write()
    canonical = tmp_path / "canonical"

    build_canonical(raw_dir, canonical, CONFIG, "train")

    first = build_group(canonical, tmp_path / "one", CONFIG, "train", datetime(2023, 4, 1), raw_dir=raw_dir)
    second = build_group(canonical, tmp_path / "two", CONFIG, "train", datetime(2023, 4, 1), raw_dir=raw_dir)

    assert first.report == second.report


def test_stage_version_tracks_sources():

    import hashlib

    from src.preprocessing import calendar as calendar_module
    from src.preprocessing import history as history_module
    from src.preprocessing import projection as projection_module
    from src.preprocessing import run as run_module
    from src.preprocessing import settings as settings_module
    from src.preprocessing.semantic import activity, as_of, build, chains, formulas, keys, merchants
    from src.preprocessing.semantic import time as semantic_time

    modules = {
        "build": build,
        "keys": keys,
        "as_of": as_of,
        "activity": activity,
        "chains": chains,
        "formulas": formulas,
        "merchants": merchants,
        "time": semantic_time,
        "projection": projection_module,
        "calendar": calendar_module,
        "history": history_module,
        "settings": settings_module,
        "run": run_module,
    }

    stored = json.loads(Path("tests/prep_stage_sources.json").read_text(encoding="utf-8"))["semantic"]

    assert stored["version"] == build.STAGE_VERSION

    actual = {
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in modules.items()
    }

    assert stored["modules"] == actual, (
        "модули этапа изменены: поднимите STAGE_VERSION и обновите tests/prep_stage_sources.json"
    )
