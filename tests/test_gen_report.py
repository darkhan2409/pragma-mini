from __future__ import annotations

from datetime import datetime

from src.generator.config import (
    HISTORY_END,
    HISTORY_START,
    INITIATOR_BANK,
    INITIATOR_CLIENT,
    SOURCES,
)
from src.generator.report.realism import _absence, _activity, _windows


# ============================================================
# МЕТРИКИ ОТЧЁТА
# ============================================================
#
# Разбор активности и исчезновений проверяется на СОБРАННЫХ
# РУКАМИ входах, без генерации: нужна не правдоподобность чисел,
# а поведение определений на краях.
#
# Края здесь такие: месяц без единой записи, клиент без
# транзакций, пауза на конце окна, молчание до первого действия,
# месяц с недостаточным покрытием.
# ============================================================


def _coverage(client_id: str, first_seen: dict | None = None,
              reasons: dict | None = None, closed_at: datetime | None = None) -> list:
    """
    Строка покрытия на каждую пару клиент и источник.

    По умолчанию все источники наблюдаются с начала истории.
    """

    first_seen = first_seen or {}
    reasons = reasons or {}

    rows = []

    for source in SOURCES:
        rows.append(
            {
                "client_id": client_id,
                "source": source,
                "first_available_at": HISTORY_START,
                "last_available_at": closed_at,
                "first_seen": first_seen.get(source, HISTORY_START),
                "coverage_status": "ended" if closed_at else "full",
                "coverage_reason": (
                    "relationship_closed" if closed_at else reasons.get(source)
                ),
                "opening_state": None,
            }
        )

    return rows


def _event(client_id: str, month: str, initiator: str = INITIATOR_CLIENT,
           event_type: str = "purchase") -> dict:
    return {
        "client_id": client_id,
        "event_time": datetime(int(month[:4]), int(month[5:]), 15),
        "change_initiator": initiator,
        "event_type": event_type,
        "source": "transactions",
    }


def _data(clients: list, coverage: list, events: list) -> dict:

    data = {
        "truth_clients": clients,
        "coverage": coverage,
        "events": events,
        "truth_events": [],
    }

    data["windows"] = _windows(data)

    return data


# ------------------------------------------------------------
# ОКНО НАБЛЮДЕНИЯ
# ------------------------------------------------------------


def test_window_starts_at_the_relationship_not_at_source_launch():
    """
    Окно берётся по first_seen, а не по first_available_at.

    first_available_at это дата запуска ИСТОЧНИКА в банке, и к
    конкретному клиенту отношения не имеет. Клиент, пришедший
    позже, лишних месяцев в начале получать не должен.
    """

    late = datetime(2025, 3, 1)

    data = _data(
        [{"client_id": "c1", "final_state": "active"}],
        _coverage("c1", first_seen={source: late for source in SOURCES}),
        [],
    )

    window = data["windows"]["c1"]

    assert window["start"] == late
    assert window["months"][0]["month"] == "2025-03"


def test_open_end_runs_to_the_history_end():
    """
    last_available_at = None означает «конец не задан», а не
    «наблюдения нет».
    """

    data = _data(
        [{"client_id": "c1", "final_state": "active"}],
        _coverage("c1"),
        [],
    )

    window = data["windows"]["c1"]

    assert window["closed_at"] is None
    assert window["end"] == HISTORY_END
    assert window["months"]


def test_client_without_transactions_stays_in_the_grid():
    """
    Клиент без единой транзакции из знаменателя не выпадает:
    остальные источники его покрывают.
    """

    data = _data(
        [{"client_id": "c1", "final_state": "active"}],
        _coverage("c1"),
        [_event("c1", "2025-05", INITIATOR_BANK, "communication_sent")],
    )

    report = _activity(data)

    assert report["client_months"] > 0
    assert data["windows"]["c1"]["months"]


def test_missing_app_is_not_an_unknown_month():
    """
    Приложение не установлено — это свойство клиента, а не
    пробел наблюдения: событий этого канала у него не бывает
    по-настоящему.
    """

    absent = {"app_screens": None, "app_operations": None, "banners": None}

    data = _data(
        [{"client_id": "c1", "final_state": "active"}],
        _coverage(
            "c1",
            first_seen=absent,
            reasons={name: "client_not_onboarded" for name in absent},
        ),
        [],
    )

    assert all(item["known_any"] for item in data["windows"]["c1"]["months"])
    assert _activity(data)["unknown_months"] == 0


def test_partial_coverage_makes_the_month_unknown():
    """
    Доступной поддержки при недоступных транзакциях не хватает:
    отсутствие обращений бездействия клиента не доказывает.
    """

    data = _data(
        [{"client_id": "c1", "final_state": "active"}],
        _coverage(
            "c1",
            first_seen={"transactions": None},
            reasons={"transactions": "source_not_connected"},
        ),
        [],
    )

    assert not any(item["known_any"] for item in data["windows"]["c1"]["months"])
    assert not any(item["known_action"] for item in data["windows"]["c1"]["months"])


# ------------------------------------------------------------
# ТРИ ВИДА МЕСЯЦА
# ------------------------------------------------------------


def test_empty_month_counts_as_a_month_without_client_action():
    """
    Полностью пустой месяц входит И в «нет никаких записей», И в
    общий показатель месяцев без действий клиента. Определение
    последнего не менялось: это ВСЕ месяцы без действий клиента.
    """

    events = [_event("c1", month) for month in ("2024-06", "2024-08")]

    # банк работал в июле и сентябре, клиент молчал
    events.append(_event("c1", "2024-07", INITIATOR_BANK, "statement_issued"))
    events.append(_event("c1", "2024-09", INITIATOR_BANK, "statement_issued"))

    data = _data(
        [{"client_id": "c1", "final_state": "active"}],
        _coverage("c1", closed_at=datetime(2024, 11, 1)),
        events,
    )

    report = _activity(data)

    # октябрь остался вообще без записей
    assert report["zero_month_share"] > 0
    assert report["bank_only_month_share"] > 0

    assert report["month_kinds_sum"] == 1.0

    assert report["no_client_action_month_share"] == round(
        report["zero_month_share"] + report["bank_only_month_share"], 4
    )


def test_recorded_action_beats_a_coverage_gap():
    """
    Зафиксированное действие всегда сильнее пробела: месяц с
    действием активен при любом покрытии.
    """

    data = _data(
        [{"client_id": "c1", "final_state": "active"}],
        _coverage(
            "c1",
            first_seen={"support": None},
            reasons={"support": "source_not_connected"},
        ),
        [_event("c1", "2025-04")],
    )

    result = _absence(data)

    # Клиент действовал, значит стартовое молчание кончилось, а
    # не тянется до конца окна.
    assert result["clients_never_acted"] == 0


# ------------------------------------------------------------
# ИСЧЕЗНОВЕНИЕ И ВОЗВРАЩЕНИЕ
# ------------------------------------------------------------


def test_leading_silence_is_not_a_disappearance():
    """
    Молчание до первого действия это ещё не исчезновение:
    клиент просто не начал. Ни в один знаменатель оно не идёт.
    """

    data = _data(
        [{"client_id": "c1", "final_state": "active"}],
        _coverage("c1", closed_at=datetime(2024, 12, 1)),
        [_event("c1", month) for month in ("2024-10", "2024-11")],
    )

    result = _absence(data)

    assert result["leading_silence_months"]["max"] == 4
    assert sum(result["outcomes"].values()) == 0


def test_pause_at_the_window_end_is_not_a_departure():
    """
    Молчание на конце датасета это НЕ уход из банка: будущее
    клиента неизвестно. В подтверждённые закрытия оно не идёт,
    а в знаменатель доли возвращения входит.
    """

    data = _data(
        [{"client_id": "c1", "final_state": "dormant"}],
        _coverage("c1"),
        [_event("c1", "2024-06")],
    )

    result = _absence(data)

    assert result["outcomes"]["ongoing"] == 1
    assert result["outcomes"].get("closed", 0) == 0
    assert result["confirmed_closure_clients"] == 0

    # Доля возвращения считается от установленных исходов, и
    # продолжающаяся пауза в знаменателе.
    assert result["returned_by_window_end_share"] == 0.0
    assert result["pause_ongoing_at_window_end_share"] == 1.0


def test_return_is_counted_from_observed_actions():

    months = ("2024-06", "2024-10")

    data = _data(
        [{"client_id": "c1", "final_state": "active"}],
        _coverage("c1", closed_at=datetime(2024, 12, 1)),
        [_event("c1", month) for month in months],
    )

    result = _absence(data)

    assert result["outcomes"]["returned"] == 1
    assert result["buckets"]["3-5"]["returned"] == 1
    assert result["buckets"]["3-5"]["episodes"] == 1

    # Вернувшись, клиент снова расплатился картой.
    assert result["returned_and_used_products"] == 1


def test_unknown_month_breaks_a_pause_instead_of_gluing_it():
    """
    Две тишины вокруг неизвестного месяца не склеиваются в одну
    длинную. Про такую паузу неизвестно ничего, и в знаменатель
    доли возвращения она не входит.
    """

    # поддержка появляется только с сентября: до неё месяцы
    # неизвестны для действий клиента
    data = _data(
        [{"client_id": "c1", "final_state": "active"}],
        _coverage(
            "c1",
            first_seen={"support": datetime(2024, 9, 1)},
            closed_at=datetime(2024, 12, 1),
        ),
        [_event("c1", "2024-06"), _event("c1", "2024-11")],
    )

    result = _absence(data)

    assert result["outcomes"].get("returned", 0) == 0
    assert result["outcomes"]["observation_broken"] == 1

    # Установленных исходов нет — делить не на что.
    assert result["returned_by_window_end_share"] is None
