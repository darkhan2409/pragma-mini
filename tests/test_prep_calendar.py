from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.calendar import CALENDAR_FEATURES, calendar_features
from src.preprocessing.settings import CALENDAR_ENCODING, PreprocessingConfig


def _at(*stamps: str) -> np.ndarray:
    return calendar_features(np.array([np.datetime64(value, "us") for value in stamps]))


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right))


def test_calendar_cycles_close_at_their_boundaries():
    """
    Шесть чисел на событие: час, день недели и день месяца на
    единичной окружности. Проверяются границы, на которых
    циклическое представление и обязано отличаться от линейного.
    """

    assert CALENDAR_FEATURES == 6 == len(CALENDAR_ENCODING["features"])

    # Час учитывает минуты и секунды.
    hours = _at("2026-03-05T09:00:00", "2026-03-05T09:30:00", "2026-03-05T09:30:30")
    assert _distance(hours[0, :2], hours[1, :2]) > 0
    assert _distance(hours[1, :2], hours[2, :2]) > 0

    # Полночь: соседние минуты рядом, полдень далеко.
    midnight = _at("2026-03-05T23:59:00", "2026-03-06T00:01:00", "2026-03-05T12:00:00")
    assert _distance(midnight[0, :2], midnight[1, :2]) < 0.05
    assert _distance(midnight[0, :2], midnight[2, :2]) > 1.5

    # Понедельник начинает недельный цикл, воскресенье примыкает к нему
    # так же, как понедельник к вторнику.
    monday = _at("2026-03-02T00:00:00")[0, 2:4]
    assert monday == pytest.approx([0.0, 1.0], abs=1e-6)

    week = _at("2026-03-08T12:00:00", "2026-03-09T12:00:00", "2026-03-10T12:00:00")
    assert _distance(week[0, 2:4], week[1, 2:4]) == pytest.approx(
        _distance(week[1, 2:4], week[2, 2:4]), abs=1e-6
    )

    # День месяца: первое число начинает цикл, тридцать первое его
    # замыкает ровно одним суточным шагом.
    first = _at("2026-01-01T00:00:00")[0, 4:6]
    assert first == pytest.approx([0.0, 1.0], abs=1e-6)

    step = _distance(_at("2026-01-01T12:00:00")[0, 4:6], _at("2026-01-02T12:00:00")[0, 4:6])
    wrap = _distance(_at("2026-01-31T12:00:00")[0, 4:6], _at("2026-02-01T12:00:00")[0, 4:6])
    assert wrap == pytest.approx(step, abs=1e-6)

    # Любое число месяца с 1-го по 31-е даёт конечные значения.
    whole = _at(*[f"2026-01-{day:02d}T08:00:00" for day in range(1, 32)])
    assert whole.shape == (31, CALENDAR_FEATURES)
    assert np.isfinite(whole).all()

    # Пустой вход не ломается, размерность сохраняется.
    assert calendar_features(np.array([], dtype="datetime64[us]")).shape == (0, CALENDAR_FEATURES)


def test_calendar_values_are_not_stored_as_fields():
    """
    Час, день недели и день месяца существуют только как
    производные от event_time: в каталоге ключей выгрузки, в
    реестре единиц и в модельной проекции их нет.
    """

    from src.generator.config import key_catalogue
    from src.preprocessing.canonical.registry import UNITS
    from src.preprocessing.projection import INTERNAL_FIELDS, SEMANTIC_PAYLOAD_FIELDS

    calendar_names = {"hour", "day_of_week", "day_of_month"}

    payload_names = {
        item["name"] for info in key_catalogue().values() for item in info["fields"]
    }

    assert not calendar_names & payload_names
    assert not calendar_names & set(UNITS)
    assert not calendar_names & set(SEMANTIC_PAYLOAD_FIELDS)
    assert not calendar_names & set(INTERNAL_FIELDS)


def test_calendar_rules_live_in_the_config_and_reject_missing_time():
    """
    Правила календаря объявлены в конфигурации, входят в отпечаток
    canonical и в манифест. Событие без времени календарю не
    подходит, и это говорится прямо.
    """

    config = PreprocessingConfig()

    declared = config.as_dict()["calendar"]

    assert declared["timezone"] == config.timezone == "Asia/Almaty"
    assert declared["source"] == "event_time"
    assert declared["cycles"] == {"hour_of_day": 24, "day_of_week": 7, "day_of_month": 31}
    assert declared["week_starts_on"] == "monday"

    # Календарь не становится токеном ни на одном шаге.
    assert set(declared["excluded_from"]) == {
        "key_vocab",
        "value_vocab",
        "bpe",
        "buckets",
        "masker",
        "prediction_targets",
    }

    # Смена правила пересобирает canonical: они в его отпечатке.
    assert "calendar" in config.section("canonical")

    with pytest.raises(ValueError, match="нет event_time"):
        calendar_features(np.array([np.datetime64("2026-03-05T10:00:00", "us"), np.datetime64("NaT")]))
