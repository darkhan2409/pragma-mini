from __future__ import annotations

from dataclasses import dataclass


# ============================================================
# БЫТОВЫЕ ПОСЛЕДОВАТЕЛЬНОСТИ
# ============================================================
#
# День складывается не из независимых покупок: по будням это
# дорога, кофе у работы, обед и продукты у дома вечером, по
# выходным рынок, кафе и развлечения.
#
# Последовательность зависит от транспорта, состава семьи и
# наличия работы.
# ============================================================


@dataclass(frozen=True)
class RoutineStep:
    category: str
    hour_low: int
    hour_high: int
    zone: str
    probability: float


WEEKDAY_COMMUTE_CAR = (
    RoutineStep("fuel", 7, 10, "home", 0.28),
    RoutineStep("coffee", 8, 11, "work", 0.42),
    RoutineStep("fastfood", 12, 15, "work", 0.46),
    RoutineStep("parking", 8, 11, "work", 0.18),
    RoutineStep("grocery", 17, 21, "home", 0.55),
    RoutineStep("pharmacy", 18, 21, "home", 0.09),
)

WEEKDAY_COMMUTE_TRANSIT = (
    RoutineStep("transit", 7, 10, "home", 0.78),
    RoutineStep("coffee", 8, 11, "work", 0.34),
    RoutineStep("fastfood", 12, 15, "work", 0.48),
    RoutineStep("transit", 17, 20, "work", 0.74),
    RoutineStep("grocery", 17, 21, "home", 0.52),
    RoutineStep("convenience", 19, 22, "home", 0.22),
)

WEEKDAY_COMMUTE_TAXI = (
    RoutineStep("taxi", 7, 11, "home", 0.55),
    RoutineStep("coffee", 8, 11, "work", 0.38),
    RoutineStep("fastfood", 12, 15, "work", 0.44),
    RoutineStep("taxi", 17, 21, "work", 0.48),
    RoutineStep("grocery", 17, 22, "home", 0.50),
)

WEEKDAY_AT_HOME = (
    RoutineStep("convenience", 9, 13, "home", 0.34),
    RoutineStep("grocery", 10, 19, "home", 0.48),
    RoutineStep("pharmacy", 10, 19, "home", 0.14),
    RoutineStep("market", 9, 14, "home", 0.22),
)

WEEKEND_FAMILY = (
    RoutineStep("market", 9, 14, "home", 0.38),
    RoutineStep("grocery", 10, 18, "home", 0.58),
    RoutineStep("fastfood", 12, 17, "other", 0.34),
    RoutineStep("restaurant", 13, 21, "other", 0.24),
    RoutineStep("entertainment", 12, 20, "other", 0.20),
    RoutineStep("kids", 11, 18, "other", 0.22),
)

WEEKEND_SOLO = (
    RoutineStep("coffee", 10, 15, "other", 0.34),
    RoutineStep("grocery", 11, 20, "home", 0.46),
    RoutineStep("restaurant", 13, 22, "other", 0.26),
    RoutineStep("cinema", 14, 22, "other", 0.14),
    RoutineStep("sports", 9, 20, "other", 0.18),
)

MONTHLY = (
    RoutineStep("utilities", 9, 21, "home", 0.0),
    RoutineStep("telecom", 9, 21, "home", 0.0),
)


def routine_for(
    weekday: int,
    transport: str,
    has_job: bool,
    household_size: int,
    has_children: bool,
) -> tuple:
    """
    Последовательность дня.
    """

    if weekday >= 5:
        return WEEKEND_FAMILY if has_children or household_size > 2 else WEEKEND_SOLO

    if not has_job:
        return WEEKDAY_AT_HOME

    if transport == "car":
        return WEEKDAY_COMMUTE_CAR

    if transport == "taxi":
        return WEEKDAY_COMMUTE_TAXI

    if transport == "transit":
        return WEEKDAY_COMMUTE_TRANSIT

    return WEEKDAY_AT_HOME


__all__ = [
    "RoutineStep",
    "WEEKDAY_AT_HOME",
    "WEEKDAY_COMMUTE_CAR",
    "WEEKDAY_COMMUTE_TAXI",
    "WEEKDAY_COMMUTE_TRANSIT",
    "WEEKEND_FAMILY",
    "WEEKEND_SOLO",
    "routine_for",
]
