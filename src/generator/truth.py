from __future__ import annotations

from datetime import datetime

import pyarrow as pa

from .profile import UTC_MICROS, utc


# ============================================================
# ПРАВДА СИМУЛЯЦИИ
# ============================================================
#
# То, чего банк не видит: скрытое состояние отношений клиента
# с банком и причины его переходов. Лежит рядом с выгрузкой в
# truth/ и нужно только аудиту — проверить, из чего родилась
# история, и оценить потолок задачи.
#
# Вход модели, признаки и метки downstream truth/ не читают
# (CLAUDE.md, тесты изоляции): всё, что знает банк, уже есть в
# events.parquet и profile.parquet.
#
#   transitions  смена скрытого состояния: компонента, новое
#                значение и причина — член с наибольшим вкладом
#                в опасность перехода или baseline;
#   states       недельный снимок состояния по понедельникам.
#
# Строки — только внутри окна выгрузки, как и события: префикс
# truth при удлинении окна не меняется.
# ============================================================


TRANSITIONS_SCHEMA = pa.schema(
    [
        ("client_id", pa.string()),
        ("time", UTC_MICROS),
        ("component", pa.string()),
        ("value", pa.string()),
        ("cause", pa.string()),
    ]
)

STATES_SCHEMA = pa.schema(
    [
        ("client_id", pa.string()),
        ("time", UTC_MICROS),
        ("friction", pa.float64()),
        ("affinity", pa.float64()),
        ("target", pa.float64()),
        ("regime", pa.string()),
        ("away", pa.string()),
        ("migrating", pa.bool_()),
        ("salary_here", pa.bool_()),
        ("stress", pa.float64()),
    ]
)


def transition(client_id: str, moment: datetime, component: str, value: str, cause: str) -> dict:
    return {
        "client_id": client_id,
        "time": utc(moment),
        "component": component,
        "value": value,
        "cause": cause,
    }


__all__ = ["STATES_SCHEMA", "TRANSITIONS_SCHEMA", "transition"]
