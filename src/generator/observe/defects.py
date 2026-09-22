from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

from .. import params as params_module
from ..rng import NS_OBSERVE, keyed_rng, stable_hash
from .envelope import Event


# ============================================================
# ДЕФЕКТЫ НАБЛЮДАЕМОСТИ
# ============================================================
#
# Идеально чистая история нереалистична. Дефект принадлежит
# ИСТОЧНИКУ, а не равномерному шуму по всем данным:
#
#   пропуски полей с причиной
#   смена схемы: поле начинает собираться с определённой даты
#
# Чего здесь БОЛЬШЕ НЕТ: дублей, исправлений и сбоев источника.
# Запись приходит в выгрузку один раз и сразу окончательной, а
# отсутствие записи означает ровно одно: события не было.
#
# Проход идёт по идентичности события, а не по номеру строки:
# вставка события в другом месте ленты ничего не сдвигает.
# ============================================================


def _rng(event: Event, slot: int):
    """
    Ключ случайности события: клиент и номер выдачи. Номер живёт
    в памяти симуляции, в выгрузку не попадает и от места строки
    в файле не зависит.
    """

    return keyed_rng(NS_OBSERVE, stable_hash(event.client_id, event.ordinal) % (2 ** 31), slot)


def _apply_schema_change(event: Event) -> Event:
    """
    Поле начинает собираться с определённой даты: до неё оно
    пусто с причиной not_collected.
    """

    settings = params_module.active().defects

    payload = event.payload

    changed = False

    for rule in settings.schema_changes:

        if rule["source"] != event.source:
            continue

        field = rule["field"]

        if field not in payload or payload[field] is None:
            continue

        moment = datetime.fromisoformat(rule["from"])

        if event.event_time < moment:
            if not changed:
                payload = dict(payload)
                changed = True
            payload[field] = None

    return replace(event, payload=payload) if changed else event


def _apply_field_missing(event: Event, rng) -> Event:

    settings = params_module.active().defects

    rules = settings.field_missing.get(event.source)

    if not rules:
        return event

    payload = event.payload
    changed = False

    for field, (share, _reason) in rules.items():

        if field not in payload or payload[field] is None:
            continue

        if rng.random() >= share:
            continue

        if not changed:
            payload = dict(payload)
            changed = True

        if field == "firebase_screen":
            payload[field] = settings.ga4_not_set
        else:
            payload[field] = None

    return replace(event, payload=payload) if changed else event


def apply(events: list) -> list:
    """
    Наблюдаемая лента: к каждому событию применяются дефекты
    его источника.

    Записи не теряются: выгрузка показывает всё, что случилось,
    а отсутствие строки означает, что события не было.
    """

    observed: list[Event] = []

    for event in events:

        rng = _rng(event, 1)

        current = _apply_schema_change(event)
        current = _apply_field_missing(current, rng)

        observed.append(current)

    return observed


def plan_refunds(purchases: list) -> list:
    """
    Какие покупки будут возвращены или отменены.

    Возврат ссылается на исходную операцию и не превышает её
    сумму.
    """

    settings = params_module.active().defects

    planned = []

    for event in purchases:

        if event.payload.get("status") != "approved":
            continue

        rng = _rng(event, 5)

        amount = int(event.payload.get("amount") or 0)

        if amount <= 0:
            continue

        if rng.random() < settings.reversal_share:
            hours = rng.integers(*settings.reversal_delay_hours)
            planned.append(
                {
                    "kind": "reversal",
                    "cause": event,
                    "ts": event.event_time + timedelta(hours=int(hours)),
                    "amount": amount,
                }
            )
            continue

        if rng.random() < settings.refund_share:
            days = rng.integers(*settings.refund_delay_days)
            share = rng.uniform(0.25, 0.9) if rng.random() < settings.partial_refund_share else 1.0
            value = max(100, int(amount * share))
            planned.append(
                {
                    "kind": "refund",
                    "cause": event,
                    "ts": event.event_time + timedelta(days=int(days), hours=int(rng.integers(9, 20))),
                    "amount": min(amount, value),
                }
            )

    return planned


__all__ = ["apply", "plan_refunds"]
