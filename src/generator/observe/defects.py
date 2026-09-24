from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

from .. import params as params_module
from ..config import PAYLOAD_REQUIRED
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
# Несобранное поле ОТСУТСТВУЕТ, как любое неприменимое
# (envelope.py): ключ удаляется, а не получает null. Обязательное
# поле события дефект не трогает: без него запись нарушала бы
# контракт, а не была бы просто неполной.
#
# Чего здесь БОЛЬШЕ НЕТ: дублей, исправлений и сбоев источника.
# Запись приходит в выгрузку один раз и сразу окончательной, а
# отсутствие записи означает ровно одно: события не было.
#
# Проход идёт по идентичности события, а не по номеру строки:
# вставка события в другом месте ленты ничего не сдвигает.
# ============================================================


def _keys(events: list) -> list:
    """
    Ключ случайности каждого события: клиент, тип, время, сумма и
    номер среди совпадающих по всему этому.

    Номер выдачи для этого не годится. Событие, датированное за
    концом окна, короткая выгрузка не создаёт вовсе
    (engine_products.py:1351 и соседние), и нумерация всех
    следующих событий клиента съезжает на единицу — вместе с ней
    съезжали бы дефекты наблюдения, хотя сами события те же.
    Ключ по содержанию от границы выгрузки не зависит.

    От места строки в файле он тоже не зависит: считается по
    самому событию, а вставка другого события ничего не двигает.
    """

    seen: dict[tuple, int] = {}

    result: list[int] = []

    for event in events:

        moment = event.event_time.isoformat()
        amount = event.payload.get("amount")

        group = (event.event_type, moment, amount)

        index = seen[group] = seen.get(group, 0) + 1

        result.append(
            stable_hash(event.client_id, event.event_type, moment, amount, index) % (2 ** 31)
        )

    return result


def _rng(key: int, slot: int):
    return keyed_rng(NS_OBSERVE, key, slot)


def _apply_schema_change(event: Event) -> Event:
    """
    Поле начинает собираться с определённой даты: до неё его
    в записи нет.
    """

    settings = params_module.active().defects

    payload = event.payload

    changed = False

    for rule in settings.schema_changes:

        if rule["source"] != event.source:
            continue

        field = rule["field"]

        if field not in payload or field in _required(event):
            continue

        moment = datetime.fromisoformat(rule["from"])

        if event.event_time < moment:
            if not changed:
                payload = dict(payload)
                changed = True
            del payload[field]

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

        # GA4 пишет в обязательное поле экрана своё «(not set)»:
        # это его настоящее значение, а не пропуск.
        if field == "firebase_screen":
            payload[field] = settings.ga4_not_set
        elif field not in _required(event):
            del payload[field]

    return replace(event, payload=payload) if changed else event


def _required(event: Event) -> frozenset:
    return PAYLOAD_REQUIRED.get(event.event_type, frozenset())


def apply(events: list) -> list:
    """
    Наблюдаемая лента: к каждому событию применяются дефекты
    его источника.

    Записи не теряются: выгрузка показывает всё, что случилось,
    а отсутствие строки означает, что события не было.
    """

    observed: list[Event] = []

    for event, key in zip(events, _keys(events)):

        rng = _rng(key, 1)

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

    for event, key in zip(purchases, _keys(purchases)):

        if event.payload.get("status") != "approved":
            continue

        rng = _rng(key, 5)

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
