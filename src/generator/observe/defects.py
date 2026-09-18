from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

from .. import params as params_module
from ..rng import NS_OBSERVE, keyed_rng, stable_hash
from . import coverage
from .envelope import Event, round_to_precision


# ============================================================
# ДЕФЕКТЫ НАБЛЮДАЕМОСТИ
# ============================================================
#
# Идеально чистая история нереалистична. Дефект принадлежит
# ИСТОЧНИКУ, а не равномерному шуму по всем данным:
#
#   поздние записи
#   дубли и технические повторы
#   исправления: тот же event_id, версия выше
#   пропуски полей с причиной
#   сбой источника: записи дня не доходят вовсе
#   смена схемы: поле начинает собираться с определённой даты
#   огрубление точности времени
#
# Проход идёт по идентичности события, а не по номеру строки:
# вставка события в другом месте ленты ничего не сдвигает.
# ============================================================


# Источники, чья запись несёт проводку. Огрубление времени
# переставило бы её относительно соседних движений по счёту, и
# наблюдаемый остаток перестал бы продолжать предыдущий. Время
# теряет витрина договоров, а не касса.
MONEY_SOURCES: frozenset[str] = frozenset({"transactions", "loans"})


def _rng(event: Event, slot: int):
    return keyed_rng(NS_OBSERVE, stable_hash(event.event_id) % (2 ** 31), slot)


def _paired(event: Event) -> bool:
    """
    Запись, у которой есть обязательная вторая половина.
    """

    if event.link_type == "transfer":
        return True

    if event.payload.get("counterparty") == "own_account":
        return True

    return event.event_type == "fee_charge" and event.payload.get("reason") == "transfer_fee"


def _outage_rng(client_ordinal: int, source: str, ts: datetime):
    """
    Судьба записей сбойного дня общая для всего источника.

    Решать по каждой записи отдельно нельзя: тогда из пары
    проводок одного перевода долетала бы ровно одна, и деньги
    переставали бы сходиться.
    """

    return keyed_rng(
        NS_OBSERVE,
        client_ordinal,
        stable_hash(source) % (2 ** 31),
        ts.toordinal(),
        9,
    )


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


def _coarse_precision(event: Event, rng) -> Event:

    settings = params_module.active().defects

    if event.source in MONEY_SOURCES:
        return event

    share = settings.coarse_precision_share.get(event.source, 0.0)

    if share <= 0.0 or rng.random() >= share:
        return event

    if event.time_precision == "second":
        precision = "minute"
    elif event.time_precision == "minute":
        precision = "day"
    else:
        precision = "day"

    return replace(
        event,
        time_precision=precision,
        event_time=round_to_precision(event.event_time, precision),
    )


def plan_correction(event: Event, rng) -> tuple[str, object] | None:
    """
    Какое поле записи витрина сначала записала неверно.

    Исправление НЕ портит запись. Наоборот: первая версия
    уходит с ошибкой, а исправление возвращает настоящее
    значение. Иначе итоговая версия ленты противоречила бы
    остаткам, а препроцессинг читает именно итоговую.
    """

    fields = params_module.active().defects.correction_fields.get(event.source)

    if not fields:
        return None

    for field in fields:

        value = event.payload.get(field)

        if value is None:
            continue

        if isinstance(value, bool) or not isinstance(value, int):
            continue

        wrong = int(value * rng.uniform(0.94, 1.06))

        if wrong == value:
            wrong = value + (1 if rng.random() < 0.5 else -1)

        return field, wrong

    return None


def apply(events: list, client_ordinal: int) -> tuple[list, dict, list]:
    """
    Наблюдаемая лента: к каждому событию применяются дефекты
    его источника.

    Возвращает тройку: наблюдаемые записи, план ошибок первой
    версии `{event_id: (поле, неверное значение)}` и записи,
    которых сбой источника не донёс до выгрузки вовсе.

    Потерянная запись это потеря НАБЛЮДЕНИЯ, а не отмена события:
    деньги по ней двигались, остаток счёта её учёл, и переписывать
    остальную цепочку так, будто её не было, нельзя. Поэтому
    потери возвращаются наружу и уходят в скрытую истину.

    Ошибка первой версии вносится ПОСЛЕ простановки остатков,
    чтобы цепочка строилась по настоящей сумме, а не по опечатке
    витрины.
    """

    settings = params_module.active().defects

    observed: list[Event] = []
    corrections: dict[str, tuple] = {}
    lost: list[Event] = []

    # Событие, на которое кто-то ссылается как на причину, не
    # имеет права пропасть: иначе возврат будет указывать на
    # покупку, которой в данных нет.
    referenced = {
        event.payload.get("cause_event_id")
        for event in events
        if event.payload.get("cause_event_id")
    }

    for event in events:

        # --- сбой источника: записи дня не доходят ---

        # Сбой витрины не имеет права потерять одну сторону
        # парной проводки: у перевода вторая сторона живёт у
        # другого клиента, у перевода между своими счетами обе
        # стороны у одного, а комиссия привязана к переводу.
        # В любом из этих случаев пропажа половины разрушила бы
        # денежную связность.
        if (
            not _paired(event)
            and event.event_id not in referenced
            and coverage.in_outage(client_ordinal, event.source, event.event_time)
        ):

            recover_rng = _outage_rng(client_ordinal, event.source, event.event_time)

            if recover_rng.random() >= settings.outage_recovers_share:
                lost.append(event)
                continue

            # Источник восстановился и досдал записи. Времени
            # поступления у выгрузки нет, поэтому досланная
            # запись неотличима от обычной.

        rng = _rng(event, 1)

        current = _apply_schema_change(event)
        current = _apply_field_missing(current, rng)
        current = _coarse_precision(current, rng)

        observed.append(current)

        # --- дубль: та же запись ещё раз ---

        duplicate_share = settings.duplicate_share.get(event.source, 0.0)

        if duplicate_share and rng.random() < duplicate_share:
            observed.append(current.copy_as_duplicate())

        # --- исправление: версия выше ---
        #
        # Исправление несёт НАСТОЯЩЕЕ значение. Ошибка попадёт
        # в первую версию позже, когда остатки уже посчитаны.

        correction_share = settings.correction_share.get(event.source, 0.0)

        if correction_share and rng.random() < correction_share:

            plan = plan_correction(current, rng)

            if plan is not None:
                observed.append(current.copy_as_correction(dict(current.payload)))
                corrections[current.event_id] = plan

    return observed, corrections, lost


def apply_first_version_errors(observed: list, corrections: dict) -> None:
    """
    Вносит ошибку витрины в первую версию записи.

    Делать это раньше нельзя: остатки считаются по настоящим
    суммам, и опечатка испортила бы всю цепочку. После пересчёта
    неверное значение остаётся ровно в той версии, которую банк
    потом исправил.
    """

    if not corrections:
        return

    for event in observed:

        plan = corrections.get(event.event_id)

        if plan is None or event.event_version > 1:
            continue

        field, wrong = plan

        if field in event.payload:
            event.payload[field] = wrong


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


__all__ = ["MONEY_SOURCES", "apply", "plan_refunds"]
