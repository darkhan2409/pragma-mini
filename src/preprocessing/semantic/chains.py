from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from . import time as time_module


# ============================================================
# ИДЕЯ
# ============================================================
#
# Цепочка это связь между видимыми событиями, а не догадка о
# том, чем дело кончилось. Незавершённая цепочка так и остаётся
# незавершённой: у заявки без решения исход in_progress, и
# придумывать ему конец нельзя.
#
# Связи берутся из служебных полей слоя — correlation_id и
# cause_event_id, — но наружу они не выходят. В признаки идут
# смысловые следствия: тип связанного события, вид связи, сколько
# прошло между ними и та же ли это точка.
#
# Всё считается по событиям, видимым на cutoff. Более поздний шаг
# цепочки в неё не попадает, даже если в выгрузке он есть.
# ============================================================


CHAINS_VERSION = "1.3.0"

IN_PROGRESS = "in_progress"

TIME_ORDER_AMBIGUOUS = "time_order_ambiguous"

# Разрешение объявленной точности в сутках: внутри него источник
# порядок двух записей не различает.
PRECISION_DAYS: dict[str, float] = {
    "second": 0.0,
    "minute": 1.0 / 1440.0,
    "day": 1.0,
}

DATE_ONLY = time_module.DATE_ONLY

# Вид цепочки задаёт метка связи первого шага, а не догадка по
# типу события: одна и та же покупка может быть шагом сессии и
# шагом возврата.
KIND_OF_LINK: dict[str, str] = {
    "offer": "application",
    "application": "application",
    "contract": "contract",
    "schedule": "contract",
    "case": "case",
    "fraud_episode": "fraud",
    "transfer": "transfer",
    "session": "session",
    "refund": "operation",
    "reversal": "operation",
    "chargeback": "operation",
}

# Шаг, который закрывает цепочку своего вида. Всё остальное
# оставляет её незавершённой.
TERMINAL_TYPES: dict[str, frozenset[str]] = {
    "application": frozenset({"application_decision", "product_opened"}),
    # Снятие с вклада договор НЕ закрывает: частичное снятие это
    # обычная операция, и объявлять её исходом значит закрывать
    # действующий вклад. Закрывают только явные события.
    "contract": frozenset({"product_closed", "loan_closed"}),
    "case": frozenset({"case_resolved"}),
    "fraud": frozenset({"fraud_decision"}),
    "operation": frozenset({"refund", "reversal", "chargeback"}),
    "transfer": frozenset({"p2p_in", "transfer_in"}),
    "session": frozenset(),
}

# Вид связи по типу события-следствия.
RELATION_OF_TYPE: dict[str, str] = {
    "refund": "refund",
    "reversal": "reversal",
    "chargeback": "chargeback",
    "application_decision": "decision",
    "product_opened": "contract",
    "installment_paid": "payment",
    "installment_due": "schedule",
    "case_resolved": "resolution",
    "case_updated": "update",
    "fraud_decision": "decision",
}


@dataclass(frozen=True)
class Chain:
    kind: str
    started_at: datetime
    last_step_at: datetime
    steps: int
    first_event_type: str
    last_event_type: str
    outcome: str

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "started_at": self.started_at,
            "last_step_at": self.last_step_at,
            "steps": self.steps,
            "first_event_type": self.first_event_type,
            "last_event_type": self.last_event_type,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class Relation:
    """
    Смысловая связь события с его причиной. Идентификаторы
    остаются внутри слоя, наружу идёт смысл.

    days_since_related_event пусто, когда порядок двух записей
    неизвестен: источник объявил время грубее их разницы. Само
    наблюдение при этом сохраняется как есть — observed_days не
    обнуляется и не переворачивается, он объясняет причину.
    """

    stable_event_index: int
    related_event_type: str
    relation_type: str
    days_since_related_event: float | None
    same_merchant: bool | None
    observed_days: float
    reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "stable_event_index": self.stable_event_index,
            "related_event_type": self.related_event_type,
            "relation_type": self.relation_type,
            "days_since_related_event": self.days_since_related_event,
            "same_merchant": self.same_merchant,
            "observed_days": self.observed_days,
            "reason": self.reason,
        }


class ChainsError(ValueError):
    """
    Связь событий противоречит времени, и точностью источника это
    не объясняется.
    """


def _precision(row: dict) -> str:
    """
    Объявленная точность записи по общему правилу слоя.
    """

    try:
        return time_module.effective_precision(row)
    except time_module.PrecisionError as error:
        raise ChainsError(f"{error}: судить о порядке событий по ней нельзя") from error


def _known_days(row: dict, cause: dict) -> float:
    """
    Сутки между причиной и следствием в точности пары: оба момента
    усечены до более грубой из двух объявленных точностей. У
    дневной записи источник знает только дату, и дробные сутки
    между ней и причиной были бы выдумкой.
    """

    precision = time_module.coarser(_precision(row), _precision(cause))

    later = time_module.floor_to_precision(row["event_time"], precision)
    earlier = time_module.floor_to_precision(cause["event_time"], precision)

    return (later - earlier).total_seconds() / 86400.0


def _resolution(row: dict) -> float:
    """
    Насколько грубо источник знает время этой записи, в сутках.

    Дневное качество отметки в payload значит то же, что дневная
    точность источника: час и минута не наблюдались. Правило одно
    на весь слой — time.effective_precision.
    """

    return PRECISION_DAYS[_precision(row)]


def _interval(row: dict, cause: dict) -> tuple[float | None, float, str | None]:
    """
    Интервал между причиной и следствием как признак модели, само
    наблюдение и причина, если признака нет.

    Отрицательный интервал это не признак: причина не может
    произойти после следствия. Объяснить его может только
    объявленная точность — тогда порядок внутри её разрешения
    неизвестен, и признак не передаётся. Если обе записи точны,
    это противоречие данных, а не особенность времени.
    """

    observed = (row["event_time"] - cause["event_time"]).total_seconds() / 86400.0

    if observed >= 0:
        # Признак — в объявленной точности пары; наблюдение
        # остаётся сырым: оно объясняет признак, а не подменяет его.
        return _known_days(row, cause), observed, None

    tolerance = max(_resolution(row), _resolution(cause))

    if tolerance > 0.0 and -observed <= tolerance:
        return None, observed, TIME_ORDER_AMBIGUOUS

    raise ChainsError(
        f"причина {cause['event_type']} записана позже следствия {row['event_type']} "
        f"на {abs(observed):.4f} суток, а объявленная точность различает "
        f"{tolerance:.4f} суток: порядок противоречит данным"
    )


def _kind_of(link_type: str | None, event_type: str) -> str:

    kind = KIND_OF_LINK.get(link_type or "")

    if kind is not None:
        return kind

    if event_type in TERMINAL_TYPES["operation"]:
        return "operation"

    return "other"


def chains(rows: list[dict]) -> list[Chain]:
    """
    Цепочки по correlation_id среди видимых событий.
    """

    grouped: dict[str, list[dict]] = {}

    for row in rows:

        link = row.get("correlation_id")

        if link is None:
            continue

        grouped.setdefault(link, []).append(row)

    out: list[Chain] = []

    for _link, steps in sorted(grouped.items(), key=lambda item: item[1][0]["event_time"]):

        # Цепочка из одного шага это НАЧАВШАЯСЯ цепочка, а не
        # отсутствие цепочки: заявка без видимого решения и
        # покупка без возврата обязаны остаться в наблюдении с
        # исходом in_progress. Раньше они исчезали, и доля
        # незавершённых считалась по неполному знаменателю.
        steps = sorted(steps, key=lambda row: (row["event_time"], row["stable_event_index"]))

        first, last = steps[0], steps[-1]

        kind = _kind_of(first.get("link_type"), first["event_type"])

        terminal = TERMINAL_TYPES.get(kind, frozenset())

        # Исход это ЗАКРЫВШИЙ шаг, а не просто последний: после
        # решения по заявке в цепочке идут платежи, и называть
        # исходом платёж было бы неправдой.
        closing = next((step for step in steps if step["event_type"] in terminal), None)

        out.append(
            Chain(
                kind=kind,
                started_at=first["event_time"],
                last_step_at=last["event_time"],
                steps=len(steps),
                first_event_type=first["event_type"],
                last_event_type=last["event_type"],
                outcome=closing["event_type"] if closing is not None else IN_PROGRESS,
            )
        )

    return out


def relations(rows: list[dict]) -> list[Relation]:
    """
    Смысловые признаки связи вместо сырого cause_event_id.

    На вход идут строки вместе с их смыслом: точка сравнивается
    по локальной ссылке, а не по сырому идентификатору.

    Причина, которой не видно на cutoff, связью не становится:
    цепочка остаётся оборванной честно.
    """

    by_id = {row["event_id"]: row for row in rows}

    out: list[Relation] = []

    for row in rows:

        cause_id = row.get("cause_event_id")

        if cause_id is None:
            continue

        cause = by_id.get(cause_id)

        if cause is None:
            continue

        # Точка это прежде всего конкретная торговая точка; сеть
        # отвечает на вопрос, только когда точки не названы.
        same_merchant: bool | None = None

        for column in ("outlet_ref", "merchant_ref"):
            if row.get(column) is not None or cause.get(column) is not None:
                same_merchant = row.get(column) == cause.get(column)
                break

        days, observed, reason = _interval(row, cause)

        out.append(
            Relation(
                stable_event_index=row["stable_event_index"],
                related_event_type=cause["event_type"],
                relation_type=RELATION_OF_TYPE.get(row["event_type"], "caused_by"),
                days_since_related_event=days,
                same_merchant=same_merchant,
                observed_days=observed,
                reason=reason,
            )
        )

    return out


def chain_summary(items: list[Chain]) -> dict:

    by_kind: dict[str, int] = {}
    unfinished = 0

    for item in items:
        by_kind[item.kind] = by_kind.get(item.kind, 0) + 1
        if item.outcome == IN_PROGRESS:
            unfinished += 1

    return {
        "chains": len(items),
        "by_kind": dict(sorted(by_kind.items())),
        "unfinished": unfinished,
        "rule": "цепочка считается по видимым шагам; отсутствие конца это in_progress, а не выдуманный исход",
    }


__all__ = [
    "CHAINS_VERSION",
    "DATE_ONLY",
    "KIND_OF_LINK",
    "IN_PROGRESS",
    "PRECISION_DAYS",
    "TIME_ORDER_AMBIGUOUS",
    "ChainsError",
    "RELATION_OF_TYPE",
    "TERMINAL_TYPES",
    "Chain",
    "Relation",
    "chain_summary",
    "chains",
    "relations",
]
