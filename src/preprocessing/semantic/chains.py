from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime



# ============================================================
# ИДЕЯ
# ============================================================
#
# Цепочка это связь между видимыми событиями, а не догадка о
# том, чем дело кончилось. Незавершённая цепочка так и остаётся
# незавершённой: у заявки без решения исход in_progress, и
# придумывать ему конец нельзя.
#
# Связи берутся из деловых ключей payload — cause_event_id и
# идентификаторов заявки, договора, обращения, сессии и
# перевода, — но наружу они не выходят. В признаки идут
# смысловые следствия: тип связанного события, вид связи, сколько
# прошло между ними и та же ли это точка.
#
# Всё считается по событиям, видимым на cutoff. Более поздний шаг
# цепочки в неё не попадает, даже если в выгрузке он есть.
# ============================================================


CHAINS_VERSION = "2.0.0"

IN_PROGRESS = "in_progress"

# Вид цепочки задаёт деловой ключ payload, по которому она
# собрана. Метки связи в конверте больше нет, и догадываться по
# типу события не нужно: одна и та же покупка честно входит и в
# цепочку договора, и в цепочку сессии.
KIND_OF_FIELD: dict[str, str] = {
    "application_id": "application",
    "offer_id": "application",
    "contract_id": "contract",
    "case_id": "case",
    "session_id": "session",
    "transfer_id": "transfer",
}

# Порядок обхода ключей фиксирован: по нему собираются цепочки,
# и от него зависит порядок строк в отчёте.
CHAIN_FIELDS: tuple[str, ...] = tuple(KIND_OF_FIELD)

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


def _interval(row: dict, cause: dict) -> tuple[float | None, float, str | None]:
    """
    Интервал между причиной и следствием как признак модели и
    само наблюдение.

    Время события точное, поэтому отрицательный интервал больше
    ничем не объясняется: причина не может произойти после
    следствия, и это противоречие данных.
    """

    observed = (row["event_time"] - cause["event_time"]).total_seconds() / 86400.0

    if observed >= 0:
        return observed, observed, None

    raise ChainsError(
        f"причина {cause['event_type']} записана позже следствия {row['event_type']} "
        f"на {abs(observed):.4f} суток: время событий точное, и порядок противоречит данным"
    )


def chains(rows: list[dict]) -> list[Chain]:
    """
    Цепочки по деловым ключам payload среди видимых событий.

    Одна запись может войти в несколько цепочек: платёж из
    приложения принадлежит и договору, и сессии. Это два разреза
    одной ленты, а не двойной счёт.
    """

    grouped: dict[tuple[str, str], list[dict]] = {}

    for row in rows:

        for field_name in CHAIN_FIELDS:

            link = row.get(field_name)

            if link is None:
                continue

            grouped.setdefault((field_name, link), []).append(row)

    out: list[Chain] = []

    for (field_name, _link), steps in sorted(
        grouped.items(), key=lambda item: (item[1][0]["event_time"], item[0])
    ):

        # Цепочка из одного шага это НАЧАВШАЯСЯ цепочка, а не
        # отсутствие цепочки: заявка без видимого решения и
        # покупка без возврата обязаны остаться в наблюдении с
        # исходом in_progress. Раньше они исчезали, и доля
        # незавершённых считалась по неполному знаменателю.
        steps = sorted(steps, key=lambda row: (row["event_time"], row["stable_event_index"]))

        first, last = steps[0], steps[-1]

        kind = KIND_OF_FIELD[field_name]

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
    "IN_PROGRESS",
    "ChainsError",
    "RELATION_OF_TYPE",
    "TERMINAL_TYPES",
    "Chain",
    "Relation",
    "chain_summary",
    "chains",
    "relations",
]
