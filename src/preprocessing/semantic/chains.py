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
# Цепочки собираются по деловым ключам payload: заявке,
# договору, обращению, сессии и переводу. Ссылки на
# событие-причину в данных нет, и признаков связи с ней тоже:
# причинность не восстанавливается ни полем, ни догадкой.
#
# Всё считается по событиям, видимым на cutoff. Более поздний шаг
# цепочки в неё не попадает, даже если в выгрузке он есть.
# ============================================================


CHAINS_VERSION = "4.0.0"

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
        closing = next((step for step in steps if step["type"] in terminal), None)

        out.append(
            Chain(
                kind=kind,
                started_at=first["event_time"],
                last_step_at=last["event_time"],
                steps=len(steps),
                first_event_type=first["type"],
                last_event_type=last["type"],
                outcome=closing["type"] if closing is not None else IN_PROGRESS,
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
    "TERMINAL_TYPES",
    "Chain",
    "chain_summary",
    "chains",
]
