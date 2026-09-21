from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .schema import MENTIONS_SCHEMA, TRANSFERS_SCHEMA


# ============================================================
# ИДЕЯ
# ============================================================
#
# Наблюдаемые упоминания и переходы, а не итоговое состояние.
#
# Строка таблицы упоминаний говорит: «в этом событии назван этот
# счёт/карта/договор/заявка/обращение/предложение». Является ли
# упоминание переходом жизненного цикла, решает тип события, и
# список переходов задан явно, а не выведен по догадке.
#
# Состояние на дату собирается позже, на этапе истории, и только
# из того, что известно к cutoff. Здесь состояния нет.
# ============================================================


ENTITY_FIELDS: dict[str, str] = {
    "account_id": "account",
    "card_id": "card",
    "contract_id": "contract",
    "application_id": "application",
    "case_id": "case",
    "offer_id": "offer",
}

# (вид сущности, тип события) -> имя перехода. Всё, чего здесь
# нет, остаётся упоминанием без перехода.
TRANSITIONS: dict[tuple[str, str], str] = {
    ("account", "account_opened"): "opened",
    ("card", "card_activated"): "activated",
    ("card", "card_blocked"): "blocked",
    ("card", "card_unblocked"): "unblocked",
    ("card", "card_reissued"): "reissued",
    ("contract", "account_opened"): "opened",
    ("contract", "product_opened"): "opened",
    ("contract", "contract_terms_changed"): "terms_changed",
    ("contract", "product_repriced"): "repriced",
    ("contract", "product_renewed"): "renewed",
    ("contract", "product_migrated"): "migrated",
    ("contract", "product_closed"): "closed",
    ("contract", "schedule_created"): "schedule_created",
    ("contract", "installment_due"): "installment_due",
    ("contract", "installment_paid"): "installment_paid",
    ("contract", "installment_missed"): "installment_missed",
    ("contract", "delinquency_registered"): "delinquency_registered",
    ("contract", "arrears_cleared"): "arrears_cleared",
    ("contract", "loan_restructured"): "restructured",
    ("contract", "early_repayment"): "early_repayment",
    ("contract", "loan_closed"): "closed",
    ("application", "application_submitted"): "submitted",
    ("application", "application_decision"): "decided",
    ("case", "case_opened"): "opened",
    ("case", "case_updated"): "updated",
    ("case", "case_resolved"): "resolved",
}

# Стороны перевода по типу события.
TRANSFER_SIDES: dict[str, str] = {
    "transfer_out": "out",
    "p2p_out": "out",
    "transfer_in": "in",
    "p2p_in": "in",
}

PAIR_BOTH = "both_sides"
PAIR_ONE = "one_side_observed"

MENTION_COLUMNS = (
    "client_idx",
    "client_id",
    "event_id",
    "stable_event_index",
    "event_time",
    "event_type",
    "source",
    "raw_row",
)


def extract_mentions(table: pa.Table) -> pa.Table:
    """
    Все упоминания сущностей в пачке canonical.
    """

    pieces: list[pa.Table] = []

    event_type = table.column("event_type")

    for field_name, kind in ENTITY_FIELDS.items():

        if field_name not in table.column_names:
            continue

        column = table.column(field_name)

        mask = pc.is_valid(column)
        count = int(pc.sum(mask).as_py() or 0)

        if count == 0:
            continue

        rows = table.filter(mask)

        types = rows.column("event_type").to_pylist()
        transitions = [TRANSITIONS.get((kind, value)) for value in types]

        piece = pa.table(
            {
                "entity_kind": pa.array([kind] * count, pa.string()),
                "entity_id": rows.column(field_name).cast(pa.string()),
                **{name: rows.column(name) for name in MENTION_COLUMNS},
                "field_name": pa.array([field_name] * count, pa.string()),
                "is_transition": pa.array([value is not None for value in transitions]),
                "transition": pa.array(transitions, pa.string()),
            }
        )

        pieces.append(piece.select(MENTIONS_SCHEMA.names).cast(MENTIONS_SCHEMA))

    if not pieces:
        return MENTIONS_SCHEMA.empty_table()

    combined = pa.concat_tables(pieces)

    # Устойчивый порядок: клиент, его лента, вид, идентификатор.
    order = np.lexsort(
        (
            np.asarray(combined.column("entity_id").to_pylist(), dtype=object).argsort().argsort(),
            np.asarray(combined.column("entity_kind").to_pylist(), dtype=object).argsort().argsort(),
            np.asarray(combined.column("stable_event_index").to_pylist(), dtype=np.int64),
            np.asarray(combined.column("client_idx").to_pylist(), dtype=np.int64),
        )
    )

    return combined.take(pa.array(order))


def extract_transfers(table: pa.Table) -> list[dict]:
    """
    Стороны переводов пачки. Пара собирается позже, когда
    просмотрены все клиенты: вторая сторона живёт у другого.

    Сторона узнаётся по ТИПУ СОБЫТИЯ (TRANSFER_SIDES) и непустому
    transfer_id в payload — тем же правилом, что и индекс
    переводов истории. Тип операции решает всё: комиссия за
    перевод несёт тот же transfer_id, но стороной перевода не
    является. Раньше отбор шёл по метке связи в конверте, и
    комиссия становилась стороной без направления.
    """

    if "event_type" not in table.column_names or "transfer_id" not in table.column_names:
        return []

    mask = pc.and_(
        pc.is_in(table.column("event_type"), value_set=pa.array(sorted(TRANSFER_SIDES))),
        pc.is_valid(table.column("transfer_id")),
    )

    rows = table.filter(mask)

    if rows.num_rows == 0:
        return []

    out: list[dict] = []

    for row in rows.to_pylist():

        out.append(
            {
                "transfer_id": row["transfer_id"],
                "side": TRANSFER_SIDES.get(row["event_type"]),
                "client_idx": row["client_idx"],
                "client_id": row["client_id"],
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "stable_event_index": row["stable_event_index"],
                "event_time": row["event_time"],
                "amount": row.get("amount"),
                "direction": row.get("direction"),
                "status": row.get("status"),
                "counterparty": row.get("counterparty"),
                "raw_row": row["raw_row"],
            }
        )

    return out


def order_transfers(sides: list[dict]) -> pa.Table:
    """
    Наблюдаемые стороны переводов в устойчивом порядке.

    Парность здесь НЕ вычисляется: обе стороны внутрибанковского
    перевода живут у разных клиентов, и кто с кем сошёлся на
    дату, решает этап истории среди видимых строк.

    Одна наблюдаемая сторона и там не будет означать потерю:
    вторая либо вне банка, либо у клиента за пределами выборки, и
    по данным эти причины неразличимы.
    """

    ordered = sorted(
        sides, key=lambda row: (row["client_idx"], row["stable_event_index"], row["raw_row"])
    )

    return pa.Table.from_pylist(ordered, schema=TRANSFERS_SCHEMA)


__all__ = [
    "ENTITY_FIELDS",
    "PAIR_BOTH",
    "PAIR_ONE",
    "TRANSFER_SIDES",
    "TRANSITIONS",
    "order_transfers",
    "extract_mentions",
    "extract_transfers",
]
