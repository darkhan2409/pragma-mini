from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from src.preprocessing.canonical.events import _order, normalize_column, normalize_text
from src.preprocessing.rawdata import (
    TIMESTAMP_UTC,
    TYPE_KEY,
    RawContractError,
    event_times,
    parse_event_time,
    strict_event_times,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Быстрые пути препроцессинга обязаны давать то же, что прежние
# построчные:
#
#   - время: векторный разбор строгой записи принимает только то,
#     что parse_event_time принимает с тем же моментом; всё прочее
#     уходит на построчный разбор с той же ошибкой;
#   - нормализация текста по словарю колонки равна normalize_text
#     каждого значения, включая юникодные тонкости и пустые строки;
#   - приоритет типа по справочнику и сортировка сгруппированной по
#     типам таблицы с номером строки последним ключом дают тот же
#     порядок, что прежний поиск в словаре и устойчивая сортировка
#     исходной пачки.
# ============================================================


UTC = timezone.utc

# Записи, на которых правила разбора времени расходятся легче всего.
EDGE_TIMES = [
    "2025-01-01T24:00:00+05:00", "2025-01-01T23:59:60+05:00", "2025-01-01T10:00:00+24:00",
    "2025-01-01T10:00:00+23:59", "2024-02-29T10:00:00+05:00", "2025-02-29T10:00:00+05:00",
    "2000-02-29T00:00:00-05:30", "1900-02-29T00:00:00+05:00", "2100-02-28T23:59:59+00:00",
    "2025-01-01T10:00:00.123+05:00", "2025-01-01T10:00:00-00:00", "2025-13-01T10:00:00+05:00",
    "2025-00-10T10:00:00+05:00", "2025-01-00T10:00:00+05:00", "2025-01-01 10:00:00+05:00",
    "2025-01-01T10:00:00", "2025-01-01T10:00:00Z", " 2025-01-01T10:00:00+05:00",
    "2025-01-01T10:00:00.12+05:00", "2025-01-01T10:00:00.123456+05:00", "2025-01-01T10:00+05:00",
    "2025-04-31T10:00:00+05:00", "2025-12-31T23:59:59.999-12:45", "1969-12-31T23:59:59+00:00",
    "", "２０２５-01-01T10:00:00+05:00", "2025-01-01T10:00:00.1x3+05:00", "2025-01-01T10:00:00+05-00",
    "2025/01/01T10:00:00+05:00", "2025-01-01t10:00:00+05:00", "2025-01-01T10:00:00.123+0500",
    "2025-01-01T10:00:00+05:00\n",
]


def python_times(texts: list) -> pa.Array | type:
    """Прежний путь: parse_event_time по строке, или класс ошибки."""
    try:
        return pa.array([parse_event_time(text) for text in texts], TIMESTAMP_UTC)
    except (RawContractError, OverflowError) as error:
        return type(error)


@pytest.mark.parametrize("text", EDGE_TIMES)
def test_the_fast_time_path_accepts_only_what_python_accepts_with_the_same_moment(text):

    good = "2025-03-01T09:00:00+05:00"

    for column in (pa.array([text]), pa.array([good, text, good]).slice(1)):

        expected = python_times(column.to_pylist())
        fast = strict_event_times(column)

        if fast is not None:
            assert isinstance(expected, pa.Array) and fast.equals(expected), text


def test_the_fast_time_path_matches_python_on_random_moments():

    rng = random.Random(5)
    start = datetime(1950, 1, 1, tzinfo=UTC)

    texts = []

    for _ in range(20_000):
        moment = start + timedelta(
            seconds=rng.randrange(0, 150 * 365 * 86400),
            milliseconds=rng.choice([0, 0, rng.randrange(1, 1000)]),
        )
        local = moment.astimezone(timezone(timedelta(minutes=rng.randrange(-23 * 60 - 59, 23 * 60 + 60))))
        texts.append(local.isoformat(timespec="milliseconds") if local.microsecond else local.isoformat())

    column = pa.array(texts)

    for layout in (column, column.slice(3), pa.chunked_array([column.slice(0, 7), column.slice(7)])):
        fast = strict_event_times(layout)
        expected = python_times(layout.to_pylist() if isinstance(layout, pa.Array) else layout.combine_chunks().to_pylist())
        assert fast is not None and fast.equals(expected)


def test_a_bad_time_falls_back_to_the_same_error():

    column = pa.array(["2025-03-01T09:00:00+05:00", "2025-03-02T09:00:00"])

    with pytest.raises(RawContractError) as fast:
        event_times(column)

    with pytest.raises(RawContractError) as slow:
        parse_event_time("2025-03-02T09:00:00")

    assert str(fast.value) == str(slow.value)


# Текст, на котором нормализация легко расходится.
TEXTS = [
    "Магнум", "МАГНУМ", "магнум", "Small  Shop", "small\tshop\n", "Кофе Хаус", "Café",
    "Café", "İstanbul", "ıslak", "STRASSE", "Straße", "ﬁnance", "ＡＢＣ Market", "", "   ",
    " ", "  ", None, "  Алматы  Plaza  ", "Ǆemal", "Ⅻ", "x" * 300, "ὈΔΥΣΣΕΎΣ",
]


def test_normalizing_by_the_dictionary_equals_normalizing_each_value():

    rng = random.Random(9)
    values = [rng.choice(TEXTS) for _ in range(2_000)] + TEXTS

    expected = [normalize_text(value) for value in values]

    for layout in (pa.array(values, pa.string()), pa.chunked_array([pa.array(values[:5], pa.string()),
                                                                    pa.array(values[5:], pa.string())])):
        assert normalize_column(layout).to_pylist() == expected


def _old_order(table: pa.Table, priority: dict[str, int]) -> pa.Array:
    """Прежний порядок: приоритет поиском в словаре, устойчивая сортировка."""

    leading = ["client_id", "event_time", "__type_priority", TYPE_KEY, "source"]
    rest = sorted(name for name in table.column_names if name not in leading and name != "lifelong_source")
    unknown = len(priority)
    ranked = table.append_column(
        "__type_priority",
        pa.array([priority.get(name, unknown) for name in table.column(TYPE_KEY).to_pylist()], pa.int16()),
    )
    return pc.sort_indices(ranked, sort_keys=[(name, "ascending", "at_end") for name in leading + rest])


def test_the_order_of_a_type_grouped_table_equals_the_old_stable_order():

    rng = random.Random(13)
    priority = {"application": 0, "decision": 1, "purchase": 2, "refund": 3}
    types = list(priority) + ["unknown_kind", None]

    rows = 3_000
    table = pa.table({
        "client_id": pa.array([f"c{rng.randrange(5)}" for _ in range(rows)]),
        "event_time": pa.array([rng.randrange(20) for _ in range(rows)], pa.int64()),
        "source": pa.array([rng.choice(["a", "b"]) for _ in range(rows)]),
        TYPE_KEY: pa.array([rng.choice(types) for _ in range(rows)], pa.string()),
        "amount": pa.array([rng.choice([None, 1, 2]) for _ in range(rows)], pa.int64()),
        "lifelong_source": pa.array([rng.choice([None, "first_card_activated"]) for _ in range(rows)], pa.string()),
    })

    expected = table.take(_old_order(table, priority))

    # Та же пачка, сгруппированная по типам: строки переставлены, номер
    # строки в исходной пачке едет рядом.
    positions = np.argsort(np.array([str(item) for item in table.column(TYPE_KEY).to_pylist()]), kind="stable")
    grouped = table.take(pa.array(positions))

    assert grouped.take(_order(grouped, priority, positions)).equals(expected)
