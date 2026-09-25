from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta

import pyarrow.parquet as pq
import pytest

from src.generator import config, emit


# ============================================================
# ИДЕЯ
# ============================================================
#
# Граница выгрузки обязана резать ленту, а не переписывать её.
# Продление окна имеет право добавить продолжение и новых
# клиентов — и не имеет права изменить ни одного прежнего
# события.
#
# Прежде граница участвовала в планировании: от неё считались
# длины всех жизненных отрезков, и продление окна давало другого
# клиента с первого дня. Отдельно от этого решение о трате
# смотрит на следующую зарплату (engine.py:191-203), а список
# выплат обрывался концом окна, и последние недели выгрузки жили
# «без будущего дохода».
#
# Здесь же проверяется второе обещание emit.py: несколько выгрузок
# в одном процессе равны отдельным прогонам. Горизонт, параметры
# и состояние розыгрыша — процессные глобалы, и вторая выгрузка
# могла получить величины, посчитанные под прежним окном.
# ============================================================


CLIENTS = 8
COMMUNITY = 4

START = datetime(2024, 1, 1)
SHORT = datetime(2024, 4, 1)
LONG = datetime(2024, 7, 1)

COLUMNS = ("client_id", "event_time", "source", "payload")


def generate(out, end: datetime) -> dict:

    return emit.generate_dataset(
        total_clients=CLIENTS,
        out_dir=out,
        seed=100,
        world_seed=42,
        history_start=START,
        history_end=end,
        workers=1,
        community_size=COMMUNITY,
        quiet=True,
    )


def tape(directory) -> list[tuple]:

    table = pq.read_table(directory / "events.parquet")

    return list(zip(*[table.column(name).to_pylist() for name in COLUMNS]))


def digest(rows: list[tuple]) -> str:
    """
    Отпечаток ленты по содержимому строк, а не по байтам файла:
    байты parquet зависят ещё и от версии pyarrow.
    """

    value = hashlib.sha256()

    for row in rows:
        value.update(("\x1f".join(row) + "\n").encode("utf-8"))

    return value.hexdigest()


def milestones(directory) -> dict[str, list[tuple]]:

    rows = pq.read_table(directory / "profile.parquet").to_pylist()

    return {
        row["client_id"]: [(item["type"], item["event_time"]) for item in row["lifelong"]]
        for row in rows
    }


# Ленты обоих окон, снятые кодом до появления вех анкеты (коммит
# bb96b3f). Вехи берут готовые даты персоны и новых розыгрышей не
# делают, поэтому лента обязана остаться той же строка в строку.
# Намеренное изменение генератора меняет и эти отпечатки — тогда
# их переснимают вместе с изменением и объясняют почему.
TAPE_BEFORE_LIFELONG = {
    "short": (605, "db2a7e194ea295ecfef9b6c44987426c31571f053bd758632d3edde0282cb4b7"),
    "long": (1218, "94fcb15b52314dceb459f0f595b95e2cdca1cc7bf122c4dbbf4d2df80601a4ae"),
}


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict:
    """
    Три выгрузки ПОДРЯД В ОДНОМ процессе: короткая, длинная и
    снова короткая. Порядок выбран нарочно — он и ловит перенос
    состояния между прогонами.
    """

    base = tmp_path_factory.mktemp("horizon")

    first = base / "short-1"
    long = base / "long"
    again = base / "short-2"

    generate(first, SHORT)
    generate(long, LONG)
    generate(again, SHORT)

    return {"short": first, "long": long, "again": again}


@pytest.fixture(scope="module")
def boundary(runs) -> datetime:
    """
    Граница в том же поясе, в каком записана лента.
    """

    sample = tape(runs["long"])[0][1]

    return SHORT.replace(tzinfo=datetime.fromisoformat(sample).tzinfo)


# ============================================================
# ПРОДЛЕНИЕ ОКНА
# ============================================================


def test_long_window_repeats_the_short_tape_exactly(runs, boundary):
    """
    Общий отрезок совпадает построчно: клиент, время, источник и
    payload — вместе с суммами.
    """

    short = tape(runs["short"])
    prefix = [item for item in tape(runs["long"]) if datetime.fromisoformat(item[1]) < boundary]

    assert len(short) == len(prefix), (
        f"строк до границы: короткое окно {len(short)}, длинное {len(prefix)}"
    )

    for index, (left, right) in enumerate(zip(short, prefix)):
        assert left == right, (
            f"строка {index} разошлась:\n  короткое: {left}\n  длинное:  {right}"
        )


def test_payloads_match_by_meaning_not_only_by_text(runs, boundary):
    """
    Текст мог совпасть случайно; разбор — нет.
    """

    short = tape(runs["short"])
    prefix = [item for item in tape(runs["long"]) if datetime.fromisoformat(item[1]) < boundary]

    for left, right in zip(short, prefix):
        assert json.loads(left[3]) == json.loads(right[3])


def test_no_client_disappears_and_newcomers_have_no_past(runs, boundary):
    """
    Прежние клиенты остаются все. Появиться в длинном окне вправе
    только тот, у кого до прежней границы событий нет вовсе.
    """

    short = tape(runs["short"])
    long = tape(runs["long"])

    before = {item[0] for item in short}
    after = {item[0] for item in long}

    assert not before - after, f"клиенты пропали: {sorted(before - after)}"

    early = {
        item[0] for item in long if datetime.fromisoformat(item[1]) < boundary
    }

    assert early == before, f"в общем отрезке появились новые клиенты: {sorted(early - before)}"


def test_milestones_do_not_rewrite_the_tape(runs):

    for name, (count, expected) in TAPE_BEFORE_LIFELONG.items():

        rows = tape(runs[name])

        assert (len(rows), digest(rows)) == (count, expected), name


def test_long_window_keeps_the_early_milestones(runs, boundary):
    """
    Вехи не зависят от конца окна: всё, что короткая выгрузка
    знала до своей границы, длинная знает так же. Новые вехи
    появляются только после неё.
    """

    short = milestones(runs["short"])
    long = milestones(runs["long"])

    assert any(short.values()), "проверка вырождена: вех нет вовсе"

    for client_id, items in short.items():

        early = [item for item in long[client_id] if item[1] < boundary]

        assert items == early, client_id


def test_long_window_keeps_the_early_employment_records(runs, boundary):
    """
    Записи банка о работе, сделанные до границы короткой выгрузки,
    в длинной те же: ни начало работы, ни момент записи от конца
    окна не зависят.
    """

    def records(directory) -> dict[str, list[tuple]]:
        rows = pq.read_table(directory / "profile.parquet").to_pylist()
        return {
            row["client_id"]: [(item["start_date"], item["record_time"]) for item in row["employment"]]
            for row in rows
        }

    short = records(runs["short"])
    long = records(runs["long"])

    assert any(short.values()), "проверка вырождена: записей о работе нет вовсе"

    for client_id, items in short.items():
        assert items == [item for item in long[client_id] if item[1] < boundary], client_id


def test_profile_keeps_every_earlier_client(runs):

    short = pq.read_table(runs["short"] / "profile.parquet").to_pylist()
    long = pq.read_table(runs["long"] / "profile.parquet").to_pylist()

    by_id = {row["client_id"]: row for row in long}

    for row in short:
        other = by_id.get(row["client_id"])
        assert other is not None, f"{row['client_id']}: клиента нет в длинной выгрузке"
        for name in ("birth_date", "gender", "city"):
            if name in row:
                assert row[name] == other[name], f"{row['client_id']}: {name} изменилось"


# ============================================================
# НЕСКОЛЬКО ВЫГРУЗОК В ОДНОМ ПРОЦЕССЕ
# ============================================================


def test_repeat_after_another_window_is_identical(runs):
    """
    Короткая выгрузка после длинной обязана совпасть с первой
    короткой: процессные глобалы и кэши между прогонами не
    протекают.
    """

    assert tape(runs["short"]) == tape(runs["again"])


def test_repeat_gives_the_same_profile(runs):
    """
    Анкета с вехами и записями о работе повторяется целиком: они не
    зависят от того, что считалось в процессе раньше.
    """

    short = pq.read_table(runs["short"] / "profile.parquet").to_pylist()
    again = pq.read_table(runs["again"] / "profile.parquet").to_pylist()

    assert short == again


def test_repeat_gives_the_same_manifest(runs):

    def card(directory) -> dict:
        record = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        record.pop("created_at", None)
        return record

    assert card(runs["short"]) == card(runs["again"])


# ============================================================
# ГОРИЗОНТ ПЛАНИРОВАНИЯ
# ============================================================


def test_window_beyond_planning_horizon_is_refused():
    """
    Планы строятся до config.PLANNING_END. Окно за ним получило бы
    обрезанную жизнь клиента, поэтому оно отвергается, а не
    выгружается молча.
    """

    start, end = config.HISTORY_START, config.HISTORY_END

    try:
        with pytest.raises(ValueError, match="горизонт"):
            config.activate_horizon(START, config.PLANNING_END + timedelta(days=1))
    finally:
        config.activate_horizon(start, end)
