from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .calendar import calendar_features
from .canonical.build import EVENTS_FILE
from .canonical.schema import LIFELONG_SOURCE_COLUMN
from .keys import (
    DIRECT_KEYS,
    DYNAMIC_FIELDS,
    PROFILE_KEYS,
    key_for,
    profile_change_keys,
)
from .profile_state import INCLUDED_FIELDS, profile_at, typed_profile_value
from .projection import EVENT_TYPE_FIELD, SEMANTIC_PAYLOAD_FIELDS, model_event
from .settings import PreprocessingConfig, group_dir, raw_group_dir


# ============================================================
# ЧТЕНИЕ ГРУППЫ
# ============================================================
#
# Единственный способ добраться до данных для модели: две
# таблицы и ничего больше.
#
#   data/02_preprocessed/<group>/events.parquet   очищенная лента
#   data/01_raw/<group>/profile.parquet           анкета как есть
#
# Анкета берётся прямо из выгрузки: препроцессингу в ней
# нечего чинить, а копия с тем же содержимым была бы вторым
# источником правды о клиенте.
#
# История клиента это его строки, отобранные по client_id и
# упорядоченные так, как их уложил препроцессинг. Срез — простой
# отбор event_time < cutoff, а не отдельный этап.
#
# Срез ОДИН — cutoff T: события строго раньше T, анкета —
# Attributes на тот же T и Lifelong строго раньше T
# (profile_state.PROFILE_SEMANTICS). Данных позже T в историю не
# попадает ничего. Лента после T читается только откатом
# анкеты: он снимает изменения, которых на T ещё не было.
#
# Модель получает ФАКТИЧЕСКИЕ поля под смысловыми ключами:
# производных признаков здесь нет ни одного. Ни интервалов, ни
# отношений к доходу, ни активности по месяцам, ни цепочек — их
# не считает никто.
# ============================================================


class ReadError(ValueError):
    """
    Группа не читается: нет файлов или запрошен неизвестный
    клиент.
    """


@dataclass
class ClientEvent:
    """
    Одно событие клиента под смысловыми ключами.
    """

    client_id: str
    event_time: datetime
    source: str
    values: dict[str, object] = field(default_factory=dict)

    # Час, день недели и день месяца на окружности: отдельный
    # числовой вход модели, считанный из event_time и только из него.
    # Полем события не является и в словари не входит.
    calendar: tuple[float, ...] = ()

    # Тип вехи анкеты, чей источник записан этим событием, иначе
    # None. Значением события не является: модели не отдаётся, а
    # только выводит событие из целей MLM.
    lifelong_source: str | None = None

    def model_values(self) -> dict[str, object]:
        """
        Всё, что событие отдаёт модели. Это ровно его значения:
        добавлять к ним нечего.
        """

        return dict(self.values)

    def as_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "event_time": self.event_time,
            "source": self.source,
            "values": dict(self.values),
            "calendar": list(self.calendar),
            "lifelong_source": self.lifelong_source,
        }


@dataclass
class ClientHistory:
    """
    Клиент на срез: его события и его итоговый профиль.
    """

    client_id: str
    cutoff: datetime
    events: list[ClientEvent]

    # Attributes @ cutoff: значения полей анкеты под смысловыми
    # ключами.
    profile: dict[str, object]

    # Есть ли у клиента анкета вообще. Пустой словарь значений
    # ответом не является: у известной анкеты все поля могут
    # оказаться незаполненными.
    has_profile: bool = False

    limitations: list[str] = field(default_factory=list)

    # Lifelong: вехи (тип, время) строго раньше cutoff, по времени.
    # Отдельно от profile — у вехи есть время, у поля нет.
    lifelong: list[tuple[str, datetime]] = field(default_factory=list)

    @property
    def n_events(self) -> int:
        return len(self.events)

    def summary(self) -> dict:
        return {
            "client_id": self.client_id,
            "cutoff": self.cutoff,
            "events": self.n_events,
            "keys_used": len({key for item in self.events for key in item.values}),
            "limitations": self.limitations,
        }


class Group:
    """
    Группа для модели: очищенная лента и анкета из выгрузки.

    Клиенты берутся из анкеты: одна строка на клиента, и
    отдельного реестра для этого не нужно.
    """

    PROFILE_FILE = "profile.parquet"

    def __init__(self, group: str, directory: Path | None = None, profile_path: Path | None = None):

        # Явные пути — для процесса, который не видит подменённых
        # глобалов каталогов (at_cutoff в процессах подготовки).
        self.group = group
        self.directory = Path(directory) if directory is not None else group_dir(group)
        self.profile_path = (
            Path(profile_path) if profile_path is not None
            else raw_group_dir(group) / Group.PROFILE_FILE
        )

        # Пояс банка нужен ровно для календаря: event_time
        # остаётся в UTC и в UTC же сравнивается с cutoff.
        self.timezone = PreprocessingConfig.load(None).bank_timezone()

        events_path = self.directory / EVENTS_FILE

        if not events_path.exists():
            raise ReadError(f"нет {events_path}: выполните preprocess {group}")

        if not self.profile_path.exists():
            raise ReadError(
                f"нет {self.profile_path}: анкета читается прямо из выгрузки, "
                f"и без неё группа неполна"
            )

        self._events = pq.ParquetFile(events_path)

        # Без пометки источников вех события-источники молча стали
        # бы целями MLM: такой слой собран прежним кодом.
        if LIFELONG_SOURCE_COLUMN not in self._events.schema_arrow.names:
            raise ReadError(
                f"{events_path}: нет колонки {LIFELONG_SOURCE_COLUMN} — слой собран "
                f"прежним кодом; выполните preprocess {group} заново"
            )

        self._profile_rows: dict[str, dict] = {
            row["client_id"]: row for row in pq.read_table(self.profile_path).to_pylist()
        }

        self.client_ids: list[str] = sorted(self._profile_rows)

        self._index: dict[str, tuple[int, int]] | None = None

        # Можно ли обойти ленту потоком (_stream): внутри каждой группы
        # строк клиенты по возрастанию id, и строки клиента подряд.
        # Узнаётся вместе с индексом.
        self._sorted_runs: bool = False

        self._edges: list[int] = [0]

        for number in range(self._events.num_row_groups):
            self._edges.append(self._edges[-1] + self._events.metadata.row_group(number).num_rows)

    # --- клиенты ---

    def _addresses(self) -> dict[str, tuple[int, int]]:
        """
        Первая строка клиента и число его строк: строки одного
        клиента лежат подряд, поэтому адреса хватает.
        """

        if self._index is not None:
            return self._index

        index: dict[str, tuple[int, int]] = {}

        position = 0

        # Поток возможен, если строки клиента идут подряд (один блок на
        # клиента во всей ленте) и внутри каждой группы строк блоки
        # по возрастанию id. Препроцессинг так и пишет; чужая лента
        # уходит на чтение по адресу.
        runs = 0
        ascending = True
        previous: str | None = None

        for number in range(self._events.num_row_groups):

            column = (
                self._events.read_row_group(number, columns=["client_id"])
                .column("client_id")
                .to_pylist()
            )

            inside: str | None = None

            for offset, client_id in enumerate(column):
                start, count = index.get(client_id, (position + offset, 0))
                index[client_id] = (start, count + 1)
                if client_id != previous:
                    runs += 1
                    previous = client_id
                if inside is not None and (client_id is None or client_id < inside):
                    ascending = False
                inside = client_id if client_id is not None else inside

            position += len(column)

        self._index = index
        self._sorted_runs = ascending and runs == len(index) and None not in index

        return index

    def events_table(self, client_id: str) -> pa.Table:
        """
        Строки клиента так, как они лежат в ленте.
        """

        address = self._addresses().get(client_id)

        if address is None:
            return self._events.schema_arrow.empty_table()

        start, count = address

        first = max(number for number, edge in enumerate(self._edges[:-1]) if edge <= start)

        last = first

        while last + 1 < len(self._edges) - 1 and self._edges[last + 1] < start + count:
            last += 1

        table = self._events.read_row_groups(list(range(first, last + 1)))

        return table.slice(start - self._edges[first], count)

    # --- история ---

    def history(
        self,
        client_id: str,
        cutoff: datetime,
    ) -> ClientHistory:
        """
        Клиент на cutoff: события строго раньше cutoff и анкета —
        состояние клиента на тот же cutoff.
        """

        if client_id not in self._profile_rows and client_id not in self._addresses():
            raise ReadError(f"клиента {client_id!r} нет в группе {self.group}")

        return self._build(client_id, cutoff, self.events_table(client_id))

    def _build(self, client_id: str, cutoff: datetime, table: pa.Table) -> ClientHistory:
        """
        История клиента из его строк ленты (все строки, как в ленте).
        """

        column = table.column("event_time")
        times = column.to_pylist()

        # События — строки раньше cutoff; строки с cutoff и позже
        # нужны только откату анкеты: он снимает изменения, которых
        # на cutoff ещё не было. В события из них не попадает ничего.
        #
        # Обычно вся лента клиента раньше cutoff (финальный cutoff это
        # конец выгрузки): тогда раньше cutoff и её максимум. Пустое
        # время сюда не проходит — его сравнение с cutoff остаётся
        # ошибкой, как и было.
        if not times or (column.null_count == 0 and pc.max(column).as_py() < cutoff):
            head, moments, later = table, times, []
        else:
            before = [index for index, moment in enumerate(times) if moment < cutoff]
            later = [index for index, moment in enumerate(times) if moment >= cutoff]
            head = table.take(pa.array(before, pa.int64()))
            moments = [times[index] for index in before]

        events, notes = client_events(head, moments, self.timezone)

        snapshot = self._profile_rows.get(client_id)

        # Снимок описывает клиента перед as_of, и вперёд его не
        # восстановить: состояние позже снимка из данных не следует.
        if snapshot is not None and cutoff > snapshot["as_of"]:
            raise ReadError(
                f"{client_id}: cutoff {cutoff.isoformat()} позже снимка анкеты "
                f"as_of {snapshot['as_of'].isoformat()}"
            )

        # Откат смотрит только на строки с cutoff и позже, в порядке ленты.
        rest = table.take(pa.array(later, pa.int64())).to_pylist() if later else []

        state = profile_at(snapshot, rest, cutoff, self.timezone)

        notes.extend(state.notes)

        # Помеченное событие раньше cutoff, и его веха того же
        # момента обязана быть в Lifelong того же cutoff.
        for item in events:
            if item.lifelong_source is not None and (
                (item.lifelong_source, item.event_time) not in state.lifelong
            ):
                raise ReadError(
                    f"{client_id}: событие {item.event_time.isoformat()} помечено источником "
                    f"вехи {item.lifelong_source}, а такой вехи в анкете на cutoff нет"
                )

        return ClientHistory(
            client_id=client_id,
            cutoff=cutoff,
            events=events,
            profile=profile_values(state.values),
            has_profile=snapshot is not None,
            limitations=sorted(set(notes)),
            lifelong=list(state.lifelong),
        )

    def histories(
        self,
        cutoff: datetime,
        clients: list[str] | None = None,
    ):
        """
        Клиенты по одному, в устойчивом порядке.

        Обход всей группы (clients не задан) читает каждую группу строк
        ленты ОДИН раз: history по одному клиенту читает всю его группу
        строк, и при 64 клиентах на группу лента читалась бы 64 раза.
        Порядок клиентов и сами истории те же, что у history.
        """

        if clients is None:
            self._addresses()
            if self._sorted_runs:
                yield from self._stream(cutoff)
                return

        for client_id in (clients if clients is not None else self.client_ids):
            yield self.history(client_id, cutoff)

    def _stream(self, cutoff: datetime):
        """
        Истории клиентов анкеты в порядке self.client_ids, а строки —
        курсором по каждой группе строк: внутри группы клиенты по
        возрастанию id, поэтому группы сливаются, как отсортированные
        отрезки. Клиент, разрезанный границей куска или группы строк,
        собирается целиком; клиент ленты без анкеты пропускается, как
        и в history-обходе.
        """

        path = self.directory / EVENTS_FILE

        cursors = [_ClientCursor(path, number) for number in range(self._events.num_row_groups)]

        empty = self._events.schema_arrow.empty_table()

        for client_id in self.client_ids:

            pieces: list[pa.Table] = []

            for cursor in cursors:
                cursor.skip_before(client_id)
                pieces.extend(cursor.take(client_id))

            table = pa.concat_tables(pieces) if pieces else empty

            yield self._build(client_id, cutoff, table)


class _ClientCursor:
    """
    Курсор по одной группе строк ленты: блоки клиентов по порядку,
    кусками фиксированного размера. Вся группа в памяти не лежит.
    """

    BATCH_ROWS = 2048

    def __init__(self, path: Path, row_group: int):
        self._batches = pq.ParquetFile(path).iter_batches(batch_size=self.BATCH_ROWS, row_groups=[row_group])
        self._rest: pa.Table | None = None
        self._next()

    def _next(self) -> None:
        """Следующий непустой кусок в _rest или None, если группа кончилась."""
        for batch in self._batches:
            if batch.num_rows:
                self._rest = pa.Table.from_batches([batch])
                return
        self._rest = None

    @property
    def head(self) -> str | None:
        return None if self._rest is None else self._rest.column("client_id")[0].as_py()

    def _block(self) -> pa.Table:
        """Строки клиента head в начале _rest; _rest сдвигается за них."""

        head = self.head
        column = self._rest.column("client_id")
        others = pc.indices_nonzero(pc.not_equal(column, head))

        if len(others):
            end = others[0].as_py()
            block, self._rest = self._rest.slice(0, end), self._rest.slice(end)
            return block

        block = self._rest
        self._next()
        return block

    def take(self, client_id: str) -> list[pa.Table]:
        """Все строки клиента, если он сейчас в начале курсора."""

        pieces: list[pa.Table] = []

        while self.head == client_id:
            pieces.append(self._block())

        return pieces

    def skip_before(self, client_id: str) -> None:
        """Пропустить клиентов ленты с id меньше client_id (их нет в анкете)."""

        while self.head is not None and self.head < client_id:
            self._block()


# ============================================================
# ЗНАЧЕНИЯ ПОД СМЫСЛОВЫМИ КЛЮЧАМИ
# ============================================================


# Смысловой ключ поля по (имя, источник): key_for — чистая функция
# реестра ключей, а спрашивают её на каждое поле каждого события.
# Запоминается только найденный ключ: ошибка повторяется каждый раз.
_KEYS: dict[tuple[str, str], str] = {}


def _key(name: str, source: str) -> str:

    key = _KEYS.get((name, source))

    if key is None:
        key = _KEYS[(name, source)] = key_for(name, source).key

    return key


def client_events(table: pa.Table, times: list[datetime], timezone) -> tuple[list[ClientEvent], list[str]]:
    """
    События из строк ленты (times — их event_time) и заметки о
    неразобранных значениях профиля.

    Значения собираются по колонкам: из 62 колонок ленты у строки
    заполнены единицы, и словарь на каждую строку целиком не
    строится. Ответ тот же, что у построчного разбора
    (client_events_by_rows): ключи событий в том же порядке —
    тип, затем поля в порядке проекции, затем изменение профиля.
    Ошибку данных воспроизводит построчный разбор: какое поле и
    какая строка названы в ней первыми, решает он.
    """

    if not times:
        return [], []

    try:
        values, notes = _values_by_columns(table)
    except (ValueError, KeyError):
        return client_events_by_rows(table.to_pylist(), timezone)

    # Календарь считается по местному времени банка: перевод
    # пояса живёт внутри calendar_features и наружу не
    # выходит. Само event_time ниже кладётся как есть, в UTC.
    calendars = calendar_features(times, timezone).tolist()

    client_ids = table.column("client_id").to_pylist()
    sources = table.column("source").to_pylist()
    marks = table.column(LIFELONG_SOURCE_COLUMN).to_pylist()

    events = [
        ClientEvent(
            client_id=client_ids[index],
            event_time=times[index],
            source=sources[index],
            values=values[index],
            calendar=tuple(calendars[index]),
            lifelong_source=marks[index],
        )
        for index in range(len(times))
    ]

    return events, notes


def _values_by_columns(table: pa.Table) -> tuple[list[dict[str, object]], list[str]]:
    """
    event_values каждой строки, собранные по колонкам.
    """

    count = table.num_rows
    names = set(table.column_names)
    sources = table.column("source").to_pylist()

    # Тип у model_event идёт первым, даже пустой.
    type_key = DIRECT_KEYS[EVENT_TYPE_FIELD].key
    values: list[dict[str, object]] = [{type_key: kind} for kind in table.column(EVENT_TYPE_FIELD).to_pylist()]

    def present(name: str) -> tuple[list[int], list]:
        """Строки с заполненным полем и его значения в них."""

        if name not in names:
            return [], []

        column = table.column(name)

        if column.null_count == count:
            return [], []

        if column.null_count == 0:
            return list(range(count)), column.to_pylist()

        rows = np.flatnonzero(pc.is_valid(column).to_numpy(zero_copy_only=False)).tolist()

        return rows, pc.drop_null(column).to_pylist()

    for name in SEMANTIC_PAYLOAD_FIELDS:

        if name == EVENT_TYPE_FIELD or name in DYNAMIC_FIELDS:
            continue

        rows, items = present(name)

        if not rows:
            continue

        keys = {source: _key(name, source) for source in {sources[index] for index in rows}}

        for index, value in zip(rows, items):
            values[index][keys[sources[index]]] = value

    # Изменение профиля дописывается после полей строки, как
    # values.update(changed) в event_values.
    notes: list[str] = []

    rows, names_changed = present("field_name")

    if rows:

        old = table.column("old_value").to_pylist() if "old_value" in names else None
        new = table.column("new_value").to_pylist() if "new_value" in names else None

        for index, field_name in zip(rows, names_changed):

            changed, problems = _profile_change_values({
                "field_name": field_name,
                "old_value": None if old is None else old[index],
                "new_value": None if new is None else new[index],
            })

            values[index].update(changed)
            notes.extend(problems)

    return values, notes


def client_events_by_rows(rows: list[dict], timezone) -> tuple[list[ClientEvent], list[str]]:
    """
    Тот же разбор построчно: эталон для client_events и путь, на
    котором ошибка данных называет свою строку и своё поле.
    """

    calendars = (
        calendar_features([row["event_time"] for row in rows], timezone).tolist()
        if rows
        else []
    )

    events: list[ClientEvent] = []
    notes: list[str] = []

    for index, row in enumerate(rows):

        values, problems = event_values(row, row["source"])

        notes.extend(problems)

        events.append(
            ClientEvent(
                client_id=row["client_id"],
                event_time=row["event_time"],
                source=row["source"],
                values=values,
                calendar=tuple(calendars[index]),
                lifelong_source=row[LIFELONG_SOURCE_COLUMN],
            )
        )

    return events, notes


def event_values(row: dict, source: str) -> tuple[dict[str, object], list[str]]:
    """
    Значения одного события под смысловыми ключами.

    Состав берёт модельная проекция: наружу проходит ровно то,
    что ей разрешено.
    """

    fields = model_event(row).fields

    values: dict[str, object] = {}

    for name, value in fields.items():

        if name == EVENT_TYPE_FIELD:
            values[DIRECT_KEYS[EVENT_TYPE_FIELD].key] = value
            continue

        if name in DYNAMIC_FIELDS:
            # Смысл задаёт field_name, разбор ниже.
            continue

        values[_key(name, source)] = value

    changed, notes = _profile_change_values(fields)

    values.update(changed)

    return values, notes


def _profile_change_values(fields: dict) -> tuple[dict[str, object], list[str]]:
    """
    Прежнее и новое значение изменившегося поля профиля в его
    собственном смысле.

    Доход остаётся числом, город категорией, число детей числом.
    Один текстовый ключ на всё это был бы неправдой: BPE резал бы
    доход так же, как название города.
    """

    name = fields.get("field_name")

    if name is None:
        return {}, []

    old_key, new_key = profile_change_keys(name)

    out: dict[str, object] = {}
    notes: list[str] = []

    for key, raw in ((old_key, fields.get("old_value")), (new_key, fields.get("new_value"))):

        if raw is None:
            continue

        value, note = typed_profile_value(key, raw, name)

        if note is not None:
            notes.append(note)
            continue

        out[key.key] = value

    return out, notes


def profile_values(profile: dict | None) -> dict[str, object]:
    """
    Профиль клиента под смысловыми ключами.

    Состав полей задаёт INCLUDED_FIELDS и только он: поле, чьё
    значение на нужный момент из данных не следует, в модельную
    анкету не идёт ни у кого. Список одинаков для всех клиентов,
    поэтому отсутствие поля ни о ком ничего не сообщает.
    """

    if profile is None:
        return {}

    return {
        PROFILE_KEYS[name].key: profile[name]
        for name in INCLUDED_FIELDS
        if profile.get(name) is not None
    }


__all__ = [
    "ClientEvent",
    "ClientHistory",
    "Group",
    "ReadError",
    "client_events",
    "client_events_by_rows",
    "event_values",
    "profile_values",
]
