from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .calendar import calendar_features
from .canonical.build import EVENTS_FILE
from .keys import (
    CATEGORICAL,
    COUNT,
    DIRECT_KEYS,
    DYNAMIC_FIELDS,
    NUMERIC,
    PROFILE_KEYS,
    SemanticKey,
    key_for,
    profile_change_keys,
)
from .projection import EVENT_TYPE_FIELD, model_event
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
        }


@dataclass
class ClientHistory:
    """
    Клиент на срез: его события и его итоговый профиль.
    """

    client_id: str
    cutoff: datetime | None
    events: list[ClientEvent]
    profile: dict[str, object]

    # Есть ли у клиента анкета вообще. Пустой словарь значений
    # ответом не является: у известной анкеты все поля могут
    # оказаться незаполненными.
    has_profile: bool = False

    limitations: list[str] = field(default_factory=list)

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

    def __init__(self, group: str):

        self.group = group
        self.directory = group_dir(group)
        self.profile_path = raw_group_dir(group) / Group.PROFILE_FILE

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

        self._profile_rows: dict[str, dict] = {
            row["client_id"]: row for row in pq.read_table(self.profile_path).to_pylist()
        }

        self.client_ids: list[str] = sorted(self._profile_rows)

        self._index: dict[str, tuple[int, int]] | None = None

        self._edges: list[int] = [0]

        for number in range(self._events.num_row_groups):
            self._edges.append(self._edges[-1] + self._events.metadata.row_group(number).num_rows)

    # --- клиенты ---

    def profile_row(self, client_id: str) -> dict | None:
        return self._profile_rows.get(client_id)

    def _addresses(self) -> dict[str, tuple[int, int]]:
        """
        Первая строка клиента и число его строк: строки одного
        клиента лежат подряд, поэтому адреса хватает.
        """

        if self._index is not None:
            return self._index

        index: dict[str, tuple[int, int]] = {}

        position = 0

        for number in range(self._events.num_row_groups):

            column = (
                self._events.read_row_group(number, columns=["client_id"])
                .column("client_id")
                .to_pylist()
            )

            for offset, client_id in enumerate(column):
                start, count = index.get(client_id, (position + offset, 0))
                index[client_id] = (start, count + 1)

            position += len(column)

        self._index = index

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

    def history(self, client_id: str, cutoff: datetime | None = None) -> ClientHistory:
        """
        Клиент на срез: события строго раньше cutoff.

        Без cutoff отдаётся вся лента клиента. Ничего, кроме
        отбора по времени, срез не делает.
        """

        if client_id not in self._profile_rows and client_id not in self._addresses():
            raise ReadError(f"клиента {client_id!r} нет в группе {self.group}")

        table = self.events_table(client_id)

        if cutoff is not None and table.num_rows:
            # Срез и время события живут в одной шкале — UTC.
            moment = pa.scalar(cutoff, type=pa.timestamp("us", tz="UTC"))
            table = table.filter(pc.less(table.column("event_time"), moment))

        rows = table.to_pylist()

        # Календарь считается по местному времени банка: перевод
        # пояса живёт внутри calendar_features и наружу не
        # выходит. Само event_time ниже кладётся как есть, в UTC.
        calendars = (
            calendar_features([row["event_time"] for row in rows], self.timezone)
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
                    calendar=tuple(float(value) for value in calendars[index]),
                )
            )

        profile = self._profile_rows.get(client_id)

        return ClientHistory(
            client_id=client_id,
            cutoff=cutoff,
            events=events,
            profile=profile_values(profile),
            has_profile=profile is not None,
            limitations=sorted(set(notes)),
        )

    def histories(self, cutoff: datetime | None = None, clients: list[str] | None = None):
        """
        Клиенты по одному, в устойчивом порядке.
        """

        for client_id in (clients if clients is not None else self.client_ids):
            yield self.history(client_id, cutoff)


# ============================================================
# ЗНАЧЕНИЯ ПОД СМЫСЛОВЫМИ КЛЮЧАМИ
# ============================================================


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

        values[key_for(name, source).key] = value

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

        value, note = _typed_profile_value(key, raw, name)

        if note is not None:
            notes.append(note)
            continue

        out[key.key] = value

    return out, notes


def _typed_profile_value(key: SemanticKey, raw: object, field_name: str) -> tuple[object, str | None]:
    """
    Значение профиля в виде своего ключа. Неразобранное число не
    подменяется текстом и признаком не становится.

    Счётчик объявлен категорией, но числом быть не перестал.
    Изменение профиля приходит строкой, и без привода «2» у
    прежнего значения и 2 у самого поля стали бы разными записями
    одного факта: словарь хранит запись значения вместе с его
    типом. Дробный счётчик это ошибка, а не повод молча стать
    числом с точкой.
    """

    counter = key.kind == CATEGORICAL and key.unit == COUNT

    if key.kind != NUMERIC and not counter:
        return raw, None

    text = str(raw)

    try:
        return int(text), None
    except ValueError:
        pass

    if counter:
        return None, f"значение профиля {field_name} не разобрано как целый счётчик: {text!r}"

    try:
        return float(text), None
    except ValueError:
        return None, f"значение профиля {field_name} не разобрано как число: {text!r}"


def profile_values(profile: dict | None) -> dict[str, object]:
    """
    Итоговый профиль клиента под смысловыми ключами.
    """

    if profile is None:
        return {}

    return {
        key.key: profile[name]
        for name, key in PROFILE_KEYS.items()
        if profile.get(name) is not None
    }


__all__ = [
    "ClientEvent",
    "ClientHistory",
    "Group",
    "ReadError",
    "event_values",
    "profile_values",
]
