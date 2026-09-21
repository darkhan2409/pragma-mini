from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .artifacts import read_json
from .canonical.build import (
    CLIENT_INDEX_FILE,
    COVERAGE_FILE,
    EVENTS_FILE,
    PROFILE_FILE,
)
from .canonical.entities import ENTITY_FIELDS, TRANSFER_SIDES, TRANSITIONS


# ============================================================
# ИДЕЯ
# ============================================================
#
# Один вопрос и один ответ: что банк знал о клиенте ДО момента
# cutoff.
#
# У записи одно время — время события. Времени поступления в
# хранилище у выгрузки нет, поэтому известность записи и её
# событие совпадают: произошло до cutoff — известно, иначе нет.
# Cutoff — исключительная граница: запись ровно на ней ещё не
# известна.
#
# Версий у события может быть несколько; действует наибольшая.
# Техническая повторная доставка не создаёт второго действия.
# Исправление уточняет старое событие НА ЕГО МЕСТЕ, потому что
# порядок задан временем бизнеса и местом первой версии.
#
# Знание из будущего сюда не просачивается двумя путями, и оба
# закрыты явно: недатированные статусы покрытия в состояние на
# дату не входят; встречная сторона перевода видна, только если
# её событие уже произошло.
#
# Это функция чтения, а не датасет: срезов она не назначает и
# ничего не материализует.
# ============================================================


STAGE = "history"
STAGE_VERSION = "5.0.0"

# Колонки canonical, которые НЕ выдаются как знание клиента.
#
# В видимом наборе повторов идентификатора нет, поэтому флаг
# там всегда пуст и только сбивал бы с толку.
INTERNAL_COLUMNS: tuple[str, ...] = ("is_repeated_event_id",)

# Состояние сущности после перехода. Переходы обслуживания
# (платежи, просрочка, смена условий) состояния не меняют и
# записываются как последний переход.
ENTITY_STATE: dict[tuple[str, str], str] = {
    ("account", "opened"): "open",
    ("card", "activated"): "active",
    ("card", "blocked"): "blocked",
    ("card", "unblocked"): "active",
    ("card", "reissued"): "active",
    ("contract", "opened"): "open",
    ("contract", "closed"): "closed",
    ("contract", "migrated"): "migrated",
    ("application", "submitted"): "submitted",
    ("application", "decided"): "decided",
    ("case", "opened"): "open",
    ("case", "updated"): "open",
    ("case", "resolved"): "resolved",
}

# Состояния покрытия, выводимые ТОЛЬКО из датированных полей.
COVERAGE_NOT_LAUNCHED = "source_not_launched"
COVERAGE_NOT_SEEN = "client_not_seen_yet"
COVERAGE_ENDED = "ended"
COVERAGE_AVAILABLE = "available"

# Недатированные причины: они описывают выгрузку целиком и не
# могут быть отнесены к конкретной дате.
#
# source_outage сюда больше не входит: покрытие называет дни
# сбоя поимённо, и причина стала датированной.
UNDATED_REASONS: frozenset[str] = frozenset(
    {"no_consent", "client_not_onboarded"}
)

PAIR_VISIBLE = "counterpart_visible"
PAIR_NOT_VISIBLE = "counterpart_not_visible"
# Обе стороны у одного человека: перевод между своими счетами.
# Встречная сторона видна целиком, и звать её «невидимой» было
# просто неверно — она лежит в той же ленте.
PAIR_OWN = "own_account_both_sides"


class HistoryError(ValueError):
    """
    Вопрос задан некорректно: нет клиента, cutoff за границей
    выгрузки.
    """


# ============================================================
# СОСТОЯНИЯ
# ============================================================


@dataclass(frozen=True)
class SourceState:
    source: str
    state: str
    first_available_at: datetime | None
    first_seen: datetime | None
    last_available_at: datetime | None
    # Недатированная причина выгрузки: она описывает весь период
    # наблюдения, а не момент cutoff, и в state не входит. Но
    # знать её нужно: «клиент не подключён» и «нет согласия»
    # означают, что источник к этому клиенту НЕПРИМЕНИМ, а не
    # что он молчал.
    reason: str | None = None
    # Дни, в которые источник не донёс строки до витрины. Причина
    # source_outage без них была бессодержательной: «сбой был»
    # без ответа на вопрос когда.
    outage_days: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "state": self.state,
            "first_available_at": self.first_available_at,
            "first_seen": self.first_seen,
            "last_available_at": self.last_available_at,
            "reason": self.reason,
            "outage_days": list(self.outage_days),
        }


@dataclass(frozen=True)
class EntityState:
    kind: str
    entity_id: str
    state: str | None
    since: datetime | None
    opening_observed: bool
    first_mention: datetime | None
    last_transition: str | None
    last_transition_at: datetime | None

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "entity_id": self.entity_id,
            "state": self.state,
            "since": self.since,
            "opening_observed": self.opening_observed,
            "first_mention": self.first_mention,
            "last_transition": self.last_transition,
            "last_transition_at": self.last_transition_at,
        }


@dataclass(frozen=True)
class TransferSide:
    transfer_id: str
    side: str | None
    event_id: str
    event_time: datetime
    amount: int | None
    direction: str | None
    pair_state: str
    counterpart_client_id: str | None

    def as_dict(self) -> dict:
        return {
            "transfer_id": self.transfer_id,
            "side": self.side,
            "event_id": self.event_id,
            "event_time": self.event_time,
            "amount": self.amount,
            "direction": self.direction,
            "pair_state": self.pair_state,
            "counterpart_client_id": self.counterpart_client_id,
        }


@dataclass(frozen=True)
class Relationship:
    observed_start: datetime | None
    observed_days: int | None
    history_incomplete: bool
    incomplete_reasons: tuple[str, ...]
    closed_at: datetime | None
    closed_reason: str | None

    def as_dict(self) -> dict:
        return {
            "observed_start": self.observed_start,
            "observed_days": self.observed_days,
            "history_incomplete": self.history_incomplete,
            "incomplete_reasons": list(self.incomplete_reasons),
            "closed_at": self.closed_at,
            "closed_reason": self.closed_reason,
        }


@dataclass
class ClientHistory:
    client_id: str
    client_idx: int
    cutoff: datetime
    events: pa.Table
    counts: dict[str, int]
    profile: dict | None
    profile_meta: dict
    coverage: list[SourceState]
    entities: list[EntityState]
    transfers: list[TransferSide]
    relationship: Relationship
    products: dict[str, dict]
    limitations: list[str]

    @property
    def n_events(self) -> int:
        return self.events.num_rows

    def summary(self) -> dict:
        return {
            "client_id": self.client_id,
            "client_idx": self.client_idx,
            "cutoff": self.cutoff,
            "counts": self.counts,
            "profile": self.profile_meta,
            "coverage": [item.as_dict() for item in self.coverage],
            "entities": [item.as_dict() for item in self.entities],
            "transfers": [item.as_dict() for item in self.transfers],
            "relationship": self.relationship.as_dict(),
            "products": self.products,
            "limitations": self.limitations,
        }


# ============================================================
# ХРАНИЛИЩЕ CANONICAL
# ============================================================


class CanonicalStore:
    """
    Чтение одной группы canonical по адресу клиента.

    Строки клиента лежат подряд внутри одного row group, поэтому
    история читается точечно, а не полным сканом файла.
    """

    REPORT_FILE = "canonical_report.json"

    def __init__(self, directory: Path, products: pa.Table | None = None):

        self.directory = Path(directory)

        self.report = read_json(self.directory / self.REPORT_FILE)

        self.extract_time = datetime.fromisoformat(self.report["raw"]["extract_time"])
        self.history_start = datetime.fromisoformat(self.report["raw"]["history_start"])

        self._events = pq.ParquetFile(self.directory / EVENTS_FILE)

        self.clients = pq.read_table(self.directory / CLIENT_INDEX_FILE).to_pylist()
        self.by_id = {row["client_id"]: row for row in self.clients}
        self.by_idx = {row["client_idx"]: row for row in self.clients}

        self._profile = pq.read_table(self.directory / PROFILE_FILE)
        self._coverage = pq.read_table(self.directory / COVERAGE_FILE)

        self._profile_rows: dict[str, list[dict]] = {}
        for row in self._profile.to_pylist():
            self._profile_rows.setdefault(row["client_id"], []).append(row)

        self._coverage_rows: dict[str, list[dict]] = {}
        for row in self._coverage.to_pylist():
            self._coverage_rows.setdefault(row["client_id"], []).append(row)

        self.products = products

        self._transfers: tuple[dict, dict] | None = None
        self._group_cache: tuple[int, pa.Table] | None = None

        # Первая глобальная строка каждой row group: по ней
        # клиент, лежащий на границе, читается точечно.
        self._group_offsets: list[int] = []

        position = 0

        for index in range(self._events.num_row_groups):
            self._group_offsets.append(position)
            position += self._events.metadata.row_group(index).num_rows

    # --- лента ---

    def client_row(self, client: str | int) -> dict:

        row = self.by_idx.get(client) if isinstance(client, int) else self.by_id.get(client)

        if row is None:
            raise HistoryError(f"клиента {client!r} нет в canonical этой группы")

        return row

    def client_events(self, client: str | int) -> pa.Table:

        row = self.client_row(client)

        if not row["row_count"]:
            return self._events.schema_arrow.empty_table()

        if row["spans_row_groups"]:

            # Клиент лёг на границу групп: читаются только те
            # группы, в которых он есть, а не весь файл. Полное
            # чтение стоило бы всей ленты группы ради одного
            # клиента.
            first = row["row_group"]
            last = first

            offset = self._group_offsets[first]

            while (
                last + 1 < len(self._group_offsets)
                and self._group_offsets[last + 1] < row["global_row_start"] + row["row_count"]
            ):
                last += 1

            table = self._events.read_row_groups(list(range(first, last + 1)))

            return table.slice(row["global_row_start"] - offset, row["row_count"])

        group = row["row_group"]

        if self._group_cache is None or self._group_cache[0] != group:
            self._group_cache = (group, self._events.read_row_group(group))

        return self._group_cache[1].slice(row["row_offset"], row["row_count"])

    # --- спутники ---

    def profile_rows(self, client_id: str) -> list[dict]:
        return self._profile_rows.get(client_id, [])

    def coverage_rows(self, client_id: str) -> list[dict]:
        return self._coverage_rows.get(client_id, [])

    # --- переводы ---

    @property
    def transfer_index(self) -> tuple[dict, dict]:
        """
        Стороны переводов всей группы, разложенные дважды: по
        клиенту и по переводу.

        Индекс строится один раз по ленте по исходным строкам
        переводов: пара собирается только среди сторон, чьё
        событие уже произошло к cutoff.

        Два словаря вместо одного списка потому, что история
        одного клиента обязана стоить его переводов, а не всех
        переводов группы: линейный проход по общему списку на
        каждого клиента давал квадратичную стоимость этапа.

        Сторона узнаётся по ТИПУ СОБЫТИЯ и непустому transfer_id,
        тем же правилом, что и таблица переводов canonical
        (TRANSFER_SIDES). Метка связи описывает связь, а не вид
        операции: комиссия, привязанная к переводу той же меткой,
        стороной не является, а исправленная строка перевода
        остаётся переводом и из индекса не выпадает. Версии и место
        в ленте берутся здесь же, чтобы действующую версию выбирало
        то же правило, что и в истории событий.
        """

        if self._transfers is None:

            columns = [
                "transfer_id",
                "client_id",
                "event_id",
                "event_type",
                "event_time",
                "is_repeated_event_id",
                "raw_row",
                "amount",
                "direction",
                # Счёт нужен, чтобы отличить встречную сторону от
                # другой записи с тем же ключом: у настоящей пары
                # счета РАЗНЫЕ.
                "account_id",
            ]

            sides = pa.array(sorted(TRANSFER_SIDES))

            by_client: dict[str, list[dict]] = {}
            by_transfer: dict[str, list[dict]] = {}

            for index in range(self._events.num_row_groups):

                chunk = self._events.read_row_group(index, columns=columns)
                chunk = chunk.filter(
                    pc.and_(
                        pc.is_in(chunk.column("event_type"), value_set=sides),
                        pc.is_valid(chunk.column("transfer_id")),
                    )
                )

                for row in chunk.to_pylist():
                    by_client.setdefault(row["client_id"], []).append(row)
                    by_transfer.setdefault(row["transfer_id"], []).append(row)

            self._transfers = (by_client, by_transfer)

        return self._transfers


# ============================================================
# ВИДИМОСТЬ
# ============================================================


def _as_datetime64(column) -> np.ndarray:
    return column.to_numpy(zero_copy_only=False).astype("datetime64[us]")


def _client_view(table: pa.Table) -> pa.Table:
    """
    Единственный выход наружу: служебные поля снимаются здесь и
    только здесь.

    Пустой набор проходит тем же путём, что и непустой, иначе
    «наружу не выдаём» держалось бы на удачном ветвлении.
    """

    return table.drop_columns([name for name in INTERNAL_COLUMNS if name in table.column_names])


def visible_events(table: pa.Table, cutoff: datetime) -> tuple[pa.Table, dict[str, int], list[dict]]:
    """
    Строки, известные банку строго до cutoff, по одной на событие.

    Известность записи совпадает с её событием: произошло до
    cutoff — известно. Версий у записи нет, она сразу
    окончательна.

    Возвращает видимый набор в бизнес-порядке и счётчики
    отброшенных строк по причинам.
    """

    counts = {
        "rows": table.num_rows,
        "visible": 0,
        "event_not_happened": 0,
        "repeated": 0,
    }

    if table.num_rows == 0:
        return _client_view(table), counts, []

    moment = np.datetime64(cutoff, "us")

    event_time = _as_datetime64(table.column("event_time"))
    repeated = np.asarray(table.column("is_repeated_event_id").to_pylist(), dtype=bool)

    happened = event_time < moment

    counts["event_not_happened"] = int((~happened).sum())
    counts["repeated"] = int((happened & repeated).sum())

    indices = np.flatnonzero(happened & ~repeated)

    if indices.size == 0:
        return _client_view(table.slice(0, 0)), counts, []

    selected = indices.astype(np.int64)


    # Бизнес-порядок: устойчивый номер строки клиента уже задан
    # временем события и приоритетом его типа.
    stable = np.asarray(table.column("stable_event_index").to_pylist(), dtype=np.int64)
    selected = selected[np.argsort(stable[selected], kind="stable")]

    visible = table.take(pa.array(selected))

    counts["visible"] = visible.num_rows

    return _client_view(visible), counts, []


# ============================================================
# ПРОФИЛЬ
# ============================================================


def profile_of(rows: list[dict]) -> tuple[dict | None, dict]:
    """
    Анкета клиента: одна итоговая строка на границу выгрузки.

    Версий у профиля нет, поэтому выбирать нечего — строка либо
    есть, либо банк клиента ещё не посчитал.

    ВАЖНОЕ ОГРАНИЧЕНИЕ. Эта строка описывает состояние на конец
    выгрузки, а не на cutoff. Она честна ровно на конечном срезе
    группы; на любом более раннем срезе она была бы знанием из
    будущего, и именно поэтому датасет ранние срезы запрещает.
    """

    meta = {
        "rows": len(rows),
        "rule": (
            "одна итоговая строка на клиента на границу выгрузки; "
            "версий и границ действия у профиля нет"
        ),
    }

    if not rows:
        meta["state"] = "absent"
        return None, meta

    if len(rows) > 1:
        raise HistoryError(
            f"у клиента {rows[0].get('client_id')} {len(rows)} строк профиля: "
            "контракт обещает ровно одну"
        )

    meta["state"] = "known"

    return rows[0], meta


# ============================================================
# ПОКРЫТИЕ
# ============================================================


def _outages_before(
    value,
    first_seen: datetime | None,
    last_available: datetime | None,
    cutoff: datetime,
) -> tuple[str, ...]:
    """
    Дни сбоя, о которых на этот момент уже известно.

    Отрезок закрыт с трёх сторон:

      сбой будущего в состояние на дату не входит — на cutoff
      банк его ещё не пережил;

      сбой до первого наблюдения клиента ничего не отнял у того,
      чего ещё не было;

      сбой после конца покрытия — после ухода клиента или
      закрытия источника — тоже ничего не отнял: наблюдать к
      тому моменту было уже нечего.
    """

    if not value:
        return ()

    days = [str(item) for item in json.loads(value)]

    floor = None if first_seen is None else first_seen.date().isoformat()

    ceiling = cutoff.date().isoformat()

    if last_available is not None:
        ceiling = min(ceiling, last_available.date().isoformat())

    return tuple(
        day
        for day in days
        if day < ceiling and (floor is None or day >= floor)
    )


def coverage_as_of(rows: list[dict], cutoff: datetime) -> tuple[list[SourceState], list[str]]:
    """
    Состояние источника на дату по ДАТИРОВАННЫМ полям.

    Итоговые статусы выгрузки (partial, ended, none) и причины без
    даты известности (no_consent, client_not_onboarded) в
    состояние не входят: они описывают весь период наблюдения, а не
    момент cutoff.

    Дни сбоя датированы, поэтому они в состояние ВХОДЯТ — но
    только те, что уже случились к cutoff.
    """

    states: list[SourceState] = []
    undated: set[str] = set()

    for row in sorted(rows, key=lambda item: item["source"]):

        first_available = row["first_available_at"]
        first_seen = row["first_seen"]
        last_available = row["last_available_at"]

        if first_available is not None and first_available >= cutoff:
            state = COVERAGE_NOT_LAUNCHED
        elif first_seen is None or first_seen >= cutoff:
            state = COVERAGE_NOT_SEEN
        elif last_available is not None and last_available < cutoff:
            state = COVERAGE_ENDED
        else:
            state = COVERAGE_AVAILABLE

        if row.get("coverage_reason") in UNDATED_REASONS:
            undated.add(row["coverage_reason"])

        states.append(
            SourceState(
                source=row["source"],
                state=state,
                first_available_at=first_available,
                first_seen=first_seen,
                last_available_at=last_available,
                reason=row.get("coverage_reason"),
                outage_days=_outages_before(
                    row.get("outage_days"), first_seen, last_available, cutoff
                ),
            )
        )

    notes: list[str] = []

    if undated:
        notes.append(
            "итоговые причины покрытия "
            + ", ".join(sorted(undated))
            + " не датированы и в состояние на дату не входят: они описывают всю выгрузку"
        )

    return states, notes


# ============================================================
# СУЩНОСТИ
# ============================================================


def entity_states_as_of(events: pa.Table, cutoff: datetime) -> list[EntityState]:
    """
    Состояния счетов, карт, договоров, заявок и обращений по
    известным переходам.

    Переход вступает в силу тем мгновением, которым записан:
    отдельного момента вступления в силу у записи нет.

    ОГРАНИЧЕНИЕ. Прежний контракт различал «банк узнал» и
    «изменение действует», и объявленная заранее блокировка
    применялась позже уже состоявшейся разблокировки. Теперь
    такое различие выразить нечем, и событие обязано рождаться
    в тот момент, когда изменение действительно наступает.
    """

    if events.num_rows == 0:
        return []

    columns = ["event_type", "event_time", "stable_event_index"] + [
        name for name in ENTITY_FIELDS if name in events.column_names
    ]

    rows = events.select([name for name in columns if name in events.column_names]).to_pylist()

    tracked: dict[tuple[str, str], dict] = {}

    for row in rows:

        for field_name, kind in ENTITY_FIELDS.items():

            entity_id = row.get(field_name)

            if not entity_id:
                continue

            key = (kind, entity_id)

            item = tracked.setdefault(
                key,
                {
                    "state": None,
                    "since": None,
                    "opening_observed": False,
                    "first_mention": row["event_time"],
                    "last_transition": None,
                    "last_transition_at": None,
                    "effective": [],
                },
            )

            transition = TRANSITIONS.get((kind, row["event_type"]))

            if transition is None:
                continue

            effective = row["event_time"]

            item["effective"].append(
                (
                    effective,
                    row["event_time"],
                    row.get("stable_event_index") or 0,
                    transition,
                )
            )

    # Состояние собирается из переходов в порядке времени
    # события. Отдельного момента вступления в силу у записи
    # нет: изменение действует с того мгновения, которым оно
    # записано.
    for (kind, _entity_id), item in tracked.items():

        for effective, event_time, _index, transition in sorted(item["effective"]):

            item["last_transition"] = transition
            item["last_transition_at"] = event_time

            state = ENTITY_STATE.get((kind, transition))

            if state is not None:
                item["state"] = state
                item["since"] = effective

            if transition in ("opened", "submitted", "activated"):
                item["opening_observed"] = True

    return [
        EntityState(
            kind=kind,
            entity_id=entity_id,
            state=item["state"],
            since=item["since"],
            opening_observed=item["opening_observed"],
            first_mention=item["first_mention"],
            last_transition=item["last_transition"],
            last_transition_at=item["last_transition_at"],
        )
        for (kind, entity_id), item in sorted(tracked.items())
    ]


# ============================================================
# ПЕРЕВОДЫ
# ============================================================


def _acting_sides(rows: list[dict], cutoff: datetime) -> dict[str, dict]:
    """
    Стороны перевода, произошедшие до cutoff.

    Правило то же, что и в истории событий: повтор
    идентификатора пропускается, из оставшихся берётся первая
    строка ленты.
    """

    chosen: dict[str, dict] = {}

    for row in rows:

        if row["is_repeated_event_id"] or row["event_time"] >= cutoff:
            continue

        known = chosen.get(row["event_id"])

        if known is None or -row["raw_row"] > -known["raw_row"]:
            chosen[row["event_id"]] = row

    return chosen


def _matching_side(row: dict, other: dict) -> bool:
    """
    Другая строка описывает ВСТРЕЧНУЮ сторону того же перевода.

    Одного transfer_id недостаточно: сходиться должны сторона
    (одна списывает, другая зачисляет), сумма и счёт. Списание и
    зачисление одной и той же суммы на один и тот же счёт — это
    не перевод, а две записи об одном.
    """

    if other["event_id"] == row["event_id"]:
        return False

    mine = TRANSFER_SIDES.get(row["event_type"])
    theirs = TRANSFER_SIDES.get(other["event_type"])

    if mine is None or theirs is None or mine == theirs:
        return False

    if row["amount"] is None or other["amount"] is None:
        return False

    if int(row["amount"]) != int(other["amount"]):
        return False

    return row.get("account_id") != other.get("account_id")


def transfers_as_of(index: tuple[dict, dict], client_id: str, cutoff: datetime) -> list[TransferSide]:
    """
    Стороны переводов клиента и видимость встречной стороны.

    Пара собирается только среди сторон, чьё событие уже
    произошло: пока встречной стороны нет, перевод для банка
    односторонний, и знать о контрагенте он не может.

    Стоимость по СВОИМ переводам: берутся стороны клиента, и
    встречные ищутся по transfer_id каждой из них. Проход по
    всем переводам группы на каждого клиента давал квадратичную
    стоимость этапа.
    """

    by_client, by_transfer = index

    mine = _acting_sides(by_client.get(client_id, ()), cutoff)

    out: list[TransferSide] = []

    for row in mine.values():

        transfer_id = row["transfer_id"]

        counterparts = _acting_sides(by_transfer.get(transfer_id, ()), cutoff)

        # Встречная сторона это ДРУГАЯ СТРОКА того же перевода, а
        # не обязательно строка другого человека. Перевод между
        # своими счетами наблюдается целиком у одного клиента, и
        # отбор «чей client_id отличается» терял его: обе ноги
        # помечались как перевод без встречной стороны.
        #
        # Но одного общего ключа мало. Пара обязана СОЙТИСЬ ПО
        # ФОРМЕ: противоположная сторона, та же сумма и другой
        # счёт. Иначе стороной становилась бы любая строка с тем
        # же transfer_id — например, вторая попытка перевода той
        # же суммы или соседняя нога, случайно попавшая под тот
        # же ключ.
        holders = sorted(
            {
                item["client_id"]
                for item in counterparts.values()
                if _matching_side(row, item)
            }
        )

        outside = [name for name in holders if name != client_id]

        if not holders:
            pair_state = PAIR_NOT_VISIBLE
            counterpart = None
        elif not outside:
            pair_state = PAIR_OWN
            counterpart = client_id
        else:
            pair_state = PAIR_VISIBLE
            counterpart = outside[0] if len(outside) == 1 else None

        out.append(
            TransferSide(
                transfer_id=transfer_id,
                side=TRANSFER_SIDES.get(row["event_type"]),
                event_id=row["event_id"],
                event_time=row["event_time"],
                amount=row["amount"],
                direction=row["direction"],
                pair_state=pair_state,
                counterpart_client_id=counterpart,
            )
        )

    return sorted(out, key=lambda item: (item.event_time, item.event_id))


# ============================================================
# ОТНОШЕНИЯ
# ============================================================


def relationship_as_of(
    coverage: list[dict],
    cutoff: datetime,
    history_start: datetime,
    profile: dict | None,
) -> Relationship:
    """
    Наблюдаемое начало отношений и честная оговорка о неполноте.

    first_seen это начало наблюдения источником, а не обязательно
    знакомство с банком: более ранний период может быть просто не
    виден.
    """

    starts = [row["first_seen"] for row in coverage if row["first_seen"] is not None and row["first_seen"] < cutoff]

    observed_start = min(starts) if starts else None

    reasons: list[str] = []

    pre_window = 0
    for row in coverage:
        for key, value in row.get("opening_state_values") or []:
            if key == "contracts_before_window" and value:
                pre_window = max(pre_window, int(value))

    if observed_start is not None and observed_start <= history_start:
        reasons.append("наблюдение начинается с границы выгрузки: более ранний период неизвестен")

    if pre_window:
        reasons.append(f"opening_state объявляет {pre_window} договор(ов) до начала окна")

    observed_days = (cutoff - observed_start).days if observed_start is not None else None

    if profile is not None and profile.get("relationship_months") is not None and observed_days is not None:
        declared_days = int(profile["relationship_months"]) * 30
        if declared_days > observed_days + 31:
            reasons.append(
                f"профиль заявляет {profile['relationship_months']} мес. отношений, "
                f"наблюдается {observed_days // 30} мес."
            )

    closed_at = None
    closed_reason = None

    for row in coverage:
        end = row["last_available_at"]
        if end is not None and end < cutoff and row.get("coverage_reason") == "relationship_closed":
            if closed_at is None or end < closed_at:
                closed_at = end
                closed_reason = row["coverage_reason"]

    return Relationship(
        observed_start=observed_start,
        observed_days=observed_days,
        history_incomplete=bool(reasons),
        incomplete_reasons=tuple(reasons),
        closed_at=closed_at,
        closed_reason=closed_reason,
    )


# ============================================================
# СПРАВОЧНИК ПРОДУКТОВ
# ============================================================


def product_key(product_id: str, product_version=None, tariff_version=None) -> str:
    """
    Адрес строки справочника: продукт и названная событием версия.

    Прежний продукт версии не называет, поэтому обе её части
    остаются пустыми.
    """

    return f"{product_id}|v{product_version}|t{tariff_version}"


def products_as_of(products: pa.Table | None, events: pa.Table, cutoff: datetime) -> dict[str, dict]:
    """
    Атрибуты продуктов, названных видимыми событиями, по версии
    условий самого события.

    Справочник с несколькими версиями не размножает событие: берётся
    ровно та строка, которую событие называет, и только если она уже
    известна к cutoff.

    Прежний продукт перехода версии не называет: для него берётся
    последняя версия, известная к cutoff.
    """

    if products is None or events.num_rows == 0 or "product_id" not in events.column_names:
        return {}

    wanted: set[tuple] = set()

    columns = ["product_id", "product_version", "tariff_version", "previous_product_id"]

    for row in events.select([name for name in columns if name in events.column_names]).to_pylist():

        if row.get("product_id"):
            wanted.add((row["product_id"], row.get("product_version"), row.get("tariff_version")))

        if row.get("previous_product_id"):
            wanted.add((row["previous_product_id"], None, None))

    if not wanted:
        return {}

    catalogue = products.to_pylist()

    out: dict[str, dict] = {}

    for product_id, product_version, tariff_version in sorted(wanted, key=lambda item: str(item)):

        candidates = [
            row
            for row in catalogue
            if row["product_id"] == product_id
            and (product_version is None or row["product_version"] == product_version)
            and (tariff_version is None or row["tariff_version"] == tariff_version)
        ]

        known = [row for row in candidates if row["valid_from"] is None or row["valid_from"] < cutoff]

        key = product_key(product_id, product_version, tariff_version)

        if not candidates:
            out[key] = {"state": "not_in_catalogue"}
            continue

        if not known:
            out[key] = {"state": "catalogue_version_unknown_at_cutoff"}
            continue

        row = max(known, key=lambda item: (item["valid_from"] or datetime.min, item["product_version"]))

        out[key] = {
            "state": "known",
            "product_code": row["product_code"],
            "product_family": row["product_family"],
            "product_name": row["product_name"],
            "status": row["status"],
            "valid_from": row["valid_from"],
            "product_version": row["product_version"],
            "tariff_version": row["tariff_version"],
        }

    return out


# ============================================================
# ИНТЕРФЕЙС
# ============================================================


def history_as_of(store: CanonicalStore, client: str | int, cutoff: datetime) -> ClientHistory:
    """
    Что банк знал о клиенте строго до cutoff.
    """

    if cutoff > store.extract_time:
        raise HistoryError(
            f"cutoff {cutoff.isoformat()} позже границы выгрузки {store.extract_time.isoformat()}: "
            "за ней у банка нет ничего"
        )

    row = store.client_row(client)

    client_id = row["client_id"]

    table = store.client_events(client)

    events, counts, _ = visible_events(table, cutoff)

    profile_rows = store.profile_rows(client_id)
    profile, profile_meta = profile_of(profile_rows)

    coverage_rows = store.coverage_rows(client_id)
    coverage, coverage_notes = coverage_as_of(coverage_rows, cutoff)

    entities = entity_states_as_of(events, cutoff)
    transfers = transfers_as_of(store.transfer_index, client_id, cutoff)
    relationship = relationship_as_of(coverage_rows, cutoff, store.history_start, profile)
    products = products_as_of(store.products, events, cutoff)

    limitations = list(coverage_notes)

    limitations.extend(relationship.incomplete_reasons)

    without_opening = [item for item in entities if not item.opening_observed and item.kind in ("account", "card", "contract")]
    if without_opening:
        limitations.append(
            f"сущностей без наблюдаемого открытия: {len(without_opening)} "
            "(открыты до начала наблюдения либо открытие не попало в выгрузку)"
        )

    return ClientHistory(
        client_id=client_id,
        client_idx=row["client_idx"],
        cutoff=cutoff,
        events=events,
        counts=counts,
        profile=profile,
        profile_meta=profile_meta,
        coverage=coverage,
        entities=entities,
        transfers=transfers,
        relationship=relationship,
        products=products,
        limitations=limitations,
    )


__all__ = [
    "COVERAGE_AVAILABLE",
    "COVERAGE_ENDED",
    "COVERAGE_NOT_LAUNCHED",
    "COVERAGE_NOT_SEEN",
    "ENTITY_STATE",
    "INTERNAL_COLUMNS",
    "PAIR_NOT_VISIBLE",
    "PAIR_OWN",
    "PAIR_VISIBLE",
    "STAGE",
    "STAGE_VERSION",
    "UNDATED_REASONS",
    "CanonicalStore",
    "ClientHistory",
    "EntityState",
    "HistoryError",
    "Relationship",
    "SourceState",
    "TransferSide",
    "coverage_as_of",
    "entity_states_as_of",
    "history_as_of",
    "product_key",
    "products_as_of",
    "profile_of",
    "relationship_as_of",
    "transfers_as_of",
    "visible_events",
]
