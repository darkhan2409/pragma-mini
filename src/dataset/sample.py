from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from src.preprocessing.settings import GroupWindow
from src.tokenization.layout import PAD, FrozenArtifacts

from .context import EventStub, Selection, select
from .dependencies import Dependencies, resolve
from .encoding import EncodedHistory
from .settings import ContextPolicy
from .targets import coverage_codes, coverage_details, eligible, history_age_days, hours_to_cutoff


# ============================================================
# ИДЕЯ
# ============================================================
#
# Один пример это один клиент на один срез.
#
# Внутри примера три круга колонок, и граница между ними не
# декоративная:
#
#   model    входы модели: токены событий и профиля, каналы
#            времени, доступность источников;
#   masker   границы значений, признак допустимой цели, вес и
#            происхождение значений;
#   service  трассировка: тождество записей, причины отбора,
#            даты покрытия, ограничения. В embedding ничего из
#            этого не входит.
#
# Адрес значения это пара «событие, столбец внутри него». Он
# одинаков и в одиночном примере, и после сборки batch, поэтому
# Masker прячет значение одним и тем же кодом в обоих случаях.
#
# Клиент без событий остаётся примером. Никакой выдуманной
# покупки: пустая история это факт о клиенте, а не пустое место.
# ============================================================


class SampleError(ValueError):
    """
    Пример собрать нельзя.
    """


@dataclass
class Sample:
    """
    Готовый пример.
    """

    # --- тождество ---
    sample_id: str
    group: str
    client_id: str
    cutoff: datetime
    weight: float
    sample_seed: int

    # --- события: входы модели ---
    key_ids: np.ndarray
    value_ids: np.ndarray
    positions: np.ndarray
    event_starts: np.ndarray
    event_lengths: np.ndarray

    # --- события: каналы ---
    calendar: np.ndarray
    hours_to_cutoff: np.ndarray
    event_eligible: np.ndarray

    # --- значения событий ---
    value_event: np.ndarray
    value_start: np.ndarray
    value_length: np.ndarray
    value_key_id: np.ndarray

    # --- профиль ---
    profile_key_ids: np.ndarray
    profile_value_ids: np.ndarray
    profile_positions: np.ndarray
    profile_value_start: np.ndarray
    profile_value_length: np.ndarray
    profile_value_key_id: np.ndarray
    profile_state: str
    has_profile: bool

    # --- клиент на срез ---
    #
    # Состояние источника ОДНО на пример и описывает момент
    # среза, а не доступность во время каждого события истории.
    coverage_at_cutoff: np.ndarray
    history_age_days: float | None
    history_age_reason: str | None

    # --- цели и отбор ---
    n_eligible_events: int
    has_targets: bool
    truncated: bool
    selection: dict

    # --- происхождение ---
    dependencies: Dependencies

    # --- служебное ---
    events: list[dict] = field(default_factory=list)
    coverage: list[dict] = field(default_factory=list)
    relationship: dict = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    # --------------------------------------------------------

    @property
    def n_events(self) -> int:
        return int(self.event_starts.size)

    @property
    def n_tokens(self) -> int:
        return int(self.key_ids.size)

    @property
    def n_values(self) -> int:
        return int(self.value_event.size)

    @property
    def profile_tokens(self) -> int:
        return int(self.profile_key_ids.size)

    def check(self, artifacts: FrozenArtifacts) -> None:
        """
        Инварианты примера.

        Проверяется то, на что будет опираться Masker и модель:
        согласованность длин, покрытие токенов границами, один
        маркер на запись и принадлежность каждого ID своему
        диапазону.
        """

        if not (self.key_ids.size == self.value_ids.size == self.positions.size):
            raise SampleError(f"{self.sample_id}: массивы токенов разной длины")

        if self.event_starts.size != self.event_lengths.size:
            raise SampleError(f"{self.sample_id}: границы событий разной длины")

        if self.calendar.size != self.n_events * 6:
            raise SampleError(f"{self.sample_id}: календарь не по шесть чисел на событие")

        for name in ("hours_to_cutoff", "event_eligible"):
            if getattr(self, name).size != self.n_events:
                raise SampleError(f"{self.sample_id}: канал {name} не по одному значению на событие")

        # События лежат подряд и покрывают все токены.
        covered = 0

        for start, length in zip(self.event_starts.tolist(), self.event_lengths.tolist()):

            if start != covered:
                raise SampleError(f"{self.sample_id}: событие начинается в {start}, а покрыто {covered}")

            if length < 1:
                raise SampleError(f"{self.sample_id}: событие без единого токена")

            # Маркер: одинаковый код в обоих слотах на нулевой
            # позиции внутри события.
            if self.key_ids[start] != self.value_ids[start] or self.positions[start] != 0:
                raise SampleError(f"{self.sample_id}: событие начинается не с маркера")

            covered += length

        if covered != self.n_tokens:
            raise SampleError(f"{self.sample_id}: границы покрывают {covered} токенов из {self.n_tokens}")

        _check_record(self.sample_id, "профиль", self.profile_key_ids, self.profile_value_ids,
                      self.profile_positions)

        # Границы значений: внутри своего события, без нахлёста, и
        # маркер под них не попадает.
        _check_spans(self.sample_id, self.value_event, self.value_start, self.value_length,
                     self.event_lengths.tolist())

        _check_spans(self.sample_id, np.zeros(self.profile_value_start.size, dtype=np.int64),
                     self.profile_value_start, self.profile_value_length, [self.profile_tokens])

        # ID в своих диапазонах, [PAD] нигде не написан.
        pad = artifacts.special(PAD)
        size = artifacts.size

        for name in ("key_ids", "value_ids", "profile_key_ids", "profile_value_ids"):

            values = getattr(self, name)

            if values.size and (values.min() < 0 or values.max() >= size):
                raise SampleError(f"{self.sample_id}: {name} выходит за пространство ID {size}")

            if values.size and bool((values == pad).any()):
                raise SampleError(
                    f"{self.sample_id}: в {name} встретился [PAD]. Он существует только для "
                    "выравнивания batch и в сохранённом примере невозможен"
                )

    def as_index_row(self, shard: str, row: int) -> dict:
        """
        Строка указателя: где лежит пример и каков он на вид.
        """

        return {
            "sample_id": self.sample_id,
            "group": self.group,
            "client_id": self.client_id,
            "cutoff": self.cutoff,
            "shard": shard,
            "row": row,
            "n_events": self.n_events,
            "n_tokens": self.n_tokens,
            "n_values": self.n_values,
            "profile_tokens": self.profile_tokens,
            "weight": self.weight,
            "has_targets": self.has_targets,
            "n_eligible_events": self.n_eligible_events,
            "truncated": self.truncated,
            "has_profile": self.has_profile,
            "excluded_events": int(self.selection.get("excluded_events", 0)),
            "excluded_tokens": int(self.selection.get("excluded_tokens", 0)),
            "excluded_eligible": int(self.selection.get("excluded_eligible", 0)),
            "excluded_milestones": int(self.selection.get("excluded_milestones", 0)),
            "budget_binding": self.selection.get("budget_binding"),
        }


def _check_record(sample_id: str, what: str, key_ids: np.ndarray, value_ids: np.ndarray,
                  positions: np.ndarray) -> None:

    if not (key_ids.size == value_ids.size == positions.size):
        raise SampleError(f"{sample_id}: {what} — массивы разной длины")

    if key_ids.size == 0:
        raise SampleError(f"{sample_id}: {what} без единого токена: маркер обязан быть всегда")

    if key_ids[0] != value_ids[0] or positions[0] != 0:
        raise SampleError(f"{sample_id}: {what} начинается не с маркера")


def _check_spans(sample_id: str, owner: np.ndarray, start: np.ndarray, length: np.ndarray,
                 lengths: list[int]) -> None:

    if not (owner.size == start.size == length.size):
        raise SampleError(f"{sample_id}: границы значений разной длины")

    covered: dict[int, int] = {}

    for index in range(owner.size):

        which = int(owner[index])
        first = int(start[index])
        size = int(length[index])

        # Маркер занимает нулевой столбец записи и значением не
        # является: span на него наложиться не может.
        expected = covered.get(which, 1)

        if first != expected:
            raise SampleError(
                f"{sample_id}: значение записи {which} начинается в столбце {first}, "
                f"а покрыто до {expected}"
            )

        covered[which] = first + size

    for which, size in covered.items():
        if size != lengths[which]:
            raise SampleError(
                f"{sample_id}: значения записи {which} покрывают {size} токенов из {lengths[which]}"
            )


def sample_id_of(client_id: str, cutoff: datetime) -> str:
    return f"{client_id}@{cutoff.isoformat()}"


def sample_seed_of(sample_id: str) -> int:
    """
    Устойчивый номер примера для будущего Masker.

    Считается от тождества примера, а не от его места в файле:
    иначе маска зависела бы от порядка чтения и от размера
    batch.
    """

    digest = hashlib.blake2b(sample_id.encode("utf-8"), digest_size=8).digest()

    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def build_sample(
    artifacts: FrozenArtifacts,
    encoded: EncodedHistory,
    group: str,
    window: GroupWindow,
    weight: float,
    sources: tuple[str, ...],
    policy: ContextPolicy,
) -> Sample:
    """
    Пример из закодированной истории на дату.
    """

    cutoff = encoded.cutoff

    flags = [eligible(item.event_time, window) for item in encoded.events]

    stubs = [
        EventStub(index=number, event_type=item.event_type, n_tokens=item.n_tokens,
                  eligible=flags[number])
        for number, item in enumerate(encoded.events)
    ]

    selection: Selection = select(stubs, policy)

    if encoded.profile.n_tokens > policy.max_profile_tokens:
        raise SampleError(
            f"клиент {encoded.client_id} на срезе {cutoff.isoformat()}: представление профиля "
            f"занимает {encoded.profile.n_tokens} токенов при пределе {policy.max_profile_tokens}"
        )

    key_ids: list[int] = []
    value_ids: list[int] = []
    positions: list[int] = []
    event_starts: list[int] = []
    event_lengths: list[int] = []
    calendar: list[float] = []
    to_cutoff: list[float] = []
    event_eligible: list[bool] = []

    value_event: list[int] = []
    value_start: list[int] = []
    value_length: list[int] = []
    value_key_id: list[int] = []
    value_offsets: list[int] = []

    for slot, position in enumerate(selection.kept):

        item = encoded.events[position]
        record = item.record

        event_starts.append(len(key_ids))
        event_lengths.append(record.n_tokens)

        value_offsets.append(len(value_event))

        for local in range(record.n_values):
            value_event.append(slot)
            value_start.append(record.value_starts[local])
            value_length.append(record.value_lengths[local])
            value_key_id.append(record.key_ids[record.value_starts[local]])

        key_ids.extend(record.key_ids)
        value_ids.extend(record.value_ids)
        positions.extend(record.positions)

        if len(item.calendar) != 6:
            raise SampleError(
                f"событие {item.event_id}: календарь из {len(item.calendar)} чисел вместо шести"
            )

        calendar.extend(item.calendar)
        to_cutoff.append(hours_to_cutoff(cutoff, item.event_time))
        event_eligible.append(flags[position])

    profile = encoded.profile
    meta = encoded.profile_meta

    age, age_reason = history_age_days(encoded.relationship, cutoff)

    dependencies = resolve(
        events=encoded.events,
        kept=selection.kept,
        value_offsets=value_offsets,
        profile=profile,
        cause_of=encoded.cause_of,
    )

    identifier = sample_id_of(encoded.client_id, cutoff)

    sample = Sample(
        sample_id=identifier,
        group=group,
        client_id=encoded.client_id,
        cutoff=cutoff,
        weight=weight,
        sample_seed=sample_seed_of(identifier),
        key_ids=_ints(key_ids),
        value_ids=_ints(value_ids),
        positions=_ints(positions),
        event_starts=_ints(event_starts),
        event_lengths=_ints(event_lengths),
        calendar=np.asarray(calendar, dtype=np.float32),
        hours_to_cutoff=np.asarray(to_cutoff, dtype=np.float64),
        event_eligible=np.asarray(event_eligible, dtype=bool),
        value_event=_ints(value_event),
        value_start=_ints(value_start),
        value_length=_ints(value_length),
        value_key_id=_ints(value_key_id),
        profile_key_ids=_ints(profile.key_ids),
        profile_value_ids=_ints(profile.value_ids),
        profile_positions=_ints(profile.positions),
        profile_value_start=_ints(profile.value_starts),
        profile_value_length=_ints(profile.value_lengths),
        profile_value_key_id=_ints(
            [profile.key_ids[start] for start in profile.value_starts]
        ),
        profile_state=meta.get("state", ""),
        has_profile=encoded.has_profile,
        coverage_at_cutoff=np.asarray(coverage_codes(encoded.coverage, sources), dtype=np.int8),
        history_age_days=age,
        history_age_reason=age_reason,
        n_eligible_events=sum(1 for slot in selection.kept if flags[slot]),
        has_targets=any(flags[slot] for slot in selection.kept),
        truncated=selection.truncated,
        selection=_selection_report(selection, policy),
        dependencies=dependencies,
        events=_service_rows(encoded, selection, flags),
        coverage=coverage_details(encoded.coverage),
        relationship=_relationship(encoded.relationship),
        limitations=list(encoded.limitations),
    )

    sample.check(artifacts)

    return sample


def _ints(values) -> np.ndarray:
    return np.asarray(list(values), dtype=np.int32)


def _selection_report(selection: Selection, policy: ContextPolicy) -> dict:

    out = selection.as_dict()

    out["policy"] = policy.policy
    out["max_events"] = policy.max_events
    out["max_tokens"] = policy.max_tokens
    out["milestone_share"] = policy.milestone_share

    return out


def _relationship(relationship) -> dict:

    if relationship is None:
        return {}

    return {
        "observed_start": getattr(relationship, "observed_start", None),
        "observed_days": getattr(relationship, "observed_days", None),
        "history_incomplete": bool(getattr(relationship, "history_incomplete", False)),
        "incomplete_reasons": list(getattr(relationship, "incomplete_reasons", ()) or ()),
    }


def _service_rows(encoded: EncodedHistory, selection: Selection, flags: list[bool]) -> list[dict]:
    """
    Строка на КАЖДОЕ видимое событие, а не только на отобранное.

    Исключённое событие обязано остаться названным: иначе
    «контекст потерян» превратилось бы в «контекста и не было», а
    зависимость на него нельзя было бы объяснить.
    """

    reason_of = dict(zip(selection.kept, selection.reasons))
    slot_of = {position: slot for slot, position in enumerate(selection.kept)}
    excluded_of = dict(zip(selection.excluded, selection.excluded_reasons))

    out: list[dict] = []

    for position, item in enumerate(encoded.events):

        kept = position in slot_of

        out.append(
            {
                "event_index": slot_of.get(position, -1),
                "position": position,
                "kept": kept,
                "selection_reason": reason_of.get(position),
                "exclusion_reason": excluded_of.get(position),
                "event_id": item.event_id,
                "stable_event_index": item.stable_event_index,
                "event_time": item.event_time,
                "source": item.source,
                "event_type": item.event_type,
                "n_values": item.n_values,
                "n_tokens": item.n_tokens,
                "eligible": flags[position],
                "value_keys": list(item.record.value_keys) if kept else [],
                "unknown_keys": list(item.record.unknown_keys) if kept else [],
                "refs": item.refs if kept else {},
                "provenance": item.provenance if kept else {},
                "absent_reasons": item.absent_reasons if kept else {},
            }
        )

    return out


def sample_from_row(row: dict) -> Sample:
    """
    Пример обратно из строки набора.

    Служебные колонки здесь не восстанавливаются: они лежат в
    указателе и в таблице событий, а модели и Masker не нужны.
    Пустое служебное поле честнее выдуманного.
    """

    dependencies = Dependencies(
        value=list(row.get("dep_value") or ()),
        source_event=list(row.get("dep_source_event") or ()),
        source_value=list(row.get("dep_source_value") or ()),
        status=list(row.get("dep_status") or ()),
        key=[],
    )

    return Sample(
        sample_id=row["sample_id"],
        group=row["group"],
        client_id=row["client_id"],
        cutoff=row["cutoff"],
        weight=float(row["weight"]),
        sample_seed=int(row["sample_seed"]),
        key_ids=_ints(row["key_ids"]),
        value_ids=_ints(row["value_ids"]),
        positions=_ints(row["positions"]),
        event_starts=_ints(row["event_starts"]),
        event_lengths=_ints(row["event_lengths"]),
        calendar=np.asarray(row["calendar"], dtype=np.float32),
        hours_to_cutoff=np.asarray(row["hours_to_cutoff"], dtype=np.float64),
        event_eligible=np.asarray(row["event_eligible"], dtype=bool),
        value_event=_ints(row["value_event"]),
        value_start=_ints(row["value_start"]),
        value_length=_ints(row["value_length"]),
        value_key_id=_ints(row["value_key_id"]),
        profile_key_ids=_ints(row["profile_key_ids"]),
        profile_value_ids=_ints(row["profile_value_ids"]),
        profile_positions=_ints(row["profile_positions"]),
        profile_value_start=_ints(row["profile_value_start"]),
        profile_value_length=_ints(row["profile_value_length"]),
        profile_value_key_id=_ints(row["profile_value_key_id"]),
        profile_state=row["profile_state"],
        has_profile=bool(row["has_profile"]),
        coverage_at_cutoff=np.asarray(row["coverage_at_cutoff"], dtype=np.int8),
        history_age_days=row["history_age_days"],
        history_age_reason=None,
        n_eligible_events=int(row["n_eligible_events"]),
        has_targets=bool(row["has_targets"]),
        truncated=bool(row["truncated"]),
        selection={},
        dependencies=dependencies,
    )


__all__ = [
    "Sample",
    "SampleError",
    "build_sample",
    "sample_from_row",
    "sample_id_of",
    "sample_seed_of",
]
