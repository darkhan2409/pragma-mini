from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.dataset.settings import dataset_dir
from src.masking.apply import apply
from src.masking.choose import choose
from src.masking.settings import MaskingConfig
from src.temporal.position import TemporalError
from src.temporal.samples import SamplesError, TemporalGroup
from src.tokenization.specials import EVT, MASK, PAD, UNK, USR, load_special_tokens


# ============================================================
# ВХОД СЛОЯ ЭМБЕДДИНГОВ
# ============================================================
#
# Батч — одна группа строк собранного набора:
#
#   data/05_dataset/<group>/samples.parquet
#
# Маска разыгрывается при чтении тем же маскером, что у обучения
# (choose + apply, конфиг masking), время анкеты — тем же читателем
# набора (TemporalGroup). Файлов между набором и слоем нет.
#
# Слой считает прямоугольные тензоры [B, T] и [B, P], поэтому здесь,
# и только здесь, клиенты группы дополняются заполнителем до общей
# ширины: [PAD] в ключах и значениях, ноль в позициях, маска
# «настоящее / заполнитель» рядом. Настоящее лежит слева, поэтому
# маска это ровно «номер меньше длины», а смещения событий остаются
# теми же числами. На диск выровненное не пишется.
#
# labels сюда не приходят вовсе: из маскера берутся только видимые
# значения, и обещание «в эмбеддинг подаются только видимые
# значения» видно по коду чтения, а не по доверию к коду ниже.
#
# Календарь и event_time_log не читаются: это вход эмбеддингов, а
# не энкодера события. Энкодеру события календарь нужен, энкодеру
# анкеты — время её токенов (profile_time_log), поэтому у читателя
# есть флаги: просить колонку или нет.
# ============================================================


# Что берётся из набора. Календаря и времени анкеты здесь намеренно
# нет: их просят отдельно.
SAMPLE_COLUMNS = [
    "client_id",
    "key_ids",
    "value_ids",
    "positions",
    "event_starts",
    "event_lengths",
    "event_time",
    "target_event_mask",
    "profile_key_ids",
    "profile_value_ids",
    "profile_positions",
]

# Шесть чисел на событие. Нужны энкодеру события, не входу.
CALENDAR_COLUMN = "calendar"

# Давность токенов анкеты до cutoff. Нужна энкодеру анкеты, не
# входу.
PROFILE_TIME_COLUMN = "profile_time_log"

CALENDAR_PER_EVENT = 6

# Устройство последовательности: границы событий и их время, уже
# выровненные. Слой их не видит — по ним собирают события те, кто
# идёт дальше.
STRUCTURE_COLUMNS = [
    "client_id",
    "n_tokens",
    "n_events",
    "profile_n_tokens",
    "event_starts",
    "event_lengths",
    "event_time",
    "event_mask",
    "target_event_mask",
]


class InputError(ValueError):
    """
    Вход модели собрать нельзя.
    """


@dataclass(frozen=True)
class BatchInput:
    """
    Ровно то, что видит слой эмбеддингов.

    Значения уже видимые: часть из них заменена маскером на
    [MASK] или [UNK]. Исходных значений и labels здесь нет.
    """

    batch_index: int
    client_ids: tuple[str, ...]

    # события, [B, T]
    key_ids: torch.Tensor
    value_ids: torch.Tensor
    positions: torch.Tensor
    token_mask: torch.Tensor

    # анкета, [B, P]
    profile_key_ids: torch.Tensor
    profile_value_ids: torch.Tensor
    profile_positions: torch.Tensor
    profile_token_mask: torch.Tensor

    @property
    def clients(self) -> int:
        return int(self.key_ids.shape[0])

    @property
    def width(self) -> int:
        return int(self.key_ids.shape[1])

    @property
    def profile_width(self) -> int:
        return int(self.profile_key_ids.shape[1])


@dataclass(frozen=True)
class Loaded:
    """
    Батч целиком: вход слоя, материал отчёта и итоги сверки.
    """

    model: BatchInput
    rows: list[dict]

    # [B, E * 6], и только если календарь просили. Иначе None:
    # пустая матрица выглядела бы как «календарь есть, просто
    # нулевой».
    calendar: np.ndarray | None = None

    # [B, P], и только если время анкеты просили.
    profile_time_log: np.ndarray | None = None


class Source:
    """
    Вход слоя одной группы, открытый один раз.

    masking — по какому конфигу разыгрывается маска; без него
    MaskingConfig(), как у стадий 13 и обучения по умолчанию.
    """

    def __init__(
        self,
        group: str,
        with_calendar: bool = False,
        with_profile_time: bool = False,
        masking: MaskingConfig | None = None,
    ):

        self.group = group
        self.with_calendar = with_calendar
        self.with_profile_time = with_profile_time
        self.masking = masking if masking is not None else MaskingConfig()

        self.columns = list(SAMPLE_COLUMNS)

        if with_calendar:
            self.columns.append(CALENDAR_COLUMN)

        if with_profile_time:
            self.columns.append(PROFILE_TIME_COLUMN)

        self.directory = dataset_dir(group)

        try:
            self._samples = TemporalGroup(group, self.directory)
        except SamplesError as error:
            raise InputError(str(error)) from error

        self._specials = load_special_tokens()

    @property
    def count(self) -> int:
        return self._samples.count

    def batch(self, index: int) -> Loaded:
        """
        Одна группа строк набора, выровненная в памяти.
        """

        if index < 0 or index >= self.count:
            raise InputError(
                f"батча {index} нет: в группе {self.count} батчей, "
                f"номера от 0 до {self.count - 1}"
            )

        try:
            rows = self._samples.row_group(index, columns=self.columns).to_pylist()
        except TemporalError as error:
            raise InputError(str(error)) from error

        if not rows:
            raise InputError(f"батч {index} пуст: клиентов в нём нет")

        pad = self._specials[PAD]

        for row in rows:
            row["visible_value_ids"] = self._visible(row)

        padded = _padded(rows, pad)

        _check_markers(index, padded, (self._specials[EVT], self._specials[USR]))

        model = BatchInput(
            batch_index=index,
            client_ids=tuple(row["client_id"] for row in padded),
            key_ids=_stack(padded, "key_ids", torch.int64),
            value_ids=_stack(padded, "visible_value_ids", torch.int64),
            positions=_stack(padded, "positions", torch.int64),
            token_mask=_stack(padded, "token_mask", torch.bool),
            profile_key_ids=_stack(padded, "profile_key_ids", torch.int64),
            profile_value_ids=_stack(padded, "profile_value_ids", torch.int64),
            profile_positions=_stack(padded, "profile_positions", torch.int64),
            profile_token_mask=_stack(padded, "profile_token_mask", torch.bool),
        )

        calendar = None

        if self.with_calendar:
            calendar = np.asarray([row[CALENDAR_COLUMN] for row in padded], dtype=np.float32)

        profile_time_log = None

        if self.with_profile_time:
            profile_time_log = np.asarray(
                [row[PROFILE_TIME_COLUMN] for row in padded], dtype=np.float32
            )

        return Loaded(
            model=model,
            rows=[{name: row[name] for name in STRUCTURE_COLUMNS} for row in padded],
            calendar=calendar,
            profile_time_log=profile_time_log,
        )

    def _visible(self, row: dict) -> list[int]:
        """
        Значения клиента, которые модели разрешено видеть.
        """

        selection = choose(self.group, row, self.masking)

        masked = apply(
            row["client_id"], row, selection.choices,
            self._specials[MASK], self._specials[UNK], selection.corrupted,
        )

        return masked["value_ids"]


def load_batch(group: str, index: int) -> Loaded:
    """
    Один батч группы, когда остальные не нужны.
    """

    return Source(group).batch(index)


def _padded(rows: list[dict], pad: int) -> list[dict]:
    """
    Клиенты группы, дополненные до общей ширины по трём осям:
    токены событий T, события E (календарь — 6 * E) и токены
    анкеты P.

    Заполнитель смещения события — КОНЕЦ последовательности, а не
    ноль: пустое событие в конце безвредно, а ноль указывал бы на
    первый настоящий токен клиента. У времени события заполнителя
    нет (None): нулевого момента не существует.
    """

    tokens = max(len(row["key_ids"]) for row in rows)
    events = max(len(row["event_starts"]) for row in rows)
    profile = max(len(row["profile_key_ids"]) for row in rows)

    result = []

    for row in rows:

        n_tokens = len(row["key_ids"])
        n_events = len(row["event_starts"])
        profile_n_tokens = len(row["profile_key_ids"])

        item = {
            "client_id": row["client_id"],
            "n_tokens": n_tokens,
            "n_events": n_events,
            "profile_n_tokens": profile_n_tokens,
            "key_ids": _pad(row["key_ids"], tokens, pad),
            "visible_value_ids": _pad(row["visible_value_ids"], tokens, pad),
            "positions": _pad(row["positions"], tokens, 0),
            "token_mask": _mask(n_tokens, tokens),
            "event_starts": _pad(row["event_starts"], events, n_tokens),
            "event_lengths": _pad(row["event_lengths"], events, 0),
            "event_time": _pad(row["event_time"], events, None),
            "event_mask": _mask(n_events, events),
            "target_event_mask": _pad(row["target_event_mask"], events, False),
            "profile_key_ids": _pad(row["profile_key_ids"], profile, pad),
            "profile_value_ids": _pad(row["profile_value_ids"], profile, pad),
            "profile_positions": _pad(row["profile_positions"], profile, 0),
            "profile_token_mask": _mask(profile_n_tokens, profile),
        }

        if CALENDAR_COLUMN in row:
            item[CALENDAR_COLUMN] = _pad(row[CALENDAR_COLUMN], events * CALENDAR_PER_EVENT, 0.0)

        # Ноль заполнителя совпадает с нулём [USR] и Attributes:
        # отличить их можно ТОЛЬКО по маске анкеты.
        if PROFILE_TIME_COLUMN in row:
            item[PROFILE_TIME_COLUMN] = _pad(row[PROFILE_TIME_COLUMN], profile, 0.0)

        result.append(item)

    return result


def _pad(values, width: int, fill) -> list:
    """
    Значения слева, заполнитель справа.
    """

    values = list(values)

    return values + [fill] * (width - len(values))


def _mask(length: int, width: int) -> list[bool]:
    return [True] * length + [False] * (width - length)


def _stack(rows: list[dict], name: str, dtype) -> torch.Tensor:
    return torch.as_tensor(np.asarray([row[name] for row in rows]), dtype=dtype)


def _check_markers(index: int, rows: list[dict], markers: tuple[int, int]) -> None:
    """
    У маркера один ID в обоих слотах, и маскер его не трогал.
    """

    for row in rows:

        keys = np.asarray(row["key_ids"])
        values = np.asarray(row["visible_value_ids"])

        wrong = np.isin(keys, markers) & (values != keys)

        if wrong.any():
            position = int(np.argwhere(wrong)[0][0])
            raise InputError(
                f"батч {index}, клиент {row['client_id']}, позиция {position}: "
                f"маркер {keys[position]} в слоте ключа, но {values[position]} в слоте значения"
            )


__all__ = [
    "CALENDAR_COLUMN",
    "CALENDAR_PER_EVENT",
    "PROFILE_TIME_COLUMN",
    "SAMPLE_COLUMNS",
    "STRUCTURE_COLUMNS",
    "BatchInput",
    "InputError",
    "Loaded",
    "Source",
    "load_batch",
]
