from __future__ import annotations

import numpy as np

from src.dataset.settings import dataset_dir
from src.masking.apply import apply_selection
from src.masking.choose import choose
from src.masking.settings import MaskingConfig
from src.masking.weights import WeightsError, load_value_weights
from src.temporal.position import TemporalError
from src.temporal.samples import SamplesError, TemporalGroup
from src.tokenization.specials import EVT, MASK, UNK, USR, load_special_tokens


# ============================================================
# ВХОД СЛОЯ ЭМБЕДДИНГОВ
# ============================================================
#
# Батч — одна группа строк собранного набора:
#
#   data/05_dataset/<group>/samples.parquet
#
# Маска разыгрывается при чтении тем же маскером, что у обучения
# (choose + apply_selection, конфиг masking), время — тем же
# читателем набора (TemporalGroup). Файлов между набором и слоем нет.
#
# Этап 06 так сверяет каждую группу строк: время читается, маска
# разыгрывается, а маркеры [EVT] и [USR] в слоте значения остаются
# собой. Прямоугольных тензоров здесь нет: модель считает клиентов
# подряд, без заполнителя (model.pack), и выравнивать их не для кого.
#
# labels сюда не приходят вовсе: из маскера берутся только видимые
# значения, и обещание «в эмбеддинг подаются только видимые
# значения» видно по коду чтения, а не по доверию к коду ниже.
# ============================================================


# Что берётся из набора.
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

# Шесть чисел календаря на событие. Нужны энкодеру события, не входу.
CALENDAR_PER_EVENT = 6


class InputError(ValueError):
    """
    Вход модели собрать нельзя.
    """


class Source:
    """
    Вход слоя одной группы, открытый один раз.

    masking — по какому конфигу разыгрывается маска; без него
    MaskingConfig(), как у обучения по умолчанию.
    """

    def __init__(self, group: str, masking: MaskingConfig | None = None):

        self.group = group
        self.masking = masking if masking is not None else MaskingConfig()

        self.directory = dataset_dir(group)

        try:
            self._samples = TemporalGroup(group, self.directory)
        except SamplesError as error:
            raise InputError(str(error)) from error

        self._specials = load_special_tokens()

        self._weights = None

        if self.masking.informativeness_weighted_masking:
            try:
                self._weights = load_value_weights()
            except WeightsError as error:
                raise InputError(str(error)) from error

    @property
    def count(self) -> int:
        return self._samples.count

    def rows(self, index: int) -> list[dict]:
        """
        Строки одной группы набора с видимыми значениями
        (visible_value_ids) и сверенными маркерами.
        """

        if index < 0 or index >= self.count:
            raise InputError(
                f"батча {index} нет: в группе {self.count} батчей, "
                f"номера от 0 до {self.count - 1}"
            )

        try:
            rows = self._samples.row_group(index, columns=SAMPLE_COLUMNS).to_pylist()
        except TemporalError as error:
            raise InputError(str(error)) from error

        if not rows:
            raise InputError(f"батч {index} пуст: клиентов в нём нет")

        for row in rows:
            row["visible_value_ids"] = self._visible(row)

        _check_markers(index, rows, (self._specials[EVT], self._specials[USR]))

        return rows

    def _visible(self, row: dict) -> np.ndarray:
        """
        Значения клиента, которые модели разрешено видеть: маска та
        же, что у apply, но массивами, как у читателя обучения.
        """

        selection = choose(self.group, row, self.masking, self._weights)

        return apply_selection(
            row["client_id"], row, selection, self._specials[MASK], self._specials[UNK]
        )["value_ids"]


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
    "CALENDAR_PER_EVENT",
    "SAMPLE_COLUMNS",
    "InputError",
    "Source",
]
