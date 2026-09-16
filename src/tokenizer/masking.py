from __future__ import annotations

import fnmatch
from dataclasses import dataclass

import numpy as np

from .config import MASK_ID
from .dataset import TokenBatch
from .vocab import N_FIELDS, Vocab


# ============================================================
# ИДЕЯ
# ============================================================
#
# Маскирование это runtime-операция, а не свойство данных.
# Сохранённые датасеты содержат исходные ID; masker работает с
# копией и своим seed, поэтому один и тот же пример на разных
# шагах обучения маскируется по-разному, а один и тот же шаг
# воспроизводится точно.
#
# Маскируются только известные непустые значения полей с
# predictable=true. Специальные токены, [UNK], [MISSING],
# metadata и весь профиль не маскируются никогда: профиль это
# контекст as-of, а не цель.
#
# Заменяется только value_id. Ключи, позиции и время остаются:
# модель обязана знать, ЧТО у неё спрашивают.
# ============================================================


IGNORE_INDEX = -100

MODE_TOKEN = "token"
MODE_KEY = "key"
MODE_EVENT = "event"
MODE_FIELD_BALANCED = "field_balanced"
MODE_COMBINED = "combined"

MODES: tuple[str, ...] = (MODE_TOKEN, MODE_KEY, MODE_EVENT, MODE_FIELD_BALANCED, MODE_COMBINED)

COMBINED_RULE = (
    "три независимые выборки по исходному входу: каждая позиция с token_rate, каждое событие "
    "с event_rate, каждая пара (пример, ключ) с key_rate; итог это объединение, пересечения "
    "считаются один раз. Ожидаемая доля скрытых значений это 1 - (1-token_rate)(1-event_rate)(1-key_rate)"
)


# ------------------------------------------------------------
# СХЕМА РОЗЫГРЫША
# ------------------------------------------------------------
#
# У каждого примера свой поток случайных чисел, засеянный его
# идентичностью (клиент, cutoff). Маска примера одна и та же
# при любом составе и размере batch. Без этого нельзя ни менять
# eval_batch_size, ни собирать validation потоком.
# ------------------------------------------------------------



# ------------------------------------------------------------
# ИСКЛЮЧЁННЫЕ ПОЛЯ
# ------------------------------------------------------------
#
# Задаются паттернами имён ключей ("profile_snapshot__*"),
# а не идентификаторами: конфиг маскирования уезжает
# в checkpoint и восстанавливается там, где словаря нет.
# ------------------------------------------------------------


def resolve_excluded(names, patterns) -> frozenset[str]:
    """
    Имена полей, подходящие под паттерны.

    Паттерн без единого совпадения это ошибка: опечатка в имени
    иначе тихо превратилась бы в «ничего не исключаем», и
    запуск считал бы цели не те, что заявлено.
    """

    names = list(names)

    resolved: set[str] = set()

    for pattern in patterns:

        found = [name for name in names if fnmatch.fnmatchcase(name, pattern)]

        if not found:
            raise ValueError(
                f"паттерн исключения {pattern!r} не совпал ни с одним полем словаря"
            )

        resolved.update(found)

    return frozenset(resolved)


def excluded_field_names(vocab: Vocab, patterns) -> frozenset[str]:
    return resolve_excluded([entry.key for entry in vocab.fields], patterns)


@dataclass(frozen=True)
class MaskingConfig:
    """
    Режимы не смешиваются: mode это одно значение.
    """

    mode: str = MODE_TOKEN
    seed: int = 20240601

    # token: вероятность выбрать eligible-позицию
    token_rate: float = 0.15

    # key: сколько полей маскировать в примере целиком
    keys_per_example: int = 1

    # event: вероятность выбрать событие
    event_rate: float = 0.15

    # field_balanced: доля eligible-позиций batch как бюджет
    balanced_share: float = 0.15

    # combined: вероятность выбрать пару (пример, ключ) целиком
    key_rate: float = 0.10

    # Поля, которые не становятся целью, оставаясь входом.
    # Паттерны имён ключей, а не идентификаторы: конфиг едет
    # в checkpoint и восстанавливается там, где словаря нет.
    exclude_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:

        if self.mode not in MODES:
            raise ValueError(f"неизвестный режим маскирования {self.mode!r}, ожидался один из {MODES}")

        for name in ("token_rate", "event_rate", "balanced_share", "key_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} должен лежать в [0, 1], получено {value}")

        if self.keys_per_example < 1:
            raise ValueError("keys_per_example должен быть положительным")

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "seed": self.seed,
            "token_rate": self.token_rate,
            "keys_per_example": self.keys_per_example,
            "event_rate": self.event_rate,
            "balanced_share": self.balanced_share,
            "key_rate": self.key_rate,
            # Ключ появляется только когда исключения есть.
            **(
                {"exclude_fields": list(self.exclude_fields)}
                if self.exclude_fields
                else {}
            ),
            "modes": list(MODES),
            "ignore_index": IGNORE_INDEX,
            "rule": (
                "маскируются только известные непустые значения полей с predictable=true; "
                "заменяется только value_id, target это исходный value_id, вне масок target = -100"
            ),
            "combined_rule": COMBINED_RULE,
        }

    @property
    def expected_share(self) -> float:
        """
        Ожидаемая доля скрытых значений у combined.

        Стратегии независимы, поэтому доли не складываются:
        0.15 + 0.10 + 0.10 это 0.35, а на деле 0.3115.
        """

        if self.mode != MODE_COMBINED:
            raise ValueError("ожидаемая доля определена только для combined")

        return 1.0 - (1.0 - self.token_rate) * (1.0 - self.event_rate) * (1.0 - self.key_rate)


@dataclass(frozen=True)
class MaskedBatch:
    """
    Копия batch с масками. Исходные массивы не меняются.
    """

    value_ids: np.ndarray
    targets: np.ndarray
    mask: np.ndarray

    profile_value_ids: np.ndarray
    profile_targets: np.ndarray

    n_eligible: int
    n_masked: int
    mode: str

    # Сколько позиций выбрала каждая стратегия ДО объединения и
    # сколько осталось после. У combined суммы не сходятся: одна
    # позиция может быть выбрана дважды, и это видно только здесь.
    selection: dict | None = None

    @property
    def masked_positions(self) -> np.ndarray:
        return np.flatnonzero(self.mask)

    @property
    def masked_fraction(self) -> float:
        return self.n_masked / self.n_eligible if self.n_eligible else 0.0

    def diagnostics(self) -> dict:
        return {
            "mode": self.mode,
            "n_eligible": self.n_eligible,
            "n_masked": self.n_masked,
            "masked_fraction": self.masked_fraction,
            "selection": self.selection,
        }


# ------------------------------------------------------------
# ВИД, ПО КОТОРОМУ ИДЁТ РОЗЫГРЫШ
# ------------------------------------------------------------
#
# Селекторы смотрят не в TokenBatch, а в вид: один пример с
# локальной нумерацией событий. Поэтому маска не зависит ни от
# соседей по batch, ни от его размера.
# ------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    """
    Позиции одного примера, по которым идёт розыгрыш.

    indices  плоские позиции в batch (их и возвращает выбор)
    keys     field_id этих позиций (идентичность ФИЗИЧЕСКОГО
             поля, а не key token: маска не имеет права
             зависеть от режима словаря)
    events   номер события этих позиций внутри примера
    """

    indices: np.ndarray
    keys: np.ndarray
    events: np.ndarray
    n_events: int

class Masker:
    """
    Runtime-маскирование batch'а.
    """

    def __init__(self, vocab: Vocab, config: MaskingConfig = MaskingConfig()):
        self.vocab = vocab
        self.config = config

        # Исключённые поля разрешаются один раз, по этому
        # словарю. Паттерн, не совпавший ни с чем, это ошибка:
        # опечатка иначе тихо стала бы «ничего не исключаем».
        self.excluded_names: frozenset[str] = excluded_field_names(
            vocab, config.exclude_fields
        )

        # Таблица по ПОЛЮ, а не по токену: исключение это
        # свойство поля и одинаково во всех режимах словаря.
        self.excluded_by_field = np.zeros(N_FIELDS + 1, dtype=bool)

        for name in self.excluded_names:
            entry = vocab.field_entry(name)
            if entry is not None:
                self.excluded_by_field[entry.field_id] = True

    # --------------------------------------------------------

    def eligible(self, field_ids: np.ndarray, value_ids: np.ndarray) -> np.ndarray:
        """
        Позиции, которые вообще можно маскировать.

        Значение должно быть настоящим значением словаря: любой
        special ID, включая [UNK] и [MISSING], меньше границы
        first_value_id и не проходит.

        Исключённое поле не проходит тоже. Значение при этом
        остаётся на месте: поле продолжает быть входом, оно
        перестаёт быть целью.
        """

        fields = np.asarray(field_ids, dtype=np.int64)
        values = np.asarray(value_ids, dtype=np.int64)

        return (
            self.vocab.predictable_by_field[fields]
            & (values >= self.vocab.first_value_id)
            & ~self.excluded_by_field[fields]
        )

    def rng_for(self, step: int, identity) -> np.random.Generator:
        """
        Поток примера: seed, шаг и идентичность самого примера.

        Идентичность это (client_id, cutoff). Пара (клиент,
        cutoff) и есть определение примера, поэтому маска
        привязана к нему, а не к его месту в batch.
        """

        return np.random.default_rng(
            [self.config.seed, int(step), int(identity[0]), int(identity[1])]
        )

    # --------------------------------------------------------

    def apply(
        self, batch: TokenBatch, step: int = 0, identities: np.ndarray | None = None
    ) -> MaskedBatch:

        eligible = self.eligible(batch.field_ids, batch.value_ids)

        profile_eligible = self.eligible(batch.profile_field_ids, batch.profile_value_ids)

        if profile_eligible.any():
            raise AssertionError("профиль не должен содержать маскируемых позиций: это контекст, а не цель")

        indices = np.flatnonzero(eligible)

        chosen, selection = self._select_per_example(batch, indices, step, identities)

        value_ids = np.array(batch.value_ids, dtype=np.int32, copy=True)

        targets = np.full(value_ids.size, IGNORE_INDEX, dtype=np.int64)

        mask = np.zeros(value_ids.size, dtype=bool)

        if chosen.size:
            targets[chosen] = value_ids[chosen]
            value_ids[chosen] = MASK_ID
            mask[chosen] = True

        return MaskedBatch(
            value_ids=value_ids,
            targets=targets,
            mask=mask,
            profile_value_ids=np.array(batch.profile_value_ids, dtype=np.int32, copy=True),
            profile_targets=np.full(batch.profile_value_ids.size, IGNORE_INDEX, dtype=np.int64),
            n_eligible=int(indices.size),
            n_masked=int(chosen.size),
            mode=self.config.mode,
            selection=selection,
        )

    # --------------------------------------------------------

    def _select_per_example(
        self,
        batch: TokenBatch,
        indices: np.ndarray,
        step: int,
        identities: np.ndarray | None,
    ) -> tuple[np.ndarray, dict]:
        """
        Тот же выбор, но по одному примеру за раз.

        Позиции примера идут в batch подряд, а события примера
        нумеруются заново с нуля, поэтому селекторам достаётся
        ровно тот же код и то же число розыгрышей, что и на
        отдельном batch из одного этого примера.
        """

        if identities is None:
            raise ValueError(
                "схема example требует идентичности примеров: "
                "маска привязана к паре (client_id, cutoff)"
            )

        identities = np.asarray(identities, dtype=np.int64)

        if identities.shape != (batch.n_examples, 2):
            raise ValueError(
                f"идентичностей {identities.shape}, а примеров {batch.n_examples}"
            )

        example_of_event = np.asarray(batch.example_of_event, dtype=np.int64)

        counts = np.bincount(example_of_event, minlength=batch.n_examples)

        starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)

        example_of_token = np.asarray(batch.example_ids, dtype=np.int64)[indices]
        event_of_token = np.asarray(batch.event_ids, dtype=np.int64)[indices]
        key_of_token = np.asarray(batch.field_ids, dtype=np.int64)[indices]

        parts: list[np.ndarray] = []

        strategies: dict[str, int] = {}
        unique = 0

        for example in range(batch.n_examples):

            inside = example_of_token == example

            view = Selection(
                indices=indices[inside],
                keys=key_of_token[inside],
                events=event_of_token[inside] - int(starts[example]),
                n_events=int(counts[example]),
            )

            chosen, selection = self._select_view(
                view, self.rng_for(step, identities[example])
            )

            parts.append(np.asarray(chosen, dtype=np.int64))

            for label, count in selection.get("strategies", {}).items():
                strategies[label] = strategies.get(label, 0) + int(count)

            unique += int(selection.get("unique", 0))

        chosen = (
            np.sort(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)
        )

        return chosen, {"strategies": strategies, "unique": unique}

    def _select_view(
        self, view: Selection, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict]:

        mode = self.config.mode

        if view.indices.size == 0:
            return view.indices, {"strategies": {mode: 0}, "unique": 0}

        if mode == MODE_COMBINED:
            return self._select_combined(view, rng)

        if mode == MODE_TOKEN:
            chosen = self._select_token(view.indices, rng)
        elif mode == MODE_KEY:
            chosen = self._select_key(view, rng)
        elif mode == MODE_EVENT:
            chosen = self._select_event(view, rng)
        else:
            chosen = self._select_field_balanced(view, rng)

        return chosen, {"strategies": {mode: int(chosen.size)}, "unique": int(chosen.size)}

    # --------------------------------------------------------

    def _select_token(self, indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Каждая eligible-позиция выбирается независимо.
        """

        return indices[rng.random(indices.size) < self.config.token_rate]

    def _select_key(self, view: Selection, rng: np.random.Generator) -> np.ndarray:
        """
        Выбирается поле, маскируются все его доступные значения
        в этом примере.
        """

        keys = view.keys

        if keys.size == 0:
            return np.zeros(0, dtype=np.int64)

        available = np.unique(keys)

        take = min(self.config.keys_per_example, available.size)

        picked = rng.choice(available, size=take, replace=False)

        return np.sort(view.indices[np.isin(keys, picked)])

    def _select_event(self, view: Selection, rng: np.random.Generator) -> np.ndarray:
        """
        Выбирается событие, маскируются все его доступные значения.
        """

        picked = rng.random(view.n_events) < self.config.event_rate

        return view.indices[picked[view.events]]

    def _select_field_balanced(
        self, view: Selection, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Бюджет масок распределяется между полями поровну.

        1. бюджет это доля eligible-позиций batch;
        2. равновероятно выбирается поле среди имеющих доступные позиции;
        3. равновероятно выбирается одна его позиция;
        4. повторяется без повторного выбора позиций, пока бюджет не исчерпан.
        """

        indices = view.indices

        budget = int(round(self.config.balanced_share * indices.size))

        if budget <= 0:
            return np.zeros(0, dtype=np.int64)

        keys = view.keys

        order = np.unique(keys)

        # Перестановка внутри поля один раз: снимать позиции
        # по порядку из неё это то же самое, что выбирать
        # равновероятно без возврата.
        pools = [rng.permutation(indices[keys == key]) for key in order]

        cursor = [0] * len(pools)

        active = list(range(len(pools)))

        chosen: list[int] = []

        while len(chosen) < budget and active:

            slot = int(rng.integers(len(active)))

            pool_index = active[slot]

            chosen.append(int(pools[pool_index][cursor[pool_index]]))

            cursor[pool_index] += 1

            if cursor[pool_index] == pools[pool_index].size:
                active.pop(slot)

        return np.sort(np.asarray(chosen, dtype=np.int64))

    # --------------------------------------------------------

    def _select_combined(
        self, view: Selection, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict]:
        """
        Три независимые выборки по исходному входу и их объединение.

        Все три розыгрыша идут по ещё не замаскированным
        позициям. Последовательное применение трёх masker'ов
        дало бы не то же самое: второй увидел бы [MASK] вместо
        значения, счёл бы позицию недоступной и потерял бы её
        target.

        Вид это один пример, поэтому и событие, и ключ целиком
        лежат внутри него.
        """

        indices = view.indices

        keys = view.keys
        events = view.events

        # 1. Отдельные позиции.
        by_token = indices[rng.random(indices.size) < self.config.token_rate]

        # 2. Целые события.
        picked_events = rng.random(view.n_events) < self.config.event_rate

        by_event = indices[picked_events[events]]

        # 3. Ключ целиком.
        unique_keys, inverse = np.unique(keys, return_inverse=True)

        picked_keys = rng.random(unique_keys.size) < self.config.key_rate

        by_key = indices[picked_keys[np.asarray(inverse).ravel()]]

        chosen = np.union1d(np.union1d(by_token, by_event), by_key)

        selection = {
            "strategies": {
                MODE_TOKEN: int(by_token.size),
                MODE_EVENT: int(by_event.size),
                MODE_KEY: int(by_key.size),
            },
            "unique": int(chosen.size),
            "n_events": int(view.n_events),
            "n_keys": int(unique_keys.size),
        }

        return chosen.astype(np.int64), selection
