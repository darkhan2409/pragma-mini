from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa

from src.preprocessing.artifacts import sha256_file
from src.tokenizer.artifacts import preprocessing_digests, vocab_digests
from src.tokenizer.build import client_runs, iter_client_blocks
from src.tokenizer.config import CONFIG_FILE, DATASET_MANIFEST_FILE
from src.tokenizer.dataset import Events, Example, TokenizedDataset, _events_of, collate
from src.tokenizer.masking import SCHEME_EXAMPLE, Masker, MaskingConfig
from src.tokenizer.vocab import Vocab

from .config import STRUCTURE_EVENT, STRUCTURE_SESSION, ModelConfig
from .history_batching import metadata_from_examples, prepare_history_batch, to_model_inputs
from .mlm_batching import MaskedTargets, build_targets
from .mlm_head import FieldTable
from .sessions import load_session_keys, sidecar_digest


# ============================================================
# ИДЕЯ
# ============================================================
#
# Обучение читает одни и те же истории тысячи раз. Ходить за
# каждым примером в parquet значит платить случайными чтениями
# row group на каждом шаге, поэтому события выбранных клиентов
# читаются ОДИН раз последовательно и живут в памяти.
#
# Пример это префикс ленты клиента (seq < seq_end), поэтому все
# примеры одного клиента это срезы одного массива, а не копии.
#
# Validation фиксируется целиком: клиенты, порядок, состав
# batch, обрезка и маски. Иначе «метрика выросла» означало бы в
# том числе «маски стали проще».
# ============================================================


UNIGRAM_FILE = "unigram_baselines.json"


# ============================================================
# ВЫБОР КЛИЕНТОВ
# ============================================================


def select_clients(
    examples: pa.Table,
    max_clients: int | None = None,
    clients: Iterable[int] | None = None,
) -> list[int]:
    """
    Клиенты сплита по возрастанию ID.

    clients это белый список на весь набор данных: распределение
    по сплитам берётся из существующего разбиения, а не задаётся
    заново счётчиком на каждый сплит.
    """

    unique = sorted({int(value) for value in examples.column("client_id").to_pylist()})

    if clients is not None:
        wanted = {int(value) for value in clients}
        unique = [value for value in unique if value in wanted]

    return unique if max_clients is None else unique[: int(max_clients)]


CUTOFFS_ALL = "all"
CUTOFFS_LAST = "last"

CUTOFF_POLICIES = (CUTOFFS_ALL, CUTOFFS_LAST)


def last_row_per_client(rows: list[dict]) -> list[int]:
    """
    По одному примеру на клиента: самый поздний cutoff.

    Строки уже упорядочены по (client_id, cutoff), поэтому
    последнее вхождение клиента и есть его последний срез.
    """

    picked: dict[int, int] = {}

    for index, row in enumerate(rows):
        picked[int(row["client_id"])] = index

    return [picked[client_id] for client_id in sorted(picked)]


def rows_for_clients(examples: pa.Table, client_ids) -> np.ndarray:
    """
    Строки examples.parquet выбранных клиентов в порядке (client_id, cutoff).
    """

    wanted = np.array(sorted({int(value) for value in client_ids}), dtype=np.int64)

    client_id = examples.column("client_id").to_numpy().astype(np.int64)
    cutoff = examples.column("cutoff").to_numpy().astype("datetime64[us]").astype(np.int64)

    rows = np.flatnonzero(np.isin(client_id, wanted))

    return rows[np.lexsort((cutoff[rows], client_id[rows]))]


@dataclass(frozen=True)
class SessionExample(Example):
    """
    Пример с ключами сессий его событий.

    Ключи едут вместе с примером, а не отдельным аргументом:
    иначе их пришлось бы протаскивать через каждый вызов
    store.examples(...) -> prepare(...). collate и
    metadata_from_examples подкласса не замечают.
    """

    session_keys: np.ndarray | None = None


def session_keys_from_examples(examples) -> np.ndarray | None:
    """
    Ключи сессий batch в том же порядке, в каком collate
    склеивает события.
    """

    keyed = [isinstance(item, SessionExample) and item.session_keys is not None for item in examples]

    if not any(keyed):
        return None

    if not all(keyed):
        raise ValueError(
            "в batch смешаны примеры с ключами сессий и без них: "
            "структура истории должна быть одна на весь batch"
        )

    for item in examples:
        if item.session_keys.size != item.events.n_events:
            raise ValueError(
                f"клиент {item.client_id}: ключей {item.session_keys.size}, "
                f"а событий {item.events.n_events}"
            )

    return np.concatenate([np.asarray(item.session_keys, dtype=np.int64) for item in examples])


def _prefix(events: Events, seq_end: int) -> Events:
    """
    История примера это первые seq_end событий клиента.
    """

    if seq_end > events.n_events:
        raise ValueError(f"пример требует {seq_end} событий, а у клиента их {events.n_events}")

    stop = int(events.offsets[seq_end])

    return Events(
        key_ids=events.key_ids[:stop],
        value_ids=events.value_ids[:stop],
        positions=events.positions[:stop],
        offsets=events.offsets[: seq_end + 1],
        event_type=events.event_type[:seq_end],
        ts=events.ts[:seq_end],
        seq=events.seq[:seq_end],
    )


# ============================================================
# ХРАНИЛИЩЕ
# ============================================================


class ClientStore:
    """
    Ленты выбранных клиентов в памяти плюс их примеры.
    """

    def __init__(
        self,
        root: Path,
        dataset: str,
        vocab_dir: Path,
        max_clients: int | None = None,
        clients: Iterable[int] | None = None,
        sessions: bool = False,
        shared: "ClientStore | None" = None,
        cutoffs: str = CUTOFFS_ALL,
    ):

        self.dataset = dataset

        self.data = TokenizedDataset(root, dataset, vocab_dir=vocab_dir)

        self.client_ids: list[int] = select_clients(self.data.examples, max_clients, clients)

        if not self.client_ids:
            raise ValueError(f"{dataset}: не нашлось ни одного клиента")

        self.row_index = rows_for_clients(self.data.examples, self.client_ids)

        self.rows = self.data.examples.take(pa.array(self.row_index)).to_pylist()

        self.cutoffs = cutoffs

        if cutoffs not in CUTOFF_POLICIES:
            raise ValueError(f"cutoffs должен быть одним из {CUTOFF_POLICIES}, получено {cutoffs!r}")

        if cutoffs == CUTOFFS_LAST:
            keep = last_row_per_client(self.rows)
            self.row_index = self.row_index[keep]
            self.rows = [self.rows[index] for index in keep]

        # Ленты можно взять по ссылке у другого набора: val_time
        # и test_time это те же train-клиенты и тот же файл
        # событий, и вторая копия в памяти была бы платой ни за
        # что. Берётся только если КАЖДЫЙ нужный клиент там уже
        # есть и группа клиентов та же: иначе набор молча стал
        # бы другим.
        self.shared_events = bool(
            shared is not None
            and shared.data.group == self.data.group
            and set(self.client_ids) <= set(shared.events)
            and (not sessions or shared.session_keys is not None)
        )

        if self.shared_events:
            self.events = {client: shared.events[client] for client in self.client_ids}
            self.session_keys = (
                {client: shared.session_keys[client] for client in self.client_ids}
                if sessions
                else None
            )
            return

        self.events = self._load_events(set(self.client_ids))

        self.session_keys = (
            self._load_session_keys(set(self.client_ids)) if sessions else None
        )

    # --------------------------------------------------------

    def _load_events(self, wanted: set[int]) -> dict[int, Events]:
        """
        Один последовательный проход до последнего нужного клиента.
        """

        limit = max(wanted)

        found: dict[int, Events] = {}

        for block in iter_client_blocks(self.data.events_path):

            column = block.column("client_id").to_numpy()

            if column.size == 0:
                continue

            if int(column[0]) > limit:
                break

            for value, lo, hi in client_runs(column):
                if value in wanted:
                    found[value] = _events_of(block, lo, hi)

        missing = sorted(wanted - set(found))

        if missing:
            raise ValueError(f"{self.dataset}: нет событий клиентов {missing[:5]}")

        return found

    # --------------------------------------------------------

    def _load_session_keys(self, wanted: set[int]) -> dict[int, np.ndarray]:
        """
        Ключи сессий с поэлементной сверкой по seq.

        Сравнения длин мало: у клиента с тем же числом событий,
        но другой лентой, длина совпала бы, а история была бы
        чужой.
        """

        group = self.data.group

        keys = load_session_keys(Path(self.data.root), str(group), wanted)

        for client_id, events in self.events.items():

            values = keys[client_id]

            if values.size != events.n_events:
                raise ValueError(
                    f"клиент {client_id}: ключей сессий {values.size}, "
                    f"а событий {events.n_events}"
                )

            if not np.array_equal(events.seq, np.arange(events.n_events, dtype=np.int64)):
                raise ValueError(
                    f"клиент {client_id}: seq токенизированных событий не плотный от нуля"
                )

        return keys

    # --------------------------------------------------------

    @staticmethod
    def from_rows(
        root: Path,
        group: str,
        vocab_dir: Path,
        rows: list[dict],
        sessions: bool = False,
    ) -> "ClientStore":
        """
        Хранилище на готовых строках примеров.

        Нужно там, где примеров нет в разбиении: downstream
        строит по одному примеру на клиента на своём cutoff, и
        каталога под них не заводит. Строки обязаны нести те же
        поля, что и examples.parquet: client_id, cutoff,
        dataset, client_group, seq_end, snapshot_ts.
        """

        store = ClientStore.__new__(ClientStore)

        store.dataset = f"{group}:rows"

        store.data = TokenizedDataset(root, None, vocab_dir=vocab_dir, group=group)

        store.rows = [dict(row) for row in rows]

        if not store.rows:
            raise ValueError(f"{group}: пустой список строк примеров")

        store.client_ids = sorted({int(row["client_id"]) for row in store.rows})

        # Порядок тот же, что у обычного набора: (client_id, cutoff).
        order = sorted(
            range(len(store.rows)),
            key=lambda index: (
                int(store.rows[index]["client_id"]),
                store.rows[index]["cutoff"],
            ),
        )

        store.rows = [store.rows[index] for index in order]

        store.row_index = np.arange(len(store.rows), dtype=np.int64)

        store.shared_events = False

        store.events = store._load_events(set(store.client_ids))

        store.session_keys = (
            store._load_session_keys(set(store.client_ids)) if sessions else None
        )

        return store

    # --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def n_events(self) -> int:
        return sum(events.n_events for events in self.events.values())

    def example(self, index: int) -> Example:

        row = self.rows[int(index)]

        client_id = int(row["client_id"])

        seq_end = int(row["seq_end"])

        common = dict(
            client_id=client_id,
            cutoff=row["cutoff"],
            dataset=row["dataset"],
            client_group=row["client_group"],
            seq_end=seq_end,
            snapshot_ts=row["snapshot_ts"],
            profile=self.data._profile_record(client_id, row["snapshot_ts"]),
            events=_prefix(self.events[client_id], seq_end),
        )

        if self.session_keys is None:
            return Example(**common)

        # Ключи режутся тем же префиксом, что и события.
        return SessionExample(**common, session_keys=self.session_keys[client_id][:seq_end])

    def examples(self, indices) -> list[Example]:
        return [self.example(int(index)) for index in indices]

    def summary(self) -> dict:
        return {
            "dataset": self.dataset,
            "clients": len(self.client_ids),
            "examples": len(self.rows),
            "events_in_memory": self.n_events,
            "shared_events": self.shared_events,
            "client_id_min": int(min(self.client_ids)),
            "client_id_max": int(max(self.client_ids)),
        }


# ============================================================
# ПОРЯДОК ОБУЧЕНИЯ
# ============================================================


class EpochSampler:
    """
    Перестановка на эпоху с сохраняемым состоянием.
    """

    def __init__(self, n_examples: int, batch_size: int, seed: int):

        if n_examples < 1:
            raise ValueError("нечего сэмплировать: примеров нет")

        if batch_size < 1:
            raise ValueError("batch_size должен быть положительным")

        self.n_examples = int(n_examples)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

        self.epoch = 0
        self.position = 0

        self.order = self._permutation(0)

    def _permutation(self, epoch: int) -> np.ndarray:
        return np.random.default_rng([self.seed, int(epoch)]).permutation(self.n_examples)

    # --------------------------------------------------------

    def next_batch(self) -> np.ndarray:
        """
        Следующий batch; на границе эпохи порядок перемешивается заново.
        """

        if self.position >= self.n_examples:
            self.epoch += 1
            self.position = 0
            self.order = self._permutation(self.epoch)

        stop = min(self.position + self.batch_size, self.n_examples)

        batch = self.order[self.position : stop]

        self.position = stop

        return batch

    # --------------------------------------------------------

    def state(self) -> dict:
        return {
            "n_examples": self.n_examples,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position,
            "order": self.order.tolist(),
        }

    def load_state(self, state: dict) -> None:

        if int(state["n_examples"]) != self.n_examples:
            raise ValueError(
                f"sampler сохранён на {state['n_examples']} примеров, сейчас их {self.n_examples}"
            )

        self.batch_size = int(state["batch_size"])
        self.seed = int(state["seed"])
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
        self.order = np.asarray(state["order"], dtype=np.int64)


# ============================================================
# ФИКСИРОВАННАЯ VALIDATION
# ============================================================


# ============================================================
# ПОТОЧНЫЙ ИСТОЧНИК
# ============================================================
#
# Хранить подготовленные batch'и всего набора это гигабайты на
# полных историях. Пересобирать их на каждую оценку дешевле по
# памяти, но воспроизводимо только при схеме example: там маска
# примера не зависит ни от соседей, ни от номера batch, и
# второй проход даёт ровно те же цели.
#
# При схеме batch поток запрещён: маска зависела бы от порядка,
# а digest перестал бы что-либо гарантировать.
# ============================================================


@dataclass(frozen=True)
class StreamSource:

    store: "ClientStore"
    masker: Masker
    table: FieldTable
    max_events: int | None
    batch_size: int
    structure: str
    chunks: tuple[range, ...]

    def batches(self):
        """
        Те же batch'и, что были посчитаны при сборке.
        """

        for chunk in self.chunks:
            yield prepared_batch(
                self.store, chunk, self.masker, self.table, self.max_events, self.structure
            )


def prepared_batch(
    store: "ClientStore",
    chunk,
    masker: Masker,
    table: FieldTable,
    max_events: int | None,
    structure: str,
    step: int = 0,
) -> tuple[object, MaskedTargets]:
    """
    Один batch набора: примеры, маски, цели.
    """

    examples = store.examples(chunk)

    keys = session_keys_from_examples(examples)

    if structure == STRUCTURE_SESSION and keys is None:
        raise ValueError(
            "структура session требует ключей сессий: "
            "ClientStore должен быть открыт с sessions=True"
        )

    history = prepare_history_batch(
        collate(examples),
        metadata_from_examples(examples),
        max_events,
        masker=masker,
        step=step,
        session_keys=keys if structure == STRUCTURE_SESSION else None,
        structure=structure,
    )

    return history, build_targets(history, table)


@dataclass(frozen=True)
class FixedSplit:
    """
    Набор, который у всех checkpoint один и тот же.
    """

    name: str
    clients: tuple[int, ...]
    row_index: np.ndarray

    # Клиент каждого примера в порядке нумерации набора: единица
    # пересэмплирования это клиент, а не месячный срез.
    client_of_example: np.ndarray

    # Хранится подготовленный, но ещё не выровненный batch:
    # padded int64 на полных историях весит около гигабайта,
    # а плоские int32 вчетверо меньше. Маски и цели при этом
    # зафиксированы раз и навсегда.
    prepared: tuple[tuple[object, MaskedTargets], ...] | None

    # Заполнен вместо prepared при потоковой сборке.
    source: StreamSource | None

    n_targets: int
    n_eligible: int
    n_degenerate: int
    n_masked: int
    selected_by: dict

    original_lengths: np.ndarray
    used_lengths: np.ndarray
    truncated: np.ndarray

    settings: dict
    digest: str

    @property
    def n_examples(self) -> int:
        return int(self.row_index.size)

    @property
    def n_clients(self) -> int:
        return len(self.clients)

    @property
    def n_batches(self) -> int:
        return len(self.prepared) if self.prepared is not None else len(self.source.chunks)

    @property
    def streamed(self) -> bool:
        return self.prepared is None

    def iter_batches(self, model_config: ModelConfig):
        """
        Вход модели строится на каждую оценку заново.

        pad_flat векторный и стоит секунды, а держать выровненные
        тензоры всех batch'ей в памяти на полных историях нельзя.

        При потоковой сборке заново собирается и сам batch. Это
        те же самые цели: схема example привязывает маску к
        примеру, а не к его месту в наборе.
        """

        stored = self.prepared if self.prepared is not None else self.source.batches()

        for history, targets in stored:
            yield to_model_inputs(history, model_config), targets

    def description(self) -> dict:
        return {
            "name": self.name,
            "clients": list(self.clients),
            "n_clients": self.n_clients,
            "n_examples": self.n_examples,
            "n_batches": self.n_batches,
            "row_index_sha256": hashlib.sha256(
                np.asarray(self.row_index, dtype=np.int64).tobytes()
            ).hexdigest(),
            "n_targets": self.n_targets,
            "n_eligible": self.n_eligible,
            "n_degenerate": self.n_degenerate,
            "n_masked": self.n_masked,
            "masked_fraction": self.n_masked / self.n_eligible if self.n_eligible else 0.0,
            "selected_by": dict(self.selected_by),
            "settings": dict(self.settings),
            "targets_sha256": self.digest,
        }

    @staticmethod
    def build(
        name: str,
        store: ClientStore,
        vocab: Vocab,
        table: FieldTable,
        model_config: ModelConfig,
        masking: MaskingConfig,
        max_events: int | None,
        batch_size: int,
        structure: str = STRUCTURE_EVENT,
        stream: bool = False,
    ) -> "FixedSplit":
        """
        Собирает batch'и один раз: маски, обрезка и раскладка
        больше не меняются.
        """

        if stream and masking.scheme != SCHEME_EXAMPLE:
            raise ValueError(
                "поток требует схемы масок example: при схеме batch маска "
                "зависит от номера batch, и второй проход дал бы другие цели"
            )

        masker = Masker(vocab, masking)

        prepared: list[tuple[object, MaskedTargets]] = []

        chunks: list[range] = []

        original: list[np.ndarray] = []
        used: list[np.ndarray] = []
        truncated: list[np.ndarray] = []

        digest = hashlib.sha256()

        n_targets = n_eligible = n_degenerate = n_masked = 0

        selected_by: dict[str, int] = {}

        for index, start in enumerate(range(0, len(store), batch_size)):

            chunk = range(start, min(start + batch_size, len(store)))

            chunks.append(chunk)

            # Схема example не смотрит на номер batch, поэтому
            # шаг фиксируется нулём: иначе поток пришлось бы
            # заново нумеровать теми же индексами, и это была бы
            # воспроизводимость по договорённости, а не по сути.
            history, targets = prepared_batch(
                store,
                chunk,
                masker,
                table,
                max_events,
                structure,
                step=0 if masking.scheme == SCHEME_EXAMPLE else index,
            )

            if not stream:
                prepared.append((history, targets))

            original.append(history.info.original_history_length)
            used.append(history.info.used_history_length)
            truncated.append(history.info.truncated)

            n_targets += targets.n
            n_degenerate += targets.n_degenerate

            diagnostics = history.masking or {}

            # Доступных позиций всегда больше скрытых: eligible
            # считает masker, а n_masked это его выбор.
            n_eligible += int(diagnostics.get("n_eligible", 0))
            n_masked += int(diagnostics.get("n_masked", 0))

            for label, count in (diagnostics.get("selection") or {}).get("strategies", {}).items():
                selected_by[label] = selected_by.get(label, 0) + int(count)

            digest.update(str(index).encode("utf-8"))
            digest.update(targets.digest().encode("utf-8"))

        return FixedSplit(
            name=name,
            clients=tuple(store.client_ids),
            row_index=np.asarray(store.row_index, dtype=np.int64),
            client_of_example=np.array(
                [int(row["client_id"]) for row in store.rows], dtype=np.int64
            ),
            prepared=None if stream else tuple(prepared),
            source=(
                StreamSource(
                    store=store,
                    masker=masker,
                    table=table,
                    max_events=max_events,
                    batch_size=int(batch_size),
                    structure=structure,
                    chunks=tuple(chunks),
                )
                if stream
                else None
            ),
            n_targets=n_targets,
            n_eligible=n_eligible,
            n_degenerate=n_degenerate,
            n_masked=n_masked,
            selected_by=dict(sorted(selected_by.items())),
            original_lengths=np.concatenate(original) if original else np.zeros(0, dtype=np.int64),
            used_lengths=np.concatenate(used) if used else np.zeros(0, dtype=np.int64),
            truncated=np.concatenate(truncated) if truncated else np.zeros(0, dtype=bool),
            settings={
                "max_events_per_history": None if max_events is None else int(max_events),
                "batch_size": int(batch_size),
                "masking": masking.as_dict(),
                "structure": structure,
                "stream": bool(stream),
            },
            digest=digest.hexdigest(),
        )


# ============================================================
# ОТПЕЧАТКИ ARTIFACTS
# ============================================================


def artifact_hashes(root: Path, vocab_dir: Path, artifacts_dir: Path) -> dict:
    """
    Всё, от чего зависят ID, кандидаты и baseline.

    unigram_baselines.json не входит в отпечаток tokenizer,
    поэтому его хэш фиксируется здесь: обучение сравнивается с
    ним, и подмена baseline меняла бы смысл NCE.

    Sidecar сессий тоже: он задаёт СТРУКТУРУ истории, и его
    подмена изменила бы модель, не тронув ни один токен.
    Отсутствие sidecar это законное состояние прежней
    структуры, поэтому None, а не ошибка.
    """

    return {
        "tokenizer_config": sha256_file(Path(vocab_dir) / CONFIG_FILE),
        "vocab": vocab_digests(vocab_dir),
        "preprocessing": preprocessing_digests(artifacts_dir),
        "unigram_baselines": sha256_file(Path(artifacts_dir) / UNIGRAM_FILE),
        "tokenized_manifest": sha256_file(Path(root) / DATASET_MANIFEST_FILE),
        "sessions": sidecar_digest(root),
    }
