from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.embedding.inputs import CALENDAR_PER_EVENT
from src.embedding.layer import InputEmbedding
from src.embedding.settings import WEIGHTS_FILE as EMBEDDING_WEIGHTS
from src.embedding.settings import embeddings_dir
from src.event.encoder import EventEncoder
from src.event.settings import WEIGHTS_FILE as EVENT_WEIGHTS
from src.event.settings import EventConfig, events_dir
from src.history.encoder import HistoryEncoder
from src.history.settings import WEIGHTS_FILE as HISTORY_WEIGHTS
from src.history.settings import HistoryConfig, history_dir
from src.profile.encoder import ProfileEncoder
from src.profile.settings import WEIGHTS_FILE as PROFILE_WEIGHTS
from src.profile.settings import ProfileConfig, profiles_dir
from src.tokenization.specials import EVT, USR, load_special_tokens

from .inputs import IGNORE, Client
from .varlen import VarlenLayout, assemble


# ============================================================
# МОДЕЛЬ ЦЕЛИКОМ И ГОЛОВА
# ============================================================
#
# Сквозной проход, в котором градиент доходит от потерь до общей
# таблицы эмбеддингов:
#
#   InputEmbedding -> Event Encoder -> Profile Encoder ->
#   History Encoder -> MLM
#
# Единица прохода — micro-batch: несколько клиентов, собранных
# pack в ПЛОСКИЕ массивы без заполнителя, с границами в
# cu_seqlens (varlen.py). Модель вызывается ОДИН раз на весь
# micro-batch. Внутри три уровня сегментов, и каждый считается
# по корзинам близкой длины:
#
#   события    — сегмент = токены одного события, энкодер события;
#   анкеты     — сегмент = токены анкеты клиента, энкодер анкеты;
#   истории    — сегмент = [анкета, события клиента], энкодер
#                истории: слот анкеты первым, как [USR].
#
# Прямоугольник с заполнителем есть только внутри корзины и
# только до её наибольшей длины: длинный клиент не растягивает
# остальных. Это плоское представление с запасным путём через
# корзины, а не внимание без заполнителя.
#
# Голова получает на каждую размеченную позицию три вектора:
#
#   1) контекстный вектор самого токена после энкодера события;
#   2) вектор ЕГО события после энкодера истории;
#   3) итоговый вектор ЕГО клиента после энкодера истории.
#
# Они склеиваются в 3d, проходят одну линейную проекцию 3d -> d,
# и логиты берутся скалярным произведением с ТОЙ ЖЕ таблицей
# эмбеддингов. Отдельной выходной таблицы нет: градиент течёт в
# одни и те же веса и со стороны входа, и со стороны выхода.
#
# Потери — среднее по ВСЕМ целям micro-batch: цель одного клиента
# весит столько же, сколько цель другого.
#
# Внутри forward нет ни detach, ни NumPy, ни no_grad. Перевод
# клиентов в тензоры и раскладка по корзинам сделаны в pack, до
# входа в граф.
# ============================================================


@dataclass(frozen=True)
class PackedBatch:
    """
    Micro-batch из B клиентов плоскими массивами. Всё, что видит
    модель.

    Заполнителя здесь нет вовсе: T — сумма токенов событий всех
    клиентов, E — сумма их событий, P — сумма токенов их анкет.

    labels лежат рядом, но в саму модель не подаются: они нужны
    только чтобы выбрать позиции и посчитать потери.
    """

    clients: int

    # --- токены событий, [T]; сегменты — события ---
    key_ids: torch.Tensor
    value_ids: torch.Tensor
    positions: torch.Tensor
    labels: torch.Tensor
    events: VarlenLayout          # cu_seqlens_event [E + 1]
    event_of_token: torch.Tensor  # [T]

    # --- события, [E]; сегменты — истории клиентов ---
    event_time_log: torch.Tensor
    calendar: torch.Tensor        # [E, 6]
    user_of_event: torch.Tensor   # [E]

    # --- анкета, [P]; сегменты — анкеты клиентов ---
    profile_key_ids: torch.Tensor
    profile_value_ids: torch.Tensor
    profile_positions: torch.Tensor
    profiles: VarlenLayout        # cu_seqlens_profile [B + 1]

    # --- истории, [B + E]: у клиента слот анкеты и его события ---
    history: VarlenLayout         # сегмент клиента длиной n_events + 1
    history_profile_slot: torch.Tensor  # [B]
    history_event_slot: torch.Tensor    # [E]
    history_positions: torch.Tensor     # [B + E]: 0 у анкеты, event_time_log у события

    # --- цели, [M], по возрастанию плоского номера токена ---
    target_token: torch.Tensor    # плоский номер токена
    target_event: torch.Tensor    # глобальный номер события
    target_client: torch.Tensor   # номер клиента в micro-batch
    target_place: torch.Tensor    # номер токена внутри клиента
    target_local: torch.Tensor    # номер события внутри клиента
    target_inside: torch.Tensor   # позиция внутри события
    target_bucket: torch.Tensor   # корзина его события
    target_row: torch.Tensor      # строка его события в корзине


@dataclass(frozen=True)
class Predicted:
    """
    Что вернул проход по micro-batch.

    place, event и client называют каждую цель: позицию токена и
    событие внутри клиента и номер клиента в micro-batch.
    """

    logits: torch.Tensor    # [M, словарь]
    targets: torch.Tensor   # [M]
    loss: torch.Tensor      # скаляр, связанный с графом: среднее по M целям
    place: torch.Tensor     # [M] номер токена у клиента
    event: torch.Tensor     # [M] номер его события у клиента
    client: torch.Tensor    # [M] номер клиента в micro-batch

    @property
    def count(self) -> int:
        return int(self.targets.numel())


def pack(clients: list[Client], device: torch.device) -> PackedBatch:
    """
    Клиенты в один плоский micro-batch. Делается ДО графа значений.

    Массивы клиентов идут подряд, без заполнителя. События клиента
    обязаны лежать подряд и покрывать все его токены: только тогда
    конкатенация токенов клиентов совпадает с конкатенацией их
    событий, и границы событий однозначны.
    """

    for client in clients:

        expected = np.concatenate(
            [[0], np.cumsum(client.event_lengths)[:-1]]
        ) if client.n_events else client.event_starts

        if (
            not np.array_equal(client.event_starts, expected)
            or int(client.event_lengths.sum()) != client.n_tokens
        ):
            raise ValueError(
                f"{client.client_id}: события не лежат подряд или не покрывают "
                "все токены клиента"
            )

    def join(name: str, dtype) -> np.ndarray:
        return np.concatenate([getattr(client, name) for client in clients]).astype(dtype)

    tokens_per_client = np.array([client.n_tokens for client in clients], dtype=np.int64)
    events_per_client = np.array([client.n_events for client in clients], dtype=np.int64)
    profile_per_client = np.array([client.profile_n_tokens for client in clients], dtype=np.int64)

    size = len(clients)
    total_events = int(events_per_client.sum())

    event_lengths = join("event_lengths", np.int64)
    labels = join("labels", np.int64)
    event_time_log = join("event_time_log", np.float32)

    event_of_token = np.repeat(np.arange(total_events, dtype=np.int64), event_lengths)
    user_of_event = np.repeat(np.arange(size, dtype=np.int64), events_per_client)

    events = VarlenLayout.build(event_lengths, device, "события")
    profiles = VarlenLayout.build(profile_per_client, device, "анкеты")
    history = VarlenLayout.build(events_per_client + 1, device, "истории")

    # Слоты истории: у клиента c сначала анкета, затем его события.
    # До слотов клиента c лежат c анкет и все события прежних
    # клиентов.
    first_event = np.concatenate([[0], np.cumsum(events_per_client)[:-1]]).astype(np.int64)
    history_profile_slot = first_event + np.arange(size, dtype=np.int64)
    history_event_slot = np.arange(total_events, dtype=np.int64) + user_of_event + 1

    history_positions = np.zeros(size + total_events, dtype=np.float32)
    history_positions[history_event_slot] = event_time_log

    # Цели и их владельцы: токен -> событие -> клиент.
    first_token = np.concatenate([[0], np.cumsum(tokens_per_client)[:-1]]).astype(np.int64)
    event_start = np.concatenate([[0], np.cumsum(event_lengths)[:-1]]).astype(np.int64)

    target_token = np.nonzero(labels != IGNORE)[0]
    target_event = event_of_token[target_token]
    target_client = user_of_event[target_event]

    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, device=device)

    return PackedBatch(
        clients=size,
        key_ids=tensor(join("key_ids", np.int64)),
        value_ids=tensor(join("value_ids", np.int64)),
        positions=tensor(join("positions", np.int64)),
        labels=tensor(labels),
        events=events,
        event_of_token=tensor(event_of_token),
        event_time_log=tensor(event_time_log),
        calendar=tensor(
            np.concatenate([client.calendar for client in clients]).astype(np.float32)
            .reshape(-1, CALENDAR_PER_EVENT)
        ),
        user_of_event=tensor(user_of_event),
        profile_key_ids=tensor(join("profile_key_ids", np.int64)),
        profile_value_ids=tensor(join("profile_value_ids", np.int64)),
        profile_positions=tensor(join("profile_positions", np.int64)),
        profiles=profiles,
        history=history,
        history_profile_slot=tensor(history_profile_slot),
        history_event_slot=tensor(history_event_slot),
        history_positions=tensor(history_positions),
        target_token=tensor(target_token),
        target_event=tensor(target_event),
        target_client=tensor(target_client),
        target_place=tensor(target_token - first_token[target_client]),
        target_local=tensor(target_event - first_event[target_client]),
        target_inside=tensor(target_token - event_start[target_event]),
        target_bucket=tensor(events.bucket_of[target_event]),
        target_row=tensor(events.row_of[target_event]),
    )


class Mlm(nn.Module):
    """
    Голова: три вектора в один, дальше связанные логиты.

    Ни нормировки, ни активации, ни dropout — ровно одна линейная
    проекция, как в эталоне.
    """

    def __init__(self, dim: int, seed: int):

        super().__init__()

        self.dim = int(dim)

        with _seeded(seed):
            self.proj = nn.Linear(3 * self.dim, self.dim)

    def forward(
        self,
        token: torch.Tensor,
        event: torch.Tensor,
        client: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        [M, d] x 3 -> [M, словарь].
        """

        context = torch.cat([token, event, client], dim=-1)

        return self.proj(context) @ weight.t()


def mlm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float,
) -> torch.Tensor:
    """
    Кросс-энтропия по размеченным позициям.

    Ноль на пустом наборе возвращается СВЯЗАННЫМ С ГРАФОМ:
    torch.tensor(0.0) оборвал бы цепочку, и backward на клиенте
    без целей упал бы. Приём взят из эталона дословно.
    """

    if targets.numel() == 0 or int((targets != IGNORE).sum()) == 0:
        return logits.sum() * 0.0

    return F.cross_entropy(
        logits, targets, ignore_index=IGNORE, label_smoothing=smoothing
    )


class Model(nn.Module):
    """
    Четыре энкодера и голова, собранные в один проход.
    """

    def __init__(
        self,
        embedding: InputEmbedding,
        event: EventEncoder,
        profile: ProfileEncoder,
        history: HistoryEncoder,
        head: Mlm,
        events_per_chunk: int = 512,
        label_smoothing: float = 0.1,
    ):

        super().__init__()

        self.embedding = embedding
        self.event = event
        self.profile = profile
        self.history = history
        self.head = head

        self.events_per_chunk = int(events_per_chunk)
        self.label_smoothing = float(label_smoothing)

    def forward(self, data: PackedBatch) -> Predicted:
        """
        Micro-batch от токенов до потерь, одним проходом.
        """

        dated, token_vectors = self._events(data)

        profile = self._profiles(data)

        client_vectors, event_vectors = self._history(data, profile, dated)

        if token_vectors is None:
            # Целей нет. Контекст пуст, но граф обязан остаться
            # связным, иначе backward на таком batch оборвётся.
            token_vectors = client_vectors[:0]

        logits = self.head(
            token_vectors,
            event_vectors[data.target_event],
            client_vectors[data.target_client],
            self.embedding.weight,
        )

        targets = data.labels[data.target_token]

        return Predicted(
            logits=logits,
            targets=targets,
            loss=mlm_loss(logits, targets, self.label_smoothing),
            place=data.target_place,
            event=data.target_local,
            client=data.target_client,
        )

    def _events(self, data: PackedBatch) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Энкодер события по корзинам длины.

        Каждое событие — свой сегмент: внимание не выходит за его
        границы. Внутри корзины строки идут порциями по
        events_per_chunk, чтобы память ограничивалась порцией.

        Из порции сразу берутся векторы только целевых токенов:
        держать контекст всех токенов незачем. Возвращаются
        векторы событий [E, d] в исходном порядке и векторы целей
        [M, d] в порядке целей (None, если целей нет).
        """

        dated_parts: list[torch.Tensor] = []
        dated_index: list[torch.Tensor] = []

        token_parts: list[torch.Tensor] = []
        token_index: list[torch.Tensor] = []

        for number, bucket in enumerate(data.events.buckets):

            for first in range(0, bucket.size, self.events_per_chunk):

                last = min(first + self.events_per_chunk, bucket.size)

                index = bucket.index[first:last]
                mask = bucket.mask[first:last]
                segments = bucket.segments[first:last]

                piece = self.event(
                    self.embedding.embed(
                        data.key_ids[index],
                        data.value_ids[index],
                        data.positions[index],
                        mask,
                    ),
                    ~mask,
                    data.calendar[segments],
                )

                dated_parts.append(piece.dated)
                dated_index.append(segments)

                chosen = (
                    (data.target_bucket == number)
                    & (data.target_row >= first)
                    & (data.target_row < last)
                )

                if bool(chosen.any()):

                    ids = torch.nonzero(chosen, as_tuple=True)[0]

                    token_parts.append(
                        piece.tokens[data.target_row[ids] - first, data.target_inside[ids]]
                    )
                    token_index.append(ids)

        dated = assemble(
            dated_parts, dated_index, data.events.segments, self.embedding.weight[:0]
        )

        if not token_parts:
            return dated, None

        return dated, assemble(
            token_parts, token_index, int(data.target_token.numel()), self.embedding.weight[:0]
        )

    def _profiles(self, data: PackedBatch) -> torch.Tensor:
        """
        Энкодер анкеты по корзинам длины: [B, d] в порядке клиентов.
        """

        parts: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []

        for bucket in data.profiles.buckets:

            parts.append(
                self.profile(
                    self.embedding.embed(
                        data.profile_key_ids[bucket.index],
                        data.profile_value_ids[bucket.index],
                        data.profile_positions[bucket.index],
                        bucket.mask,
                    ),
                    ~bucket.mask,
                )
            )
            indices.append(bucket.segments)

        return assemble(parts, indices, data.clients, self.embedding.weight[:0])

    def _history(
        self,
        data: PackedBatch,
        profile: torch.Tensor,
        dated: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Энкодер истории по корзинам длины.

        Плоская история [B + E, d]: у каждого клиента первым идёт
        слот анкеты, затем его события. Сегмент клиента — только
        его строки, поэтому внимание одного клиента не видит
        другого. Позиция анкеты — ноль, события — event_time_log.

        Возвращает векторы клиентов [B, d] и событий [E, d].
        """

        width = data.clients + data.events.segments

        # Вне графа только нулевой холст: index_copy возвращает
        # новый тензор, и градиент идёт к анкетам и событиям.
        flat = (
            profile.new_zeros(width, profile.shape[-1])
            .index_copy(0, data.history_profile_slot, profile)
            .index_copy(0, data.history_event_slot, dated)
        )

        parts: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []

        for bucket in data.history.buckets:

            # Позиция хвоста — ноль: маска и так закрывает его, а
            # настоящие позиции остаются ровно своими.
            positions = torch.where(
                bucket.mask, data.history_positions[bucket.index], 0.0
            )

            out = self.history(flat[bucket.index], positions, bucket.mask)

            parts.append(out[bucket.mask])
            indices.append(bucket.index[bucket.mask])

        out = assemble(parts, indices, width, flat[:0])

        return out[data.history_profile_slot], out[data.history_event_slot]


def load_model(
    group: str,
    seed: int,
    events_per_chunk: int,
    label_smoothing: float,
    device: torch.device,
) -> Model:
    """
    Четыре энкодера из весов этапов 09-12 плюс свежая голова.

    Ни одна размерность не объявляется здесь заново: и dim, и
    глубины, и seed'ы лежат в самих файлах весов вместе с
    состоянием. Поэтому сборка не может разойтись со снимками,
    по которым её будут сверять.
    """

    specials = load_special_tokens()

    saved = _weights(embeddings_dir(group) / EMBEDDING_WEIGHTS, "src.embedding", group)

    embedding = InputEmbedding(
        vocab_size=int(saved["vocab_size"]),
        dim=int(saved["dim"]),
        seed=int(saved["seed"]),
        markers=(specials[EVT], specials[USR]),
    )
    embedding.load_state_dict(saved["state_dict"])

    dim = int(saved["dim"])

    saved = _weights(events_dir(group) / EVENT_WEIGHTS, "src.event", group)
    config = EventConfig.from_dict(saved["config"])
    event = EventEncoder(dim, config.layers, config.heads, config.feedforward,
                         config.dropout, config.seed)
    event.load_state_dict(saved["state_dict"])

    saved = _weights(profiles_dir(group) / PROFILE_WEIGHTS, "src.profile", group)
    config = ProfileConfig.from_dict(saved["config"])
    profile = ProfileEncoder(dim, config.layers, config.heads, config.feedforward,
                             config.dropout, config.seed)
    profile.load_state_dict(saved["state_dict"])

    saved = _weights(history_dir(group) / HISTORY_WEIGHTS, "src.history", group)
    config = HistoryConfig.from_dict(saved["config"])
    history = HistoryEncoder(dim, config.layers, config.heads, config.feedforward,
                             config.dropout, config.rope_base, config.seed)
    history.load_state_dict(saved["state_dict"])

    model = Model(
        embedding=embedding,
        event=event,
        profile=profile,
        history=history,
        head=Mlm(dim, seed),
        events_per_chunk=events_per_chunk,
        label_smoothing=label_smoothing,
    )

    # Веса разыграны и загружены на CPU и только теперь переезжают.
    return model.to(device)


def _weights(path: Path, module: str, group: str) -> dict:

    if not path.exists():
        raise FileNotFoundError(f"нет {path}: выполните python -m {module}.run {group}")

    return torch.load(path, map_location="cpu", weights_only=True)


@contextmanager
def _seeded(seed: int):
    """
    Известное состояние генератора на время сборки весов.
    """

    state = torch.get_rng_state()

    try:
        torch.manual_seed(int(seed))
        yield
    finally:
        torch.set_rng_state(state)


__all__ = [
    "Mlm",
    "Model",
    "PackedBatch",
    "Predicted",
    "load_model",
    "mlm_loss",
    "pack",
]
