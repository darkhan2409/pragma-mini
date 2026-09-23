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
# collate в один набор тензоров с заполнителем. Модель
# вызывается ОДИН раз на весь micro-batch:
#
#   события    — настоящие события всех клиентов расплющены в
#                одну ось и идут в энкодер события порциями;
#                заполнительные события туда не попадают;
#   анкета     — [B, P] одним вызовом энкодера анкеты;
#   история    — [B, 1 + E]: у каждого клиента своя строка,
#                заполнитель исключён маской внимания;
#   голова     — по всем целям всех клиентов сразу.
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
# данных клиентов в тензоры сделан отдельным шагом, до входа в
# граф.
# ============================================================


@dataclass(frozen=True)
class BatchTensors:
    """
    Micro-batch из B клиентов в тензорах. Всё, что видит модель.

    T, E и P — наибольшие у клиентов batch число токенов событий,
    событий и токенов анкеты. Хвост короче — заполнитель: [PAD] в
    идентификаторах, ноль в числах, False в масках, -100 в labels.

    labels лежат рядом, но в саму модель не подаются: они нужны
    только чтобы выбрать позиции и посчитать потери.
    """

    key_ids: torch.Tensor             # [B, T]
    value_ids: torch.Tensor           # [B, T]
    positions: torch.Tensor           # [B, T]
    labels: torch.Tensor              # [B, T]
    token_mask: torch.Tensor          # [B, T]

    # Начала событий отсчитываются внутри строки своего клиента.
    event_starts: torch.Tensor        # [B, E]
    event_lengths: torch.Tensor       # [B, E], 0 у заполнителя
    event_time_log: torch.Tensor      # [B, E]
    calendar: torch.Tensor            # [B, E, 6]
    event_mask: torch.Tensor          # [B, E]

    profile_key_ids: torch.Tensor     # [B, P]
    profile_value_ids: torch.Tensor   # [B, P]
    profile_positions: torch.Tensor   # [B, P]
    profile_token_mask: torch.Tensor  # [B, P]


@dataclass(frozen=True)
class Predicted:
    """
    Что вернул проход по micro-batch.

    place, event и client называют каждую цель: позицию токена и
    событие внутри клиента и номер клиента в batch.
    """

    logits: torch.Tensor    # [M, словарь]
    targets: torch.Tensor   # [M]
    loss: torch.Tensor      # скаляр, связанный с графом: среднее по M целям
    place: torch.Tensor     # [M] номер токена у клиента
    event: torch.Tensor     # [M] номер его события у клиента
    client: torch.Tensor    # [M] номер клиента в batch

    @property
    def count(self) -> int:
        return int(self.targets.numel())


def collate(clients: list[Client], pad_id: int, device: torch.device) -> BatchTensors:
    """
    Клиенты в один micro-batch. Делается ДО графа значений.

    Массивы клиента копируются в начало своей строки; хвост
    строки — заполнитель. Одна копия на поле: сначала NumPy, затем
    один тензор.
    """

    size = len(clients)

    tokens = max(client.n_tokens for client in clients)
    events = max(client.n_events for client in clients)
    profile = max(client.profile_n_tokens for client in clients)

    def ids(width: int) -> np.ndarray:
        return np.full((size, width), pad_id, dtype=np.int64)

    key_ids, value_ids = ids(tokens), ids(tokens)
    positions = np.zeros((size, tokens), dtype=np.int64)
    labels = np.full((size, tokens), IGNORE, dtype=np.int64)
    token_mask = np.zeros((size, tokens), dtype=bool)

    event_starts = np.zeros((size, events), dtype=np.int64)
    event_lengths = np.zeros((size, events), dtype=np.int64)
    event_time_log = np.zeros((size, events), dtype=np.float32)
    calendar = np.zeros((size, events, CALENDAR_PER_EVENT), dtype=np.float32)
    event_mask = np.zeros((size, events), dtype=bool)

    profile_key_ids, profile_value_ids = ids(profile), ids(profile)
    profile_positions = np.zeros((size, profile), dtype=np.int64)
    profile_token_mask = np.zeros((size, profile), dtype=bool)

    for row, client in enumerate(clients):

        n, e, p = client.n_tokens, client.n_events, client.profile_n_tokens

        key_ids[row, :n] = client.key_ids
        value_ids[row, :n] = client.value_ids
        positions[row, :n] = client.positions
        labels[row, :n] = client.labels
        token_mask[row, :n] = True

        event_starts[row, :e] = client.event_starts
        event_lengths[row, :e] = client.event_lengths
        event_time_log[row, :e] = client.event_time_log
        calendar[row, :e] = client.calendar
        event_mask[row, :e] = True

        profile_key_ids[row, :p] = client.profile_key_ids
        profile_value_ids[row, :p] = client.profile_value_ids
        profile_positions[row, :p] = client.profile_positions
        profile_token_mask[row, :p] = True

    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, device=device)

    return BatchTensors(
        key_ids=tensor(key_ids),
        value_ids=tensor(value_ids),
        positions=tensor(positions),
        labels=tensor(labels),
        token_mask=tensor(token_mask),
        event_starts=tensor(event_starts),
        event_lengths=tensor(event_lengths),
        event_time_log=tensor(event_time_log),
        calendar=tensor(calendar),
        event_mask=tensor(event_mask),
        profile_key_ids=tensor(profile_key_ids),
        profile_value_ids=tensor(profile_value_ids),
        profile_positions=tensor(profile_positions),
        profile_token_mask=tensor(profile_token_mask),
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

    def forward(self, data: BatchTensors) -> Predicted:
        """
        Micro-batch от токенов до потерь, одним проходом.
        """

        tokens = data.key_ids.shape[1]

        # Настоящие события всех клиентов одной осью: номер клиента
        # и номер события у него. nonzero идёт по строкам, поэтому
        # события лежат клиент за клиентом, внутри — по порядку.
        client_of, local_of = torch.nonzero(data.event_mask, as_tuple=True)

        starts = data.event_starts[client_of, local_of]
        lengths = data.event_lengths[client_of, local_of]

        n_events = int(starts.numel())

        # Цели micro-batch: клиент и позиция токена у него.
        target_client, target_place = torch.nonzero(data.labels != IGNORE, as_tuple=True)

        # Владелец цели ищется по сквозной координате client*T +
        # позиция: у событий она строго растёт, потому что события
        # клиента лежат подряд и покрывают всю его строку.
        owner = torch.searchsorted(
            client_of * tokens + starts, target_client * tokens + target_place, right=True
        ) - 1

        inside = target_place - starts[owner]
        target_event = local_of[owner]

        column, pad = self._windows(starts, lengths)

        token_chunks: list[torch.Tensor] = []
        dated_chunks: list[torch.Tensor] = []

        for first in range(0, n_events, self.events_per_chunk):

            last = min(first + self.events_per_chunk, n_events)

            rows = client_of[first:last, None]
            cols = column[first:last]

            piece = self.event(
                self.embedding.embed(
                    data.key_ids[rows, cols],
                    data.value_ids[rows, cols],
                    data.positions[rows, cols],
                    ~pad[first:last],
                ),
                pad[first:last],
                data.calendar[client_of[first:last], local_of[first:last]],
            )

            dated_chunks.append(piece.dated)

            # Из порции сразу берутся только размеченные токены:
            # держать [события, длина, d] целиком незачем, у
            # длинного клиента это сотни мегабайт. Цели и владельцы
            # идут в одном порядке, поэтому склейка порций его
            # сохраняет.
            here = (owner >= first) & (owner < last)

            if bool(here.any()):
                token_chunks.append(piece.tokens[owner[here] - first, inside[here]])

        dated = (
            torch.cat(dated_chunks, dim=0)
            if dated_chunks
            else self.embedding.weight[:0]
        )

        client_vectors, event_vectors = self._history(data, dated, client_of, local_of)

        if token_chunks:
            token_vectors = torch.cat(token_chunks, dim=0)
        else:
            # Целей нет. Контекст пуст, но граф обязан остаться
            # связным, иначе backward на таком batch оборвётся.
            token_vectors = client_vectors[:0]

        logits = self.head(
            token_vectors,
            event_vectors[target_client, target_event],
            client_vectors[target_client],
            self.embedding.weight,
        )

        targets = data.labels[target_client, target_place]

        return Predicted(
            logits=logits,
            targets=targets,
            loss=mlm_loss(logits, targets, self.label_smoothing),
            place=target_place,
            event=target_event,
            client=target_client,
        )

    @staticmethod
    def _windows(
        starts: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Окна событий: номера токенов в строке клиента и маска хвоста.
        """

        width = int(lengths.max()) if lengths.numel() else 0

        numbers = torch.arange(width, device=starts.device)[None, :]

        pad = numbers >= lengths[:, None]

        return torch.where(pad, starts[:, None], starts[:, None] + numbers), pad

    def _history(
        self,
        data: BatchTensors,
        dated: torch.Tensor,
        client_of: torch.Tensor,
        local_of: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Анкета и события одной последовательностью на клиента.

        [B, 1 + E, d]: слот 0 — вектор анкеты, дальше события
        клиента на своих местах, хвост — заполнитель под маской.
        """

        size, events = data.event_mask.shape

        profile = self.profile(
            self.embedding.embed(
                data.profile_key_ids,
                data.profile_value_ids,
                data.profile_positions,
                data.profile_token_mask,
            ),
            ~data.profile_token_mask,
        )

        # Вне графа только нулевой холст: index_put возвращает новый
        # тензор, и градиент идёт к векторам событий.
        placed = dated.new_zeros(size, events, dated.shape[-1]).index_put(
            (client_of, local_of), dated
        )

        sequence = torch.cat([profile[:, None], placed], dim=1)

        positions = torch.cat(
            [
                torch.zeros(size, 1, dtype=torch.float32, device=dated.device),
                data.event_time_log,
            ],
            dim=1,
        )

        mask = torch.cat(
            [
                torch.ones(size, 1, dtype=torch.bool, device=dated.device),
                data.event_mask,
            ],
            dim=1,
        )

        out = self.history(sequence, positions, mask)

        return out[:, 0], out[:, 1:]


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
    "BatchTensors",
    "Mlm",
    "Model",
    "Predicted",
    "collate",
    "load_model",
    "mlm_loss",
]
