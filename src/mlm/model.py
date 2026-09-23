from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

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
# Единица прохода — КЛИЕНТ. История в History Encoder всё равно
# клиентская, и заполнителя при таком проходе не существует
# вовсе: массивы приходят обрезанными по своей длине.
#
# Голова получает на каждую размеченную позицию три вектора:
#
#   1) контекстный вектор самого токена после энкодера события;
#   2) вектор ЕГО события после энкодера истории;
#   3) итоговый вектор клиента после энкодера истории.
#
# Они склеиваются в 3d, проходят одну линейную проекцию 3d -> d,
# и логиты берутся скалярным произведением с ТОЙ ЖЕ таблицей
# эмбеддингов. Отдельной выходной таблицы нет: градиент течёт в
# одни и те же веса и со стороны входа, и со стороны выхода.
#
# Внутри forward нет ни detach, ни NumPy, ни no_grad. Перевод
# данных клиента в тензоры сделан отдельным шагом, до входа в
# граф.
# ============================================================


@dataclass(frozen=True)
class Tensors:
    """
    Один клиент в тензорах. Всё, что видит модель.

    labels здесь лежат рядом, но в саму модель не подаются: они
    нужны только чтобы выбрать позиции и посчитать потери.
    """

    key_ids: torch.Tensor
    value_ids: torch.Tensor
    positions: torch.Tensor
    labels: torch.Tensor

    event_starts: torch.Tensor
    event_lengths: torch.Tensor
    event_time_log: torch.Tensor
    calendar: torch.Tensor

    profile_key_ids: torch.Tensor
    profile_value_ids: torch.Tensor
    profile_positions: torch.Tensor


@dataclass(frozen=True)
class Predicted:
    """
    Что вернул проход по одному клиенту.
    """

    logits: torch.Tensor    # [M, словарь]
    targets: torch.Tensor   # [M]
    loss: torch.Tensor      # скаляр, связанный с графом
    place: torch.Tensor     # [M] номер токена у клиента
    event: torch.Tensor     # [M] номер его события

    @property
    def count(self) -> int:
        return int(self.targets.numel())


def to_tensors(client: Client, device: torch.device) -> Tensors:
    """
    Данные клиента в тензоры. Делается ДО графа значений.
    """

    def ints(values) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.int64, device=device)

    def floats(values) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    return Tensors(
        key_ids=ints(client.key_ids),
        value_ids=ints(client.value_ids),
        positions=ints(client.positions),
        labels=ints(client.labels),
        event_starts=ints(client.event_starts),
        event_lengths=ints(client.event_lengths),
        event_time_log=floats(client.event_time_log),
        calendar=floats(client.calendar),
        profile_key_ids=ints(client.profile_key_ids),
        profile_value_ids=ints(client.profile_value_ids),
        profile_positions=ints(client.profile_positions),
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

    def forward(self, data: Tensors) -> Predicted:
        """
        Один клиент от токенов до потерь.
        """

        n_events = int(data.event_lengths.numel())

        where = torch.nonzero(data.labels != IGNORE, as_tuple=False).squeeze(1)

        # Владелец токена: события лежат подряд и покрывают всю
        # последовательность, поэтому граница однозначна.
        owner = torch.searchsorted(data.event_starts, where, right=True) - 1
        inside = where - data.event_starts[owner]

        column, pad = self._windows(data)

        token_chunks: list[torch.Tensor] = []
        dated_chunks: list[torch.Tensor] = []

        for first in range(0, n_events, self.events_per_chunk):

            last = min(first + self.events_per_chunk, n_events)

            piece = self.event(
                self.embedding.embed(
                    data.key_ids[column[first:last]],
                    data.value_ids[column[first:last]],
                    data.positions[column[first:last]],
                    ~pad[first:last],
                ),
                pad[first:last],
                data.calendar[first:last],
            )

            dated_chunks.append(piece.dated)

            # Из порции сразу берутся только размеченные токены:
            # держать [события, длина, d] целиком незачем, у
            # длинного клиента это сотни мегабайт.
            here = (owner >= first) & (owner < last)

            if bool(here.any()):
                token_chunks.append(piece.tokens[owner[here] - first, inside[here]])

        dated = (
            torch.cat(dated_chunks, dim=0)
            if dated_chunks
            else self.embedding.weight[:0]
        )

        client_vector, event_vectors = self._history(data, dated)

        if token_chunks:
            token_vectors = torch.cat(token_chunks, dim=0)
        else:
            # Целей нет. Контекст пуст, но граф обязан остаться
            # связным, иначе backward на таком клиенте оборвётся.
            token_vectors = event_vectors[:0]

        logits = self.head(
            token_vectors,
            event_vectors[owner],
            client_vector[None].expand(where.numel(), -1),
            self.embedding.weight,
        )

        targets = data.labels[where]

        return Predicted(
            logits=logits,
            targets=targets,
            loss=mlm_loss(logits, targets, self.label_smoothing),
            place=where,
            event=owner,
        )

    def _windows(self, data: Tensors) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Окна событий: номера токенов и маска хвоста.
        """

        width = int(data.event_lengths.max()) if data.event_lengths.numel() else 0

        numbers = torch.arange(width, device=data.event_starts.device)[None, :]

        pad = numbers >= data.event_lengths[:, None]

        starts = data.event_starts[:, None]

        return torch.where(pad, starts, starts + numbers), pad

    def _history(
        self,
        data: Tensors,
        dated: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Анкета и события одной последовательностью.
        """

        profile = self.profile(
            self.embedding.embed(
                data.profile_key_ids[None],
                data.profile_value_ids[None],
                data.profile_positions[None],
                torch.ones_like(data.profile_key_ids, dtype=torch.bool)[None],
            ),
            torch.zeros_like(data.profile_key_ids, dtype=torch.bool)[None],
        )[0]

        sequence = torch.cat([profile[None], dated], dim=0)[None]

        positions = torch.cat(
            [
                torch.zeros(1, dtype=torch.float32, device=dated.device),
                data.event_time_log,
            ]
        )

        out = self.history(sequence, positions)[0]

        return out[0], out[1:]


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
    "Predicted",
    "Tensors",
    "load_model",
    "mlm_loss",
    "to_tensors",
]
