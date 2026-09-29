from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.embedding.inputs import CALENDAR_PER_EVENT
from src.embedding.layer import InputEmbedding
from src.event.encoder import EventEncoder
from src.history.encoder import HistoryEncoder
from src.profile.encoder import ProfileEncoder
from src.tokenization.specials import EVT, USR, load_special_tokens

from .backbone import load_backbone, read_embedding, recorded_backbone
from .inputs import IGNORE, Client
from .varlen import (
    BackendError,
    VarlenLayout,
    assemble,
    encoder_layer_varlen,
    history_block_varlen,
    resolve_backend,
)


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
#   анкеты     — сегмент = токены анкеты клиента, энкодер анкеты:
#                Attributes на cutoff и вехи Lifelong со своим
#                временем (profile_time_log) внутри TimeRoPE;
#   истории    — сегмент = [анкета, события клиента], энкодер
#                истории: слот анкеты первым, как [USR].
#
# Два пути внимания, один и тот же результат:
#
#   flash — CUDA, bf16/fp16 под autocast и библиотека flash-attn:
#           плоские Q/K/V по cu_seqlens, ни одной позиции
#           заполнителя (varlen.attend);
#   sdpa  — всё остальное: сегменты по корзинам близкой длины,
#           прямоугольник с заполнителем только внутри корзины.
#
# Путь выбирается внутри forward: код обучения о нём не знает.
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
# весит столько же, сколько цель другого. Считаются они кусками по
# TARGETS_PER_CHUNK целей с пересчётом в backward (mlm_loss): память
# потерь не растёт с числом целей.
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
    event_masked: torch.Tensor    # [T] bool: значение события, закрытого механизмом event
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
    profile_time_log: torch.Tensor  # давность вехи до cutoff, ноль у [USR] и Attributes
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

    logits: torch.Tensor | None  # [M, словарь]; None, если проход просили без них
    targets: torch.Tensor   # [M]
    loss: torch.Tensor      # скаляр, связанный с графом: среднее по M целям
    place: torch.Tensor     # [M] номер токена у клиента
    event: torch.Tensor     # [M] номер его события у клиента
    client: torch.Tensor    # [M] номер клиента в micro-batch

    # (угадано первым, попало в первые 5) — у прохода без логитов;
    # с логитами их считает вызывающий по logits.
    hits: tuple[int, int] | None = None

    # Вспомогательная потеря [USR] (RecentTypes), связанная с графом;
    # None — цели нет.
    aux: torch.Tensor | None = None

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

        # Время анкеты плоским массивом рядом с её токенами:
        # несовпадение длин сдвинуло бы время на токены соседа.
        if client.profile_time_log.size != client.profile_n_tokens:
            raise ValueError(
                f"{client.client_id}: время анкеты на {client.profile_time_log.size} "
                f"токенов при {client.profile_n_tokens} токенах анкеты"
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
        event_masked=tensor(np.concatenate(
            [np.asarray(client.reason, dtype=object) == "event" for client in clients]
        ).astype(bool)),
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
        profile_time_log=tensor(join("profile_time_log", np.float32)),
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


# Целей в одном куске потерь.
TARGETS_PER_CHUNK = 2048


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
    head: Mlm,
    token: torch.Tensor,
    event: torch.Tensor,
    client: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float,
    allowed: torch.Tensor | None = None,
    rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Кросс-энтропия по размеченным позициям: среднее по целям.

    Голова и потери считаются кусками по TARGETS_PER_CHUNK целей
    и пересчитываются в backward (checkpoint): до backward живут
    только входы куска. Целиком под bf16 autocast кросс-энтропия
    держала бы около 12·словарь байт на цель — bf16-логарифмы
    вероятностей, их fp32-копию и градиенты, — а token_budget
    число целей не ограничивает.

    Сумма кусков с reduction="sum", делённая на число целей, — то
    же среднее с тем же сглаживанием меток: torch сам считает
    (1 − ε)·nll + ε/V·Σ по строкам с меткой.

    allowed [ключи, словарь] и rows [M] — множества кандидатов по
    ключу цели (Model.restrict): softmax и сглаживание меток тогда
    идут только по значениям своего ключа.

    Ноль на пустом наборе возвращается СВЯЗАННЫМ С ГРАФОМ:
    torch.tensor(0.0) оборвал бы цепочку, и backward на клиенте
    без целей упал бы. Приём взят из эталона дословно.
    """

    count = int((targets != IGNORE).sum())

    if count == 0:
        return head(token, event, client, weight).sum() * 0.0

    parts = [token, event, client, targets] + ([rows] if allowed is not None else [])

    pieces = zip(*(part.split(TARGETS_PER_CHUNK) for part in parts))

    total = sum(
        checkpoint(
            _piece_loss, head, *piece[:4], weight, smoothing,
            allowed, piece[4] if allowed is not None else None, use_reentrant=False,
        )
        for piece in pieces
    )

    return total / count


def _piece_loss(
    head: Mlm,
    token: torch.Tensor,
    event: torch.Tensor,
    client: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
    smoothing: float,
    allowed: torch.Tensor | None = None,
    rows: torch.Tensor | None = None,
) -> torch.Tensor:

    logits = head(token, event, client, weight)

    if allowed is None:
        return F.cross_entropy(
            logits, targets, ignore_index=IGNORE, label_smoothing=smoothing, reduction="sum",
        )

    # Softmax по кандидатам своего ключа; сглаживание — среднее по
    # тем же кандидатам, а не по всему словарю: иначе масса ε ушла бы
    # на значения, которых у этого ключа не бывает.
    mask = allowed[rows]
    logp = logits.float().masked_fill(~mask, float("-inf")).log_softmax(dim=-1)
    nll = -logp.gather(1, targets[:, None]).squeeze(1)

    if smoothing:
        spread = -logp.masked_fill(~mask, 0.0).sum(dim=-1) / mask.sum(dim=-1)
        nll = (1.0 - smoothing) * nll + smoothing * spread

    return nll.sum()


def hits_in_pieces(
    head: Mlm,
    token: torch.Tensor,
    event: torch.Tensor,
    client: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
    allowed: torch.Tensor | None = None,
    rows: torch.Tensor | None = None,
) -> tuple[int, int]:
    """
    То же, что hits по полным логитам, но кусками по
    TARGETS_PER_CHUNK и без графа: логиты [M, словарь] целиком не
    живут ни одного мгновения. Счёт копится на устройстве, и
    синхронизация одна.
    """

    first = five = targets.new_zeros(())

    with torch.no_grad():

        parts = [token, event, client, targets] + ([rows] if allowed is not None else [])

        for piece in zip(*(part.split(TARGETS_PER_CHUNK) for part in parts)):
            logits = head(piece[0], piece[1], piece[2], weight)
            if allowed is not None:
                logits = logits.masked_fill(~allowed[piece[4]], float("-inf"))
            labels = piece[3]
            top = logits.topk(min(k, logits.shape[-1]), dim=-1).indices
            first = first + (top[:, 0] == labels).sum()
            five = five + (top == labels[:, None]).any(dim=-1).sum()

    return int(first), int(five)


def hits(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> tuple[int, int]:
    """
    Сколько целей угадано первым ответом и сколько попало в первые k.

    logits [M, словарь] и targets [M] — строки настоящих целей
    (pack берёт только позиции с меткой != -100); метка -100, если
    она всё же пришла, в счёт не идёт. Знаменатель доли — число
    целей, его считает вызывающий. Без целей — (0, 0).
    """

    # Без выборки по маске: она копировала бы [M, словарь] на
    # каждом шаге, а метка -100 и так не совпадает ни с одним
    # номером словаря.
    top = logits.detach().topk(min(k, logits.shape[-1]), dim=-1).indices

    return int((top[:, 0] == targets).sum()), int((top == targets[:, None]).any(dim=-1).sum())


# Окна вспомогательной цели [USR], в сутках до точки отсчёта времени
# (этап 06: cutoff T, а с --anchor last_event — последнее событие).
RECENT_DAYS = (7, 30, 90)


class RecentTypes(nn.Module):
    """
    Вспомогательная цель вектора клиента: по [USR] предсказать, из
    каких типов событий состоят последние 7, 30 и 90 дней истории.

    [USR] у MLM своей цели не имеет — он лишь третий вектор в голове
    каждой цели, — и вырождается в кодировку анкеты. Здесь он
    обязан помнить недавнее прошлое клиента (как в NPPR: «вспомнить
    прошлое» — самое полезное для задач уровня клиента).

    Цель — доля каждого типа среди событий окна; тип закрытого
    маской события берётся из его метки. Потеря — кросс-энтропия
    с этой долей, среднее по (клиент, окно) с событиями.
    """

    def __init__(self, dim: int, type_of_value: torch.Tensor, event_type_key: int, seed: int):

        super().__init__()

        self.types = int(type_of_value.max()) + 1
        self.event_type_key = int(event_type_key)

        self.register_buffer("type_of_value", type_of_value, persistent=False)

        with _seeded(seed):
            self.proj = nn.Linear(dim, self.types * len(RECENT_DAYS))

    def forward(self, data: "PackedBatch", usr: torch.Tensor) -> torch.Tensor:

        from src.temporal.position import TIME_SCALE

        # Исходное значение каждого токена: у закрытого маской — метка.
        original = torch.where(data.labels != IGNORE, data.labels, data.value_ids)

        typed = torch.nonzero(
            (data.key_ids == self.event_type_key) & (data.positions == 0), as_tuple=True
        )[0]

        kind = torch.full((data.events.segments,), -1, dtype=torch.long, device=usr.device)
        kind[data.event_of_token[typed]] = self.type_of_value[original[typed]]

        # Давность до точки отсчёта — из той же шкалы, что видит
        # энкодер истории: seconds = 8·expm1(позиция / 8).
        days = TIME_SCALE * torch.expm1(data.event_time_log.float() / TIME_SCALE) / 86_400.0

        known = kind >= 0
        windows = torch.tensor(RECENT_DAYS, dtype=days.dtype, device=days.device)
        inside = (days[:, None] <= windows[None, :]) & known[:, None]          # [E, W]

        counts = usr.new_zeros(data.clients, len(RECENT_DAYS), self.types, dtype=torch.float32)
        event, window = torch.nonzero(inside, as_tuple=True)
        counts.index_put_(
            (data.user_of_event[event], window, kind[event]),
            torch.ones_like(event, dtype=torch.float32), accumulate=True,
        )

        totals = counts.sum(dim=-1)
        present = totals > 0

        if not bool(present.any()):
            return (usr.sum() * 0.0).float()

        share = counts / totals.clamp(min=1.0).unsqueeze(-1)

        logq = self.proj(usr).float().view(data.clients, len(RECENT_DAYS), self.types).log_softmax(-1)

        return -(share * logq).sum(dim=-1)[present].mean()


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
        attention: str = "sdpa",
        strict: bool = False,
    ):

        super().__init__()

        self.embedding = embedding
        self.event = event
        self.profile = profile
        self.history = history
        self.head = head

        self.events_per_chunk = int(events_per_chunk)
        self.label_smoothing = float(label_smoothing)

        # attention — решённый бэкенд ("flash" или "sdpa"); strict
        # — flash выбран явно, и откат на корзины запрещён.
        self.attention = attention
        self.strict = bool(strict)

        # Кандидаты по ключу цели (restrict); None — softmax по всему
        # словарю. Буферы, а не веса: в state_dict не входят.
        self.register_buffer("key_row", None, persistent=False)
        self.register_buffer("allowed", None, persistent=False)

        # Вспомогательная цель [USR] (attach_recent); None — её нет.
        self.recent: RecentTypes | None = None

        # Чем закрыт ключ у значений события под маской event
        # (hide_event_keys); None — ключи видны.
        self.hidden_key: int | None = None

    def attach_recent(self, recent: RecentTypes) -> None:
        self.recent = recent.to(self.embedding.weight.device)

    def hide_event_keys(self, hidden: int) -> None:
        """
        У значений события под маской event ключ во входе заменяется
        на hidden: иначе набор видимых ключей выдаёт тип события и
        схему полей, и событие угадывается без истории. Какой ключ
        предсказывать, голова узнаёт из запроса — эмбеддинга ключа
        цели, прибавленного к вектору её токена.
        """

        self.hidden_key = int(hidden)

    def _event_keys(self, data: PackedBatch) -> torch.Tensor:
        """
        Ключи токенов событий во входе энкодера события.
        """

        if self.hidden_key is None:
            return data.key_ids

        return data.key_ids.masked_fill(data.event_masked, self.hidden_key)

    def restrict(self, key_row: torch.Tensor, allowed: torch.Tensor) -> None:
        """
        Предсказывать значение только среди кандидатов своего ключа:
        key_row [словарь] — строка ключа в allowed или -1, allowed
        [ключи, словарь] — кто бывает значением ключа.
        """

        device = self.embedding.weight.device

        self.key_row = key_row.to(device)
        self.allowed = allowed.to(device)

    def forward(self, data: PackedBatch, logits: bool = True) -> Predicted:
        """
        Micro-batch от токенов до потерь, одним проходом.

        logits=False — без полных логитов [M, словарь]: их граф жил бы
        весь backward ради одного счёта top-1/top-5. Обучение просит
        так; счёт тогда приходит в Predicted.hits.
        """

        token_vectors, event_vectors, client_vectors = self._encode(data)

        if token_vectors is None:
            # Целей нет. Контекст пуст, но граф обязан остаться
            # связным, иначе backward на таком batch оборвётся.
            token_vectors = client_vectors[:0]

        # Запрос головы: ключ цели. Во входе он мог быть закрыт.
        if self.hidden_key is not None:
            keys = data.key_ids[data.target_token]
            token_vectors = token_vectors + (
                self.embedding.table(keys) * self.embedding.scale
            ).to(token_vectors.dtype)

        event_rows = event_vectors[data.target_event]
        client_rows = client_vectors[data.target_client]

        targets = data.labels[data.target_token]

        rows = self.key_row[data.key_ids[data.target_token]] if self.allowed is not None else None

        # Логиты целиком — для точности, отчёта и разбора по целям;
        # потери считает mlm_loss кусками, по тем же входам головы.
        full = (
            self.head(token_vectors, event_rows, client_rows, self.embedding.weight)
            if logits else None
        )

        if full is not None and rows is not None:
            full = full.masked_fill(~self.allowed[rows], float("-inf"))

        scored = None if logits else hits_in_pieces(
            self.head, token_vectors, event_rows, client_rows, self.embedding.weight, targets,
            allowed=self.allowed, rows=rows,
        )

        return Predicted(
            logits=full,
            hits=scored,
            aux=self.recent(data, client_vectors) if self.recent is not None else None,
            targets=targets,
            loss=mlm_loss(
                self.head, token_vectors, event_rows, client_rows,
                self.embedding.weight, targets, self.label_smoothing,
                allowed=self.allowed, rows=rows,
            ),
            place=data.target_place,
            event=data.target_local,
            client=data.target_client,
        )

    def client_embeddings(self, data: PackedBatch) -> torch.Tensor:
        """
        client_embedding: [B, d] — позиция [USR] каждого клиента после
        последнего блока энкодера истории и его финальной нормы.

        Это тот же вектор клиента, что получает голова MLM, но без
        головы: не вектор энкодера анкеты (он лишь вход истории, где
        [USR] двунаправленно видит все события клиента) и не выход
        головы. Путь внимания тот же, что у forward.
        """

        return self._encode(data)[2]

    def readouts(self, data: PackedBatch) -> dict[str, torch.Tensor]:
        """
        Векторы клиента для оценки на задачах, [B, d] каждый, из
        одного прохода:

          usr         [USR] после истории — то же, что client_embeddings;
          profile     выход энкодера анкеты, до истории;
          mean_event  среднее векторов событий клиента после истории;
          last_event  вектор его последнего события после истории.

        У клиента без событий mean_event и last_event нулевые.
        """

        if self._flash():
            dated, _ = self._events_flash(data)
            profile = self._profiles_flash(data)
            usr, events = self._history_flash(data, profile, dated)
        else:
            dated, _ = self._events(data)
            profile = self._profiles(data)
            usr, events = self._history(data, profile, dated)

        events = events.float()

        counts = torch.bincount(data.user_of_event, minlength=data.clients)

        mean = events.new_zeros(data.clients, events.shape[-1]).index_add_(
            0, data.user_of_event, events
        ) / counts.clamp(min=1).unsqueeze(-1)

        # События клиента лежат подряд и по времени: последнее —
        # перед началом следующего клиента.
        last = events.new_zeros(data.clients, events.shape[-1])
        present = counts > 0
        last[present] = events[(torch.cumsum(counts, 0) - 1)[present]]

        return {"usr": usr.float(), "profile": profile.float(), "mean_event": mean, "last_event": last}

    def _encode(self, data: PackedBatch) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """
        Векторы целевых токенов (None без целей), событий и клиентов
        после энкодера истории.
        """

        if self._flash():
            dated, token_vectors = self._events_flash(data)
            profile = self._profiles_flash(data)
            client_vectors, event_vectors = self._history_flash(data, profile, dated)
        else:
            dated, token_vectors = self._events(data)
            profile = self._profiles(data)
            client_vectors, event_vectors = self._history(data, profile, dated)

        return token_vectors, event_vectors, client_vectors

    def _flash(self) -> bool:
        """
        Идти ли этим проходом через varlen-ядро.

        Ядро flash-attn считает только в bf16 и fp16, поэтому кроме
        выбранного бэкенда нужен ещё autocast CUDA в одном из них.
        Без него auto уходит на корзины, а явный flash — ошибка.
        """

        if self.attention != "flash":
            return False

        if torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") in (
            torch.bfloat16, torch.float16
        ):
            return True

        if self.strict:
            raise BackendError(
                "attention_backend=flash: проход идёт не под autocast CUDA в bf16/fp16, "
                "а flash-attn в fp32 не считает"
            )

        return False

    def _events_flash(self, data: PackedBatch) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Энкодер события на плоских токенах, без заполнителя.

        Все токены micro-batch — одна ось [T, d]; внимание не выходит
        за границы события (cu_seqlens_event). Контекст всех токенов
        живёт до конца прохода энкодера — он ограничен token_budget;
        наружу идут только векторы целевых токенов.
        """

        encoder = self.event

        x = self.embedding.embed(
            self._event_keys(data), data.value_ids, data.positions,
            torch.ones_like(data.key_ids, dtype=torch.bool),
        )

        for layer in encoder.layers:
            x = encoder_layer_varlen(layer, x, data.events)

        x = encoder.norm(x)

        # Маркер [EVT] — первый токен события: его вектор и есть
        # вектор события, к нему прибавляется календарь.
        dated = x[data.events.cu_seqlens[:-1]] + encoder.calendar(data.calendar)

        if data.target_token.numel() == 0:
            return dated, None

        return dated, x[data.target_token]

    def _profiles_flash(self, data: PackedBatch) -> torch.Tensor:
        """
        Энкодер анкеты на плоских токенах: [B, d] из позиции [USR].

        Углы TimeRoPE по времени анкеты считаются в fp32 один раз и
        служат всем блокам — как у истории.
        """

        encoder = self.profile

        x = self.embedding.embed(
            data.profile_key_ids, data.profile_value_ids, data.profile_positions,
            torch.ones_like(data.profile_key_ids, dtype=torch.bool),
        )

        cos, sin = encoder.rope.angles(data.profile_time_log)

        for block in encoder.layers:
            x = history_block_varlen(block, encoder.rope, x, cos, sin, data.profiles)

        return encoder.norm(x)[data.profiles.cu_seqlens[:-1]]

    def _history_flash(
        self,
        data: PackedBatch,
        profile: torch.Tensor,
        dated: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Энкодер истории на плоской истории [B + E, d], без заполнителя.

        Углы TimeRoPE считаются в fp32 один раз по плоским позициям
        и служат всем блокам.
        """

        encoder = self.history

        x = (
            profile.new_zeros(data.clients + data.events.segments, profile.shape[-1])
            .index_copy(0, data.history_profile_slot, profile)
            .index_copy(0, data.history_event_slot, dated)
        )

        cos, sin = encoder.rope.angles(data.history_positions)

        for block in encoder.layers:
            x = history_block_varlen(block, encoder.rope, x, cos, sin, data.history)

        x = encoder.norm(x)

        return x[data.history_profile_slot], x[data.history_event_slot]

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

        keys = self._event_keys(data)

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
                        keys[index],
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

            # Время хвоста — ноль: маска и так закрывает его, а
            # индекс хвоста смотрит на настоящий токен со своим
            # временем.
            positions = torch.where(
                bucket.mask, data.profile_time_log[bucket.index], 0.0
            )

            parts.append(
                self.profile(
                    self.embedding.embed(
                        data.profile_key_ids[bucket.index],
                        data.profile_value_ids[bucket.index],
                        data.profile_positions[bucket.index],
                        bucket.mask,
                    ),
                    positions,
                    bucket.mask,
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
    seed: int,
    events_per_chunk: int,
    label_smoothing: float,
    device: torch.device,
    attention_backend: str = "auto",
    backbone: dict | None = None,
) -> Model:
    """
    Модель: входной слой этапа 09 (train), начальные веса backbone
    (python -m src.mlm.init_backbone) и свежая голова.

    backbone — lineage backbone из чекпойнта обученной модели:
    энкодеры тогда строятся по записанной в нём архитектуре
    (recorded_backbone), а их веса придут из state_dict.

    Модель одна: обучение, validation и отчёты по любой группе
    собирают её из одних и тех же весов. Ни одна размерность не
    объявляется здесь заново: d, глубины и seed'ы лежат в файлах
    весов вместе с состоянием. Этапы 10–13 не читаются — их векторы
    и отчёты модели не нужны.
    """

    specials = load_special_tokens()

    saved = read_embedding()

    embedding = InputEmbedding(
        vocab_size=int(saved["vocab_size"]),
        dim=int(saved["dim"]),
        seed=int(saved["seed"]),
        markers=(specials[EVT], specials[USR]),
    )
    embedding.load_state_dict(saved["state_dict"])

    event, profile, history = (
        load_backbone(saved) if backbone is None else recorded_backbone(saved, backbone)
    )

    model = Model(
        embedding=embedding,
        event=event,
        profile=profile,
        history=history,
        head=Mlm(int(saved["dim"]), seed),
        events_per_chunk=events_per_chunk,
        label_smoothing=label_smoothing,
        # Недоступный явный flash — ошибка сразу, до первого шага.
        attention=resolve_backend(attention_backend, device),
        strict=attention_backend == "flash",
    )

    # Веса разыграны и загружены на CPU и только теперь переезжают.
    return model.to(device)


def recent_types(artifacts, dim: int, seed: int) -> RecentTypes:
    """
    Голова RecentTypes под словарь: номер типа у каждого значения
    ключа event_type, -1 у остальных токенов.
    """

    from src.dataset.tokenized import EVENT_TYPE_KEY

    key = artifacts.key_id(EVENT_TYPE_KEY)

    if key is None:
        raise ValueError("в словаре нет ключа event_type: вспомогательной цели не из чего строиться")

    type_of_value = torch.full((artifacts.size,), -1, dtype=torch.long)

    for number, token_id in enumerate(sorted(artifacts.values[EVENT_TYPE_KEY].values())):
        type_of_value[token_id] = number

    return RecentTypes(dim, type_of_value, key, seed)


def candidate_table(artifacts) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Кандидаты значения по ключу — из словаря, а не из данных:
    категории ключа, его корзины, куски BPE у текстового ключа и
    [UNK]. Так любая метка, даже невиданная в train, остаётся внутри
    своего множества, и бесконечной потери не бывает.
    """

    from src.tokenization.specials import UNK

    keys = sorted(artifacts.keys.items(), key=lambda item: item[1])

    key_row = torch.full((artifacts.size,), -1, dtype=torch.long)
    allowed = torch.zeros((len(keys), artifacts.size), dtype=torch.bool)

    unknown = artifacts.special(UNK)

    for row, (key, token_id) in enumerate(keys):

        key_row[token_id] = row

        kind = artifacts.kind(key)

        if kind == "categorical":
            ids = list(artifacts.values[key].values())
        elif kind == "numeric":
            ids = [bucket.token_id for bucket in artifacts.buckets[key]]
        else:
            ids = list(artifacts.bpe_ids)

        allowed[row, ids] = True
        allowed[row, unknown] = True

    return key_row, allowed


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
    "TARGETS_PER_CHUNK",
    "RECENT_DAYS",
    "RecentTypes",
    "candidate_table",
    "recent_types",
    "hits",
    "hits_in_pieces",
    "load_model",
    "mlm_loss",
    "pack",
]
