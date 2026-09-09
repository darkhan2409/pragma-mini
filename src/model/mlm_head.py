from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.preprocessing.artifacts import read_json
from src.tokenizer.config import FIELD_VALUE_IDS_FILE, IncompatibleArtifactsError
from src.tokenizer.vocab import Vocab

from .config import ModelConfig


# ============================================================
# ИДЕЯ
# ============================================================
#
# Предсказание замаскированного значения это выбор среди
# кандидатов ОДНОГО поля, а не softmax по всему словарю. У mcc
# 70 кандидатов, у direction два, у amount шестнадцать корзин;
# общая голова смешивала бы несравнимые вещи и тратила бы почти
# всю вероятностную массу на заведомо невозможные значения.
#
# Поэтому:
#
#   общая часть   concat(h_local, h_event, h_usr)
#                 → Linear(3d, d) → GELU → LayerNorm
#   на каждое поле   Linear(d, n_candidates этого поля)
#
# Все головы создаются в __init__, до optimizer: параметр,
# появившийся внутри forward, не попал бы ни в optimizer, ни в
# checkpoint, и обучался бы молча никак.
#
# Поля, у которых на train меньше двух кандидатов, головы не
# получают: там нечего выбирать, а CE была бы тождественным
# нулём и портила бы среднее по полям.
# ============================================================


class TargetError(ValueError):
    """
    Цель не принадлежит своему полю или поле не предсказуемо.
    """


# ============================================================
# КАНДИДАТЫ ПОЛЕЙ
# ============================================================


class FieldTable:
    """
    Границы кандидатов каждого поля и перевод global ↔ local.

    Массивы индексируются token ID, поэтому перевод целого
    batch'а это одна векторная операция без словарей.
    """

    def __init__(self, vocab: Vocab):

        self.vocab = vocab

        self.first_value_id = vocab.first_value_id

        size = vocab.size

        self.value_start = np.full(size, -1, dtype=np.int64)
        self.value_end = np.full(size, -1, dtype=np.int64)
        self.n_candidates = np.zeros(size, dtype=np.int64)
        self.predictable = np.zeros(size, dtype=bool)

        self._name: dict[int, str] = {}
        self._kind: dict[int, str] = {}

        for entry in vocab.keys:
            self.value_start[entry.id] = entry.value_start
            self.value_end[entry.id] = entry.value_end
            self.n_candidates[entry.id] = entry.n_values
            self.predictable[entry.id] = entry.predictable
            self._name[entry.id] = entry.key
            self._kind[entry.id] = entry.kind

        # Поле с одним кандидатом предсказывать нечем.
        self.degenerate = self.predictable & (self.n_candidates < 2)

        self.trainable = self.predictable & (self.n_candidates >= 2)

        self.trainable_key_ids: tuple[int, ...] = tuple(
            int(entry.id) for entry in vocab.keys if self.trainable[entry.id]
        )

        self.degenerate_key_ids: tuple[int, ...] = tuple(
            int(entry.id) for entry in vocab.keys if self.degenerate[entry.id]
        )

    # --------------------------------------------------------

    @staticmethod
    def load(vocab: Vocab, vocab_dir: Path) -> "FieldTable":
        """
        Тот же словарь, что записан в field_value_ids.json.

        Файл читается не ради данных, а ради проверки: если он
        разошёлся со словарём, кандидаты головы были бы не те,
        что имел в виду tokenizer.
        """

        artifact = read_json(Path(vocab_dir) / FIELD_VALUE_IDS_FILE)

        fields = artifact.get("fields", {})

        if len(fields) != vocab.n_keys:
            raise IncompatibleArtifactsError(
                f"{FIELD_VALUE_IDS_FILE}: полей {len(fields)}, а в словаре ключей {vocab.n_keys}"
            )

        for entry in vocab.keys:

            item = fields.get(entry.key)

            if item is None:
                raise IncompatibleArtifactsError(f"{FIELD_VALUE_IDS_FILE}: нет поля {entry.key}")

            if int(item["key_id"]) != entry.id:
                raise IncompatibleArtifactsError(
                    f"{FIELD_VALUE_IDS_FILE}: у поля {entry.key} ID {item['key_id']}, "
                    f"в словаре {entry.id}"
                )

            if bool(item["predictable"]) != entry.predictable or item["kind"] != entry.kind:
                raise IncompatibleArtifactsError(
                    f"{FIELD_VALUE_IDS_FILE}: описание поля {entry.key} не совпадает со словарём"
                )

            if list(item["value_ids"]) != list(range(entry.value_start, entry.value_end)):
                raise IncompatibleArtifactsError(
                    f"{FIELD_VALUE_IDS_FILE}: кандидаты поля {entry.key} не совпадают с "
                    f"диапазоном словаря [{entry.value_start}, {entry.value_end})"
                )

        return FieldTable(vocab)

    # --------------------------------------------------------

    def name(self, key_id: int) -> str:
        return self._name.get(int(key_id), f"key_{int(key_id)}")

    def kind(self, key_id: int) -> str:
        return self._kind.get(int(key_id), "unknown")

    def size_of(self, key_id: int) -> int:
        return int(self.n_candidates[int(key_id)])

    def is_trainable(self, key_id: int) -> bool:
        return bool(self.trainable[int(key_id)])

    # --------------------------------------------------------

    def check_targets(self, key_ids: np.ndarray, values: np.ndarray) -> None:
        """
        Каждая цель это настоящее значение своего поля.

        Одного вычитания смещения мало: чужое значение дало бы
        локальный индекс в диапазоне соседнего поля и молча
        обучало бы голову неверной метке.
        """

        keys = np.asarray(key_ids, dtype=np.int64)
        found = np.asarray(values, dtype=np.int64)

        if keys.shape != found.shape:
            raise TargetError(f"ключей {keys.shape}, значений {found.shape}")

        if keys.size == 0:
            return

        if int(keys.min()) < 0 or int(keys.max()) >= self.value_start.size:
            raise TargetError("ключ вне словаря")

        bad = np.flatnonzero(~self.predictable[keys])

        if bad.size:
            position = int(bad[0])
            raise TargetError(
                f"поле {self.name(int(keys[position]))} не помечено predictable, "
                "его значения не могут быть целями"
            )

        bad = np.flatnonzero(found < self.first_value_id)

        if bad.size:
            position = int(bad[0])
            raise TargetError(
                f"цель {int(found[position])} поля {self.name(int(keys[position]))} это "
                "special-токен, а не значение словаря"
            )

        outside = (found < self.value_start[keys]) | (found >= self.value_end[keys])

        bad = np.flatnonzero(outside)

        if bad.size:
            position = int(bad[0])
            key = int(keys[position])
            raise TargetError(
                f"цель {int(found[position])} не принадлежит полю {self.name(key)} "
                f"с диапазоном [{int(self.value_start[key])}, {int(self.value_end[key])})"
            )

    def to_local(self, key_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Global value ID в локальный индекс кандидата поля.
        """

        self.check_targets(key_ids, values)

        keys = np.asarray(key_ids, dtype=np.int64)

        return np.asarray(values, dtype=np.int64) - self.value_start[keys]

    def to_global(self, key_ids: np.ndarray, local: np.ndarray) -> np.ndarray:

        keys = np.asarray(key_ids, dtype=np.int64)
        index = np.asarray(local, dtype=np.int64)

        if index.size and (int(index.min()) < 0 or bool((index >= self.n_candidates[keys]).any())):
            raise TargetError("локальный индекс вне числа кандидатов поля")

        return index + self.value_start[keys]

    # --------------------------------------------------------

    def as_dict(self) -> dict:
        return {
            "n_keys": self.vocab.n_keys,
            "first_value_id": int(self.first_value_id),
            "trainable": {
                self.name(key_id): {
                    "key_id": int(key_id),
                    "kind": self.kind(key_id),
                    "n_candidates": self.size_of(key_id),
                }
                for key_id in self.trainable_key_ids
            },
            "degenerate": {
                self.name(key_id): {"key_id": int(key_id), "n_candidates": self.size_of(key_id)}
                for key_id in self.degenerate_key_ids
            },
            "rule": (
                "голову получает predictable-поле хотя бы с двумя кандидатами; "
                "поле с одним кандидатом помечается degenerate и в loss и агрегаты не входит"
            ),
        }


# ============================================================
# ГОЛОВА
# ============================================================


@dataclass(frozen=True)
class FieldLogits:
    """
    Logits одного поля и номера его позиций в batch.
    """

    key_id: int
    index: torch.Tensor
    logits: torch.Tensor

    @property
    def n_targets(self) -> int:
        return int(self.index.numel())

    @property
    def n_candidates(self) -> int:
        return int(self.logits.shape[-1])


class MLMHead(nn.Module):
    """
    Общий MLP плюс по линейной голове на поле.
    """

    def __init__(self, config: ModelConfig, table: FieldTable):

        super().__init__()

        self.config = config

        self.key_ids: tuple[int, ...] = tuple(table.trainable_key_ids)

        if not self.key_ids:
            raise ValueError("нет ни одного поля хотя бы с двумя кандидатами: голову строить не из чего")

        activation = nn.GELU() if config.activation == "gelu" else nn.ReLU()

        # Отдельное событие: токен, событие в истории, клиент.
        self.fuse = nn.Sequential(
            nn.Linear(3 * config.d_model, config.d_model),
            activation,
            nn.LayerNorm(config.d_model, eps=config.layer_norm_eps),
        )

        # Все головы существуют до optimizer.
        self.heads = nn.ModuleDict(
            {
                str(key_id): nn.Linear(config.d_model, table.size_of(key_id))
                for key_id in self.key_ids
            }
        )

        # Событие внутри сессии: добавляется его состояние из
        # Session Encoder. Строится ПОСЛЕДНИМ, чтобы при одном
        # seed общие веса обеих структур совпадали.
        self.fuse_session = None

        if config.uses_sessions:
            self.fuse_session = nn.Sequential(
                nn.Linear(4 * config.d_model, config.d_model),
                nn.GELU() if config.activation == "gelu" else nn.ReLU(),
                nn.LayerNorm(config.d_model, eps=config.layer_norm_eps),
            )

    # --------------------------------------------------------

    def forward(
        self,
        h_local: torch.Tensor,
        h_event: torch.Tensor,
        h_usr: torch.Tensor,
        key_ids: torch.Tensor,
        within=None,
    ) -> list[FieldLogits]:
        """
        Позиции группируются по полю, каждая группа идёт в свою голову.

        within это (индексы, состояния внутри сессии) для целей,
        попавших в сессию. У них своя входная проекция из четырёх
        частей; у остальных прежняя из трёх. Головы полей общие:
        различается только то, из чего собран вход.
        """

        if not (h_local.shape == h_event.shape == h_usr.shape):
            raise TargetError(
                f"представления разной формы: {tuple(h_local.shape)}, "
                f"{tuple(h_event.shape)}, {tuple(h_usr.shape)}"
            )

        if h_local.ndim != 2 or h_local.shape[1] != self.config.d_model:
            raise TargetError(f"ожидалось [n, {self.config.d_model}], получено {tuple(h_local.shape)}")

        if key_ids.ndim != 1 or int(key_ids.numel()) != int(h_local.shape[0]):
            raise TargetError(
                f"ключей {tuple(key_ids.shape)}, а представлений {int(h_local.shape[0])}"
            )

        if key_ids.numel() == 0:
            return []

        z = self.fuse(torch.cat([h_local, h_event, h_usr], dim=-1))

        if within is not None:

            if self.fuse_session is None:
                raise TargetError(
                    "переданы состояния внутри сессии, но модель собрана без "
                    "Session Encoder: структуры входа и модели не совпадают"
                )

            index, h_within = within

            if int(h_within.shape[0]) != int(index.numel()):
                raise TargetError(
                    f"состояний внутри сессии {int(h_within.shape[0])}, "
                    f"а индексов {int(index.numel())}"
                )

            grouped = self.fuse_session(
                torch.cat(
                    [h_local[index], h_within, h_event[index], h_usr[index]], dim=-1
                )
            )

            z = z.index_copy(0, index, grouped)

        out: list[FieldLogits] = []

        # unique возвращает возрастающий порядок: состав batch на
        # него не влияет, отчёты воспроизводимы.
        for key in torch.unique(key_ids).tolist():

            name = str(int(key))

            if name not in self.heads:
                raise TargetError(
                    f"у поля {int(key)} нет головы: вырожденные поля должны отсеиваться "
                    "при построении целей, а не доходить сюда"
                )

            head = self.heads[name]

            index = torch.nonzero(key_ids == key, as_tuple=True)[0]

            out.append(FieldLogits(key_id=int(key), index=index, logits=head(z[index])))

        return out

    # --------------------------------------------------------

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_head(config: ModelConfig, table: FieldTable, device=None) -> MLMHead:
    """
    Сборка на CPU, затем перенос: как у энкодеров, чтобы веса не
    зависели от устройства.
    """

    head = MLMHead(config, table)

    return head if device is None else head.to(device)
