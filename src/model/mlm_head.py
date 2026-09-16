from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.preprocessing.artifacts import read_json
from src.tokenizer.config import FIELD_VALUE_IDS_FILE, IncompatibleArtifactsError
from src.tokenizer.vocab import N_FIELDS, Vocab

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
    Кандидаты каждого поля и перевод token ↔ local.

    Массивы индексируются **field_id**, а не token ID: поле это
    и есть домен предсказания, и он один и тот же во всех
    режимах словаря. В semantic-режиме два поля делят key token,
    и таблица по токену слила бы их в одну строку.

    Перевод идёт через плотную local_lookup словаря: вычитанием
    смещения он больше не выражается, потому что в shared-режиме
    кандидаты поля не обязаны идти подряд.
    """

    def __init__(self, vocab: Vocab):

        self.vocab = vocab

        self.first_value_id = vocab.first_value_id

        rows = N_FIELDS + 1

        self.n_candidates = np.zeros(rows, dtype=np.int64)
        self.predictable = np.zeros(rows, dtype=bool)

        self._name: dict[int, str] = {}
        self._kind: dict[int, str] = {}

        for entry in vocab.fields:
            self.n_candidates[entry.field_id] = entry.n_values
            self.predictable[entry.field_id] = entry.predictable
            self._name[entry.field_id] = entry.key
            self._kind[entry.field_id] = entry.kind

        self.candidates = vocab.candidates()

        # Поле с одним кандидатом предсказывать нечем.
        self.degenerate = self.predictable & (self.n_candidates < 2)

        self.trainable = self.predictable & (self.n_candidates >= 2)

        self.trainable_field_ids: tuple[int, ...] = tuple(
            int(entry.field_id) for entry in vocab.fields if self.trainable[entry.field_id]
        )

        self.degenerate_field_ids: tuple[int, ...] = tuple(
            int(entry.field_id) for entry in vocab.fields if self.degenerate[entry.field_id]
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

        if len(fields) != vocab.n_fields:
            raise IncompatibleArtifactsError(
                f"{FIELD_VALUE_IDS_FILE}: полей {len(fields)}, а в словаре {vocab.n_fields}"
            )

        for entry in vocab.fields:

            item = fields.get(entry.key)

            if item is None:
                raise IncompatibleArtifactsError(f"{FIELD_VALUE_IDS_FILE}: нет поля {entry.key}")

            if int(item["field_id"]) != entry.field_id:
                raise IncompatibleArtifactsError(
                    f"{FIELD_VALUE_IDS_FILE}: у поля {entry.key} field_id {item['field_id']}, "
                    f"в словаре {entry.field_id}"
                )

            if int(item["key_token_id"]) != entry.key_token_id:
                raise IncompatibleArtifactsError(
                    f"{FIELD_VALUE_IDS_FILE}: у поля {entry.key} key token {item['key_token_id']}, "
                    f"в словаре {entry.key_token_id}"
                )

            if bool(item["predictable"]) != entry.predictable or item["kind"] != entry.kind:
                raise IncompatibleArtifactsError(
                    f"{FIELD_VALUE_IDS_FILE}: описание поля {entry.key} не совпадает со словарём"
                )

            # Порядок значим: локальный индекс это позиция в этом
            # списке, а у numeric он обязан равняться номеру
            # корзины. Сравнение множеств пропустило бы перестановку.
            if list(item["value_ids"]) != list(entry.candidates):
                raise IncompatibleArtifactsError(
                    f"{FIELD_VALUE_IDS_FILE}: кандидаты поля {entry.key} не совпадают со словарём "
                    "по составу или по порядку"
                )

        return FieldTable(vocab)

    # --------------------------------------------------------

    def name(self, field_id: int) -> str:
        return self._name.get(int(field_id), f"field_{int(field_id)}")

    def kind(self, field_id: int) -> str:
        return self._kind.get(int(field_id), "unknown")

    def size_of(self, field_id: int) -> int:
        return int(self.n_candidates[int(field_id)])

    def is_trainable(self, field_id: int) -> bool:
        return bool(self.trainable[int(field_id)])

    def key_token_of(self, field_id: int) -> int:
        return int(self.vocab.key_token_by_field[int(field_id)])

    # --------------------------------------------------------

    def check_targets(self, field_ids: np.ndarray, values: np.ndarray) -> None:
        """
        Каждая цель это настоящее значение своего поля.

        Чужое значение отвергается по таблице кандидатов, а не по
        диапазону: в shared-режиме значение соседнего поля может
        иметь token id внутри «диапазона» этого поля и молча
        обучило бы голову неверной метке.
        """

        fields = np.asarray(field_ids, dtype=np.int64)
        found = np.asarray(values, dtype=np.int64)

        if fields.shape != found.shape:
            raise TargetError(f"полей {fields.shape}, значений {found.shape}")

        if fields.size == 0:
            return

        if int(fields.min()) < 0 or int(fields.max()) >= self.predictable.size:
            raise TargetError("field_id вне пространства полей")

        bad = np.flatnonzero(~self.predictable[fields])

        if bad.size:
            position = int(bad[0])
            raise TargetError(
                f"поле {self.name(int(fields[position]))} не помечено predictable, "
                "его значения не могут быть целями"
            )

        bad = np.flatnonzero(found < self.first_value_id)

        if bad.size:
            position = int(bad[0])
            raise TargetError(
                f"цель {int(found[position])} поля {self.name(int(fields[position]))} это "
                "special-токен, а не значение словаря"
            )

        if int(found.max()) >= self.vocab.size:
            raise TargetError("цель вне словаря")

        local = self.candidates.to_local(fields, found)

        bad = np.flatnonzero(local < 0)

        if bad.size:
            position = int(bad[0])
            field = int(fields[position])
            raise TargetError(
                f"цель {int(found[position])} не входит в {self.size_of(field)} кандидатов "
                f"поля {self.name(field)}"
            )

    def to_local(self, field_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Value token в локальный индекс кандидата поля.
        """

        self.check_targets(field_ids, values)

        return self.candidates.to_local(field_ids, values)

    def to_global(self, field_ids: np.ndarray, local: np.ndarray) -> np.ndarray:

        fields = np.asarray(field_ids, dtype=np.int64)
        index = np.asarray(local, dtype=np.int64)

        if index.size and (int(index.min()) < 0 or bool((index >= self.n_candidates[fields]).any())):
            raise TargetError("локальный индекс вне числа кандидатов поля")

        return self.candidates.to_global(fields, index)

    # --------------------------------------------------------

    def as_dict(self) -> dict:
        return {
            "n_fields": self.vocab.n_fields,
            "n_keys": self.vocab.n_key_tokens,
            "modes": self.vocab.modes,
            "first_value_id": int(self.first_value_id),
            "trainable": {
                self.name(field_id): {
                    "field_id": int(field_id),
                    "key_token_id": self.key_token_of(field_id),
                    "kind": self.kind(field_id),
                    "n_candidates": self.size_of(field_id),
                }
                for field_id in self.trainable_field_ids
            },
            "degenerate": {
                self.name(field_id): {
                    "field_id": int(field_id),
                    "n_candidates": self.size_of(field_id),
                }
                for field_id in self.degenerate_field_ids
            },
            "rule": (
                "голову получает predictable-поле хотя бы с двумя кандидатами; "
                "поле с одним кандидатом помечается degenerate и в loss и агрегаты не входит; "
                "домен предсказания это ПОЛЕ, а не key token"
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

    field_id: int
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

        self.field_ids: tuple[int, ...] = tuple(table.trainable_field_ids)

        if not self.field_ids:
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
                str(field_id): nn.Linear(config.d_model, table.size_of(field_id))
                for field_id in self.field_ids
            }
        )

    # --------------------------------------------------------

    def forward(
        self,
        h_local: torch.Tensor,
        h_event: torch.Tensor,
        h_usr: torch.Tensor,
        field_ids: torch.Tensor,
    ) -> list[FieldLogits]:
        """
        Позиции группируются по полю, каждая группа идёт в свою голову.
        """

        if not (h_local.shape == h_event.shape == h_usr.shape):
            raise TargetError(
                f"представления разной формы: {tuple(h_local.shape)}, "
                f"{tuple(h_event.shape)}, {tuple(h_usr.shape)}"
            )

        if h_local.ndim != 2 or h_local.shape[1] != self.config.d_model:
            raise TargetError(f"ожидалось [n, {self.config.d_model}], получено {tuple(h_local.shape)}")

        if field_ids.ndim != 1 or int(field_ids.numel()) != int(h_local.shape[0]):
            raise TargetError(
                f"полей {tuple(field_ids.shape)}, а представлений {int(h_local.shape[0])}"
            )

        if field_ids.numel() == 0:
            return []

        z = self.fuse(torch.cat([h_local, h_event, h_usr], dim=-1))

        out: list[FieldLogits] = []

        # unique возвращает возрастающий порядок: состав batch на
        # него не влияет, отчёты воспроизводимы.
        for key in torch.unique(field_ids).tolist():

            name = str(int(key))

            if name not in self.heads:
                raise TargetError(
                    f"у поля {int(key)} нет головы: вырожденные поля должны отсеиваться "
                    "при построении целей, а не доходить сюда"
                )

            head = self.heads[name]

            index = torch.nonzero(field_ids == key, as_tuple=True)[0]

            out.append(FieldLogits(field_id=int(key), index=index, logits=head(z[index])))

        return out

    # --------------------------------------------------------

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

