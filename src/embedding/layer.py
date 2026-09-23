from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .inputs import BatchInput


# ============================================================
# ВХОДНОЙ СЛОЙ
# ============================================================
#
# Вектор обычного токена это сумма трёх слагаемых:
#
#   E[key_id] * scale  +  E[value_id] * scale  +  P[position]
#
# E — ОДНА обучаемая таблица на ключи и значения. Она одна
# потому, что номера ключей и номера значений живут в общем
# пространстве final_vocab.json; отдельного номера для сочетания
# ключа со значением не заводится, и сочетание получается
# сложением. Это же и экономит место: пар ключ-значение
# несравнимо больше, чем ключей и значений по отдельности.
#
# P — фиксированная синусоида от номера куска ВНУТРИ значения.
# Это не место токена в истории и не время события: текст
# «Магнит у дома» приходит несколькими кусками BPE с одним
# ключом, и без P они были бы неразличимой кучей.
#
# Маркеры [EVT] и [USR] записаны выше по конвейеру одним и тем же
# номером в ОБА слота. Складывать их дважды нельзя: получилось бы
# 2 * E[EVT], то есть маркер в другом масштабе, чем всё
# остальное. Поэтому у маркера берётся ровно одно слагаемое и не
# берётся позиция куска: куска у него нет.
#
# [PAD] зануляется маской. Отдельной ветки по номеру у него нет
# намеренно: маска — единственный способ отличить заполнитель от
# настоящего токена, и она же одна отвечает за ноль.
#
# [MASK] и [UNK] — обычные номера со своими строками таблицы.
# Ключ рядом с ними остаётся видимым: предсказывается значение.
#
# Календарь, event_time_log и границы событий сюда не приходят
# вовсе — это вход, а не энкодер события.
# ============================================================


# Знаменатель синусоиды из исходного трансформера.
PERIOD = 10000.0


@dataclass(frozen=True)
class Embedded:
    """
    Выход слоя: события и анкета со своими масками.
    """

    tokens: torch.Tensor        # [B, T, d]
    token_mask: torch.Tensor    # [B, T]
    profile: torch.Tensor       # [B, P, d]
    profile_mask: torch.Tensor  # [B, P]


class InputEmbedding(nn.Module):
    """
    Общая таблица, синусоида кусков и три правила: маркер,
    заполнитель, обычный токен.
    """

    def __init__(self, vocab_size: int, dim: int, seed: int, markers: tuple[int, ...]):

        super().__init__()

        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.seed = int(seed)

        # Синусоида имеет постоянную норму sqrt(d/2): в каждой
        # паре sin^2 + cos^2 = 1. Обучаемая часть разыгрывается
        # с N(0, 1/sqrt(d)), то есть нормой около 1, и без
        # множителя она тонула бы в позиционной. Множитель стоит
        # на ВЫХОДЕ таблицы, а не в самих весах: будущая
        # MLM-голова свяжет логиты с теми же весами, и там нужен
        # обычный масштаб.
        self.scale = math.sqrt(self.dim)

        self.table = nn.Embedding(self.vocab_size, self.dim)

        # Розыгрыш от своего генератора, а не от глобального
        # состояния torch: одинаковый seed обязан давать
        # одинаковый файл весов, чем бы ни занимался процесс
        # рядом.
        generator = torch.Generator().manual_seed(self.seed)

        with torch.no_grad():
            self.table.weight.normal_(
                mean=0.0, std=1.0 / math.sqrt(self.dim), generator=generator
            )

        # Непостоянные буферы: в файл весов попадает ровно одна
        # таблица, а маркеры и лестница частот задаются
        # конструктором и формулой.
        self.register_buffer(
            "marker_ids", torch.tensor(sorted(markers), dtype=torch.int64),
            persistent=False,
        )

        index = torch.arange(self.dim // 2, dtype=torch.float32)

        self.register_buffer(
            "frequency", torch.exp(-math.log(PERIOD) * (2.0 * index / self.dim)),
            persistent=False,
        )

    @property
    def weight(self) -> torch.Tensor:
        """
        Веса общей таблицы. Оставлены наружу намеренно: будущая
        MLM-голова предсказывает значения теми же векторами,
        которыми они поданы на вход.
        """

        return self.table.weight

    def pieces_of(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Синусоида номера куска: [.., d].

        Считается формулой от самого номера, без таблицы: у длины
        значения нет заранее известного предела, и выдумывать его
        ради таблицы незачем.
        """

        angle = positions.unsqueeze(-1).to(self.frequency.dtype) * self.frequency

        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1).flatten(-2)

    def marker_of(self, key_ids: torch.Tensor) -> torch.Tensor:
        """
        Где стоит [EVT] или [USR].
        """

        return torch.isin(key_ids, self.marker_ids)

    def embed(
        self,
        key_ids: torch.Tensor,
        value_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Векторы одной последовательности: [.., d].

        События и анкета проходят здесь одним и тем же кодом:
        разница только в том, какие массивы поданы.
        """

        visible = (~self.marker_of(key_ids)).unsqueeze(-1)

        rest = (self.table(value_ids) * self.scale + self.pieces_of(positions)) * visible

        return (self.table(key_ids) * self.scale + rest) * mask.unsqueeze(-1)

    def forward(self, batch: BatchInput) -> Embedded:
        """
        Батч целиком: события и анкета.
        """

        return Embedded(
            tokens=self.embed(
                batch.key_ids, batch.value_ids, batch.positions, batch.token_mask
            ),
            token_mask=batch.token_mask,
            profile=self.embed(
                batch.profile_key_ids,
                batch.profile_value_ids,
                batch.profile_positions,
                batch.profile_token_mask,
            ),
            profile_mask=batch.profile_token_mask,
        )


__all__ = [
    "PERIOD",
    "Embedded",
    "InputEmbedding",
]
