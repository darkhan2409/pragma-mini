from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

from src.history.encoder import Block, TimeRoPE


# ============================================================
# ЭНКОДЕР АНКЕТЫ
# ============================================================
#
# Анкета клиента на cutoff T — две части в одной
# последовательности:
#
#   [USR]        якорь, время 0;
#   Attributes   поля анкеты, состояние на T, время 0;
#   Lifelong     датированные вехи строго раньше T, время —
#                давность вехи до T (profile_time_log).
#
# Внимание двунаправленное и охватывает ВСЮ анкету одного
# клиента: полей мало, порядка между ними нет, и каждое вправе
# смотреть на каждое. Причинной маски нет.
#
# Время входит через TimeRoPE — тот же поворот Q и K, что в
# энкодере истории, и тот же блок: Block и TimeRoPE берутся из
# src/history/encoder.py без изменений, веса у анкеты свои.
# Поворот относительный: внимание между вехой и полем зависит от
# давности вехи до T. Нулевой угол у [USR] и Attributes ничего
# не поворачивает, поэтому без вех энкодер ведёт себя как
# обычный. Недатированное поле и веху непосредственно перед T
# различает ключ, а не время: искусственного сдвига времени нет.
#
# Маркер [USR] уже лежит первым токеном анкеты — его записал тот
# же set_lead, что ставит [EVT] событию. Второго маркера не
# добавляется, и вектор клиента читается из колонки 0.
#
# Клиенты разделены строками батча: внимание не выходит за
# строку, поэтому анкета одного клиента не видит анкету другого
# по построению.
#
# Событий, календаря и event_time этот слой не видит: ему их
# просто не подают.
#
# Заполнитель закрыт маской внимания. Занулять его выход, как
# делает энкодер события, здесь не нужно: наружу идёт только
# колонка 0, а она у каждого клиента настоящая — анкета без
# маркера отвергается ещё датасетом.
# ============================================================


class ProfileEncoder(nn.Module):
    """
    Блоки трансформера по анкете клиента с временем вех, вектор из
    позиции [USR].
    """

    def __init__(
        self,
        dim: int,
        layers: int,
        heads: int,
        feedforward: int,
        dropout: float,
        rope_base: float,
        seed: int,
    ):

        super().__init__()

        self.dim = int(dim)

        # Веса разыгрываются от известного состояния и не трогают
        # чужое: глобальное состояние сохраняется, подменяется и
        # возвращается назад. Тот же приём есть в энкодерах
        # события и истории; повтор намеренный — этапы держатся
        # отдельно.
        with _seeded(seed):

            # Слои поштучно: иначе все блоки стартовали бы с
            # одинаковых весов.
            self.layers = nn.ModuleList(
                [Block(self.dim, heads, feedforward, dropout) for _ in range(layers)]
            )

            self.norm = nn.LayerNorm(self.dim)

            self.rope = TimeRoPE(self.dim // heads, base=rope_base)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        [B, P, d], время [B, P] и маска [B, P] (True — настоящий
        токен) -> [B, d]: один вектор на клиента.
        """

        cos, sin = self.rope.angles(positions)

        # Маска, в которой всё настоящее, внимание не меняет, а
        # быстрое ядро SDPA с маской недоступно: её не передают.
        keys = None

        if mask is not None and not bool(mask.all()):
            keys = mask[:, None, None, :]

        x = tokens

        for layer in self.layers:
            x = layer(x, self.rope, cos, sin, keys)

        x = self.norm(x)

        # Маркер [USR] стоит первым токеном анкеты всегда, и
        # колонка 0 настоящая даже у анкеты из одного маркера.
        return x[:, 0]


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


__all__ = ["ProfileEncoder"]
