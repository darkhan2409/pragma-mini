from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from src.embedding.inputs import CALENDAR_PER_EVENT


# ============================================================
# ЭНКОДЕР СОБЫТИЯ
# ============================================================
#
# Событие это несколько токенов ключ-значение, и внимание здесь
# ограничено ОДНИМ событием: токены одного события не видят
# токены другого. Изоляция даётся тем, что событие приходит
# отдельной строкой, а её хвост закрыт src_key_padding_mask.
#
# Маркер [EVT] уже лежит первым токеном события — его записал
# этап кодирования, и в event_lengths он посчитан. Второго
# маркера не добавляется, и вектор события читается из колонки 0.
#
# Позиционного кодирования по полям здесь НЕТ: внутри события у
# полей нет осмысленного порядка, а порядок кусков BPE внутри
# значения уже несёт синусоида входного слоя. Эталон тоже держит
# энкодер события без RoPE.
#
# Время события сюда не приходит: event_time_log это TimeRoPE
# уровнем выше, в History Encoder.
#
# Календарь добавляется слагаемым к ВЕКТОРУ СОБЫТИЯ, после
# энкодера. Он описывает событие целиком, а не поле внутри него;
# добавленный к токенам, он попал бы в контекст каждого значения
# и подсказывал бы будущей MLM-голове время через любое поле.
#
# Про масштаб. Прежняя модель добавляла к вектору события
# ВРЕМЯ — 8·log1p(Δt/8) через Linear, — и утонула: норма вектора
# события была около 9, а временного члена 26 на часе, 135 на
# годе. Здесь такого не случится само собой: вход календаря лежит
# на трёх единичных окружностях и по модулю ограничен — в отличие
# от лог-времени, которое росло без предела.
# ============================================================


@dataclass(frozen=True)
class Encoded:
    """
    Выход энкодера по одной порции событий.
    """

    tokens: torch.Tensor   # [n, L, d] контекстные векторы токенов, заполнитель занулён
    dated: torch.Tensor    # [n, d] вектор события: из [EVT] и с прибавленным календарём


class EventEncoder(nn.Module):
    """
    Блоки трансформера внутри события плюс проекция календаря.
    """

    def __init__(
        self,
        dim: int,
        layers: int,
        heads: int,
        feedforward: int,
        dropout: float,
        seed: int,
    ):

        super().__init__()

        self.dim = int(dim)

        # Веса разыгрываются от известного состояния и не трогают
        # чужое: torch не даёт передать generator внутрь
        # TransformerEncoderLayer, поэтому глобальное состояние
        # сохраняется, подменяется и возвращается назад.
        with _seeded(seed):

            # Слои собираются поштучно. nn.TransformerEncoder
            # копирует ОДИН слой, и тогда все блоки стартовали бы
            # с одинаковых весов.
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=self.dim,
                        nhead=heads,
                        dim_feedforward=feedforward,
                        dropout=dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(layers)
                ]
            )

            self.norm = nn.LayerNorm(self.dim)

            # Шесть чисел календаря в размерность модели. Две
            # линейные с GELU между ними — как в эталоне. Больше
            # нелинейности не нужно: периодичность за неё уже
            # сделал препроцессинг, положив час, день недели и
            # день месяца на окружности.
            self.calendar = nn.Sequential(
                nn.Linear(CALENDAR_PER_EVENT, self.dim),
                nn.GELU(),
                nn.Linear(self.dim, self.dim),
            )

    def forward(
        self,
        tokens: torch.Tensor,
        pad: torch.Tensor,
        calendar: torch.Tensor,
    ) -> Encoded:
        """
        Одна порция событий: [n, L, d] -> векторы токенов и события.
        """

        alive = (~pad).unsqueeze(-1)

        x = tokens

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=pad)

        x = self.norm(x)

        # Заполнитель зануляется ЕЩЁ РАЗ, уже после LayerNorm:
        # нулевая строка после нормировки возвращает не ноль, а
        # bias слоя. У прежней модели он был измерен и равнялся
        # 2.46.
        x = x * alive

        # Маркер [EVT] стоит первым токеном события всегда, и
        # колонка 0 настоящая даже у события из одного маркера.
        event = x[:, 0]

        # Наружу идёт только итог: вектор до календаря нигде не
        # нужен, а промежуточное значение видно строкой выше.
        return Encoded(tokens=x, dated=event + self.calendar(calendar))


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
    "Encoded",
    "EventEncoder",
]
