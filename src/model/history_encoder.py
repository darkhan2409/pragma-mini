from __future__ import annotations

import torch
import torch.nn as nn

from .batching import BatchError
from .config import ModelConfig
from .transformer import transformer_layers


# ============================================================
# ИДЕЯ
# ============================================================
#
# История клиента это последовательность
#
#     [профиль, событие_1, ..., событие_N]
#
# где каждый элемент уже вектор d_model. Attention
# двусторонний, causal mask нет: цель не предсказание
# следующего события, а представление среза целиком. Разные
# примеры лежат разными строками и друг друга не видят.
#
# У слоёв намеренно висит пустой forward-хук. Причина не в
# логике, а в памяти: fast path nn.TransformerEncoderLayer
# материализует полную матрицу внимания, и на истории в 3637
# событий это 3.2 ГБ вместо 47 МБ. Наличие хука отключает fast
# path ровно у этих слоёв, не трогая глобальный флаг и не мешая
# Event и Profile Encoder, где короткие записи и fast path
# выгоден.
#
# Для диагностики внимание можно ограничить правилом. Правило
# приходит аргументом forward и нигде не запоминается: режим,
# оставшийся на модуле, это тихо испорченный baseline.
# ============================================================


def _keep_slow_path(module, args, output):
    """
    Пустой хук: нужен самим фактом своего существования.
    """

    return None


# ============================================================
# ПРАВИЛА ВНИМАНИЯ
# ============================================================


RULE_FULL = "full"
RULE_EVENTS_ISOLATED = "events_isolated"
RULE_EVENTS_VIA_PROFILE = "events_via_profile"
RULE_PROFILE_ISOLATED = "profile_isolated"
RULE_SELF_ONLY = "self_only"

ATTENTION_RULES: tuple[str, ...] = (
    RULE_FULL,
    RULE_EVENTS_ISOLATED,
    RULE_EVENTS_VIA_PROFILE,
    RULE_PROFILE_ISOLATED,
    RULE_SELF_ONLY,
)

RULE_NOTES: dict[str, str] = {
    RULE_FULL: "все видят всех: режим, в котором модель обучалась",
    RULE_EVENTS_ISOLATED: "событие видит себя и профиль, профиль видит только себя: пути между событиями нет вовсе",
    RULE_EVENTS_VIA_PROFILE: "событие видит себя и профиль, профиль видит всех: события общаются только через профиль",
    RULE_PROFILE_ISOLATED: "событие видит все события, но не профиль; профиль видит только себя",
    RULE_SELF_ONLY: "каждая позиция видит только себя: History Encoder как поэлементное преобразование",
}


def structural_block(rule: str, length: int, device=None) -> torch.Tensor:
    """
    Запреты правила без учёта padding: True это «нельзя смотреть».

    Строка это запрос, столбец это ключ; позиция 0 это профиль.
    Диагональ не блокируется никогда: позиция без единого
    доступного ключа дала бы softmax по пустому множеству.
    """

    if rule not in ATTENTION_RULES:
        raise ValueError(f"неизвестное правило внимания {rule!r}, ожидалось одно из {ATTENTION_RULES}")

    if length < 1:
        raise ValueError("длина последовательности должна быть положительной")

    blocked = torch.zeros(length, length, dtype=torch.bool, device=device)

    if rule == RULE_FULL:
        return blocked

    if rule == RULE_PROFILE_ISOLATED:
        blocked[1:, 0] = True
        blocked[0, 1:] = True
        return blocked

    # Остальные правила строятся от «только себя».
    blocked = ~torch.eye(length, dtype=torch.bool, device=device)

    if rule == RULE_SELF_ONLY:
        return blocked

    # События дополнительно видят профиль.
    blocked[1:, 0] = False

    if rule == RULE_EVENTS_VIA_PROFILE:
        blocked[0, :] = False

    return blocked


class HistoryEncoder(nn.Module):

    def __init__(self, config: ModelConfig):

        super().__init__()

        self.config = config

        self.layers = transformer_layers(config, config.n_history_layers)

        for layer in self.layers:
            layer.register_forward_hook(_keep_slow_path)

        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    # --------------------------------------------------------

    def attention_mask(self, rule: str, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Структурный запрет и padding в одной маске [B·heads, L, L].

        Padding попадает сюда же: слой принимает либо
        src_key_padding_mask, либо src_mask, и смешивать их
        нельзя, иначе часть запретов потерялась бы молча.
        """

        length = int(padding_mask.shape[1])

        structural = structural_block(rule, length, padding_mask.device)

        blocked = structural.unsqueeze(0) | padding_mask.unsqueeze(1)

        # Строка padded-запроса могла остаться полностью
        # закрытой; профиль padding не бывает, поэтому он и
        # служит запасным ключом. Выход этих строк всё равно
        # зануляется после финальной нормы.
        blocked[:, :, 0] &= ~padding_mask

        return blocked.repeat_interleave(self.config.n_heads, dim=0)

    # --------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        attention_rule: str | None = None,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise BatchError(f"HistoryEncoder: ожидался [B, L, d], получено {tuple(x.shape)}")

        if padding_mask.shape != x.shape[:2]:
            raise BatchError(
                f"HistoryEncoder: маска {tuple(padding_mask.shape)} не совпадает с {tuple(x.shape[:2])}"
            )

        if bool(padding_mask.all(dim=1).any()):
            raise BatchError("HistoryEncoder: есть пример целиком из padding, кодировать нечего")

        if attention_rule is None:

            for layer in self.layers:
                x = layer(x, src_key_padding_mask=padding_mask)

        else:

            mask = self.attention_mask(attention_rule, padding_mask)

            for layer in self.layers:
                x = layer(x, src_mask=mask)

        hidden = self.final_norm(x)

        # LayerNorm нулевой строки даёт bias, а не ноль.
        return hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
