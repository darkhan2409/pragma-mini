from __future__ import annotations

from dataclasses import dataclass

from src.tokenizer.config import EVT_ID, N_SPECIAL, PAD_ID, USR_ID
from src.tokenizer.semantics import (
    DEFAULT_KEY_MODE,
    DEFAULT_VALUE_MODE,
    check_mode,
    is_baseline,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Размер словаря не задаётся руками: он приходит из проверенных
# artifacts tokenizer. ID не перенумеровываются, поэтому
# embedding-таблица индексируется теми же числами, что лежат в
# tokenized-датасетах, а строка 0 это [PAD].
#
# max_position_embeddings это размер таблицы позиций, а НЕ
# лимит события. Лимиты tokenizer здесь не применяются: выход
# за размер таблицы это ошибка, а не обрезка.
#
# Умолчания здесь НЕ меняются при появлении новой архитектуры.
# Размеры нового запуска задаются явно (см. ARCHITECTURE ниже
# и поля TrainConfig), потому что смена умолчания молча меняла
# бы архитектуру всех прежних вызовов и делала бы уже лежащие
# на диске checkpoints незагружаемыми.
# ============================================================


ACTIVATIONS: tuple[str, ...] = ("gelu", "relu")


# ------------------------------------------------------------
# СТРУКТУРА ИСТОРИИ
# ------------------------------------------------------------
#
# История это [профиль, событие, событие, ...]: каждое событие
# занимает свою позицию.
# ------------------------------------------------------------


# ------------------------------------------------------------
# ВРЕМЯ ИСТОРИИ
# ------------------------------------------------------------
#
# Время живёт во внимании: q и k поворачиваются на угол,
# пропорциональный squash(часы до последнего элемента истории).
# Разность углов пары и есть их относительное время, к самому
# вектору события не прибавляется ничего. Календарь события и
# простой клиента приходят отдельными признаками.
#
# Так сделано не из вкуса: у аддитивного кодирования, которое пробовали до этого,
# норма временного слагаемого на обученном checkpoint была
# 26 (сутки) .. 143 (два года) против нормы вектора события 9,
# то есть содержание события было малой добавкой к направлению
# «возраст».
# ------------------------------------------------------------

# База частот RoPE: та же, что у синусоидального кодирования.
ROPE_BASE = 10_000.0


@dataclass(frozen=True)
class ModelConfig:

    vocab_size: int

    d_model: int = 64
    n_heads: int = 4

    n_event_layers: int = 2
    n_profile_layers: int = 2
    n_history_layers: int = 2

    dim_feedforward: int = 256
    dropout: float = 0.1
    activation: str = "gelu"

    max_position_embeddings: int = 256

    layer_norm_eps: float = 1e-5

    pad_id: int = PAD_ID
    evt_id: int = EVT_ID
    usr_id: int = USR_ID

    rope_base: float = ROPE_BASE

    # Режим словаря. На архитектуру он влияет ровно одним:
    # размером таблицы токенов. Но он обязан ехать в checkpoint,
    # потому что модель, обученная на склеенном словаре, в
    # baseline-словаре означает другое.
    key_mode: str = DEFAULT_KEY_MODE
    categorical_value_mode: str = DEFAULT_VALUE_MODE

    def __post_init__(self) -> None:

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} не делится на n_heads={self.n_heads}: "
                "голова внимания получила бы дробную размерность"
            )

        for name in ("vocab_size", "d_model", "n_heads", "dim_feedforward", "max_position_embeddings"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} должен быть положительным, получено {value}")

        for name in ("n_event_layers", "n_profile_layers", "n_history_layers"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} должен быть не меньше единицы")

        check_mode(self.key_mode, self.categorical_value_mode)

        # RoPE поворачивает ПАРЫ измерений головы.
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim={self.head_dim} нечётный: RoPE поворачивает пары измерений"
            )

        if self.rope_base <= 1.0:
            raise ValueError(f"rope_base должен быть больше единицы, получено {self.rope_base}")

        if self.vocab_size <= N_SPECIAL:
            raise ValueError(
                f"vocab_size={self.vocab_size} не больше числа special-токенов {N_SPECIAL}: "
                "словарь не похож на словарь tokenizer"
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout должен лежать в [0, 1), получено {self.dropout}")

        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation должен быть одним из {ACTIVATIONS}, получено {self.activation!r}")

        if not 0 <= self.pad_id < self.vocab_size:
            raise ValueError("pad_id вне словаря")

    # --------------------------------------------------------

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def uses_shared_vocab(self) -> bool:
        return not is_baseline(self.key_mode, self.categorical_value_mode)

    def as_dict(self) -> dict:
        """
        Отпечаток архитектуры для checkpoint.
        """

        data = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "head_dim": self.head_dim,
            "n_event_layers": self.n_event_layers,
            "n_profile_layers": self.n_profile_layers,
            "n_history_layers": self.n_history_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "activation": self.activation,
            "max_position_embeddings": self.max_position_embeddings,
            "layer_norm_eps": self.layer_norm_eps,
            "special_ids": {"pad": self.pad_id, "evt": self.evt_id, "usr": self.usr_id},
            "rope_base": self.rope_base,
            "key_mode": self.key_mode,
            "categorical_value_mode": self.categorical_value_mode,
        }

        return data


# ============================================================
# АРХИТЕКТУРА ЗАПУСКА
# ============================================================
#
# Явный набор, а не умолчания полей: запуск называет свои
# размеры сам, и они видны одним куском.
# ============================================================

ARCHITECTURE: dict = {
    "d_model": 128,
    "n_heads": 4,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "activation": "gelu",
    "n_profile_layers": 1,
    "n_event_layers": 3,
    "n_history_layers": 2,
}


def config_from_tokenizer(tokenizer, **overrides) -> ModelConfig:
    """
    Размер словаря и его режим берутся из tokenizer_config,
    который уже проверен по хэшам при загрузке.

    Режим приходит из словаря, а не из флага: модель не имеет
    права объявить себя semantic, читая baseline-словарь.
    """

    size = tokenizer.config["id_layout"]["size"]

    if size != tokenizer.vocab.size:
        raise ValueError("размер словаря в config не совпадает с загруженным словарём")

    specials = tokenizer.config["special_tokens"]["ids"]

    modes = {
        "key_mode": tokenizer.vocab.key_mode,
        "categorical_value_mode": tokenizer.vocab.value_mode,
    }

    for name, value in modes.items():
        if name in overrides and overrides[name] is not None and overrides[name] != value:
            raise ValueError(
                f"запрошен {name}={overrides[name]!r}, а словарь собран как {value!r}: "
                "режим берётся из словаря, а не из флага"
            )

    return ModelConfig(
        vocab_size=int(size),
        pad_id=int(specials["[PAD]"]),
        evt_id=int(specials["[EVT]"]),
        usr_id=int(specials["[USR]"]),
        **{**overrides, **modes},
    )
