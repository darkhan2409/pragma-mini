from __future__ import annotations

from dataclasses import dataclass

from src.tokenizer.config import EVT_ID, N_SPECIAL, PAD_ID, USR_ID


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
# Размеры нового запуска задаются явно (см. ARCHITECTURES ниже
# и поля TrainConfig), потому что смена умолчания молча меняла
# бы архитектуру всех прежних вызовов и делала бы уже лежащие
# на диске checkpoints незагружаемыми.
# ============================================================


ACTIVATIONS: tuple[str, ...] = ("gelu", "relu")


# ------------------------------------------------------------
# СТРУКТУРА ИСТОРИИ
# ------------------------------------------------------------
#
# event    прежняя лента: [профиль, событие, событие, ...]
# session  app-события одной сессии сначала сворачиваются
#          Session Encoder в один вектор
# ------------------------------------------------------------

STRUCTURE_EVENT = "event"
STRUCTURE_SESSION = "session"

STRUCTURES: tuple[str, ...] = (STRUCTURE_EVENT, STRUCTURE_SESSION)

# Паузы внутри сессии измеряются минутами, а не часами:
# 8·log1p(t/8) от 0.001 ч это 0.001, на три порядка меньше
# самих векторов событий, и признак был бы мёртвым.
SESSION_GAP_UNIT = "minutes"


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

    # Структура истории и число слоёв Session Encoder.
    # Умолчание event: без него каждое существующее место
    # построения конфига молча переключилось бы на сессии.
    structure: str = STRUCTURE_EVENT
    n_session_layers: int = 1

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

        for name in ("n_event_layers", "n_profile_layers", "n_history_layers", "n_session_layers"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} должен быть не меньше единицы")

        if self.structure not in STRUCTURES:
            raise ValueError(f"structure должен быть одним из {STRUCTURES}, получено {self.structure!r}")

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
    def uses_sessions(self) -> bool:
        return self.structure == STRUCTURE_SESSION

    def as_dict(self) -> dict:
        """
        Отпечаток архитектуры для checkpoint.

        n_session_layers и session_gap_unit пишутся ТОЛЬКО в
        режиме сессий. Иначе будущая смена их умолчания сделала
        бы несовместимыми event-checkpoints, которые про Session
        Encoder ничего не знают и знать не должны.
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
            "structure": self.structure,
        }

        if self.uses_sessions:
            data["n_session_layers"] = self.n_session_layers
            data["session_gap_unit"] = SESSION_GAP_UNIT

        return data


# ============================================================
# АРХИТЕКТУРА НОВОГО ЗАПУСКА
# ============================================================
#
# Явные наборы, а не новые умолчания: старые checkpoints
# продолжают собираться из своей исходной архитектуры, а новый
# запуск называет свои размеры сам.
#
# Оба набора одинаковы по размерам и отличаются только
# структурой: сравнение режимов должно идти при равных d_model,
# ширине FFN и числе слоёв, иначе оно сравнивало бы не то.
# ============================================================

SESSION_ARCHITECTURE: dict = {
    "d_model": 128,
    "n_heads": 4,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "activation": "gelu",
    "n_profile_layers": 1,
    "n_event_layers": 3,
    "n_session_layers": 1,
    "n_history_layers": 2,
    "structure": STRUCTURE_SESSION,
}

EVENT_ARCHITECTURE: dict = {**SESSION_ARCHITECTURE, "structure": STRUCTURE_EVENT}

ARCHITECTURES: dict[str, dict] = {
    STRUCTURE_EVENT: EVENT_ARCHITECTURE,
    STRUCTURE_SESSION: SESSION_ARCHITECTURE,
}


def config_from_tokenizer(tokenizer, **overrides) -> ModelConfig:
    """
    Размер словаря берётся из tokenizer_config, который уже
    проверен по хэшам при загрузке.
    """

    size = tokenizer.config["id_layout"]["size"]

    if size != tokenizer.vocab.size:
        raise ValueError("размер словаря в config не совпадает с загруженным словарём")

    specials = tokenizer.config["special_tokens"]["ids"]

    return ModelConfig(
        vocab_size=int(size),
        pad_id=int(specials["[PAD]"]),
        evt_id=int(specials["[EVT]"]),
        usr_id=int(specials["[USR]"]),
        **overrides,
    )
