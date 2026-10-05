from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping


# ============================================================
# ИДЕЯ
# ============================================================
#
# Всё, что решает человек, а не данные, живёт здесь: сколько
# блоков трансформера, сколько голов внимания, какой FFN, какой
# dropout, каким seed разыграны веса и сколько событий считается
# за раз.
#
# Длины вектора d здесь НЕТ: она приходит из весов входного слоя
# (data/06_embeddings/<group>/weights.pt). Два числа про одно и
# то же молча разошлись бы.
# ============================================================


DEVICES = ("auto", "cpu", "cuda")


class ConfigError(ValueError):
    """
    Конфигурация энкодера события невозможна.
    """


@dataclass(frozen=True)
class EventConfig:
    """
    Решения человека об энкодере события.
    """

    # Seed розыгрыша весов энкодера. Свой, отдельно от входного
    # слоя: тот разыгран раньше и лежит на диске.
    seed: int = 42

    # Блоков трансформера. Энкодер события — самый глубокий в
    # модели, как в эталоне: 5 блоков против 1 у анкеты и 2 у
    # истории.
    layers: int = 5

    # Голов внимания. d обязана делиться на это число.
    heads: int = 4

    # Ширина FFN. Эталон держит 4 * d.
    feedforward: int = 512

    # Dropout при обучении; при оценке модель стоит в eval.
    dropout: float = 0.1

    # events_per_chunk и device остались от диагностического этапа
    # 08 и обучением не читаются: порцию и устройство обучения задаёт
    # MlmConfig. Поля хранятся, потому что входят в конфиг весов
    # 07_backbone и чекпойнтов, а from_dict строг к ключам.
    events_per_chunk: int = 1024

    device: str = "auto"

    def validate(self) -> None:

        for name in ("layers", "heads", "events_per_chunk"):

            value = getattr(self, name)

            if value < 1:
                raise ConfigError(f"{name} обязан быть положительным, получено {value}")

        if self.feedforward < 1:
            raise ConfigError(f"feedforward обязан быть положительным, получено {self.feedforward}")

        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"dropout обязан лежать в [0, 1), получено {self.dropout}")

        if self.device not in DEVICES:
            raise ConfigError(f"device обязан быть одним из {list(DEVICES)}, получено {self.device!r}")

    def check_dim(self, dim: int) -> None:
        """
        Сверка с длиной вектора, пришедшей из весов этапа 06.
        """

        if dim % self.heads:
            raise ConfigError(
                f"d = {dim} не делится на {self.heads} голов: "
                "каждая голова берёт свою часть вектора"
            )

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "layers": self.layers,
            "heads": self.heads,
            "feedforward": self.feedforward,
            "dropout": self.dropout,
            "events_per_chunk": self.events_per_chunk,
            "device": self.device,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "EventConfig":

        base = EventConfig()

        unknown = set(data) - set(base.as_dict())

        if unknown:
            raise ConfigError(f"неизвестные ключи конфига энкодера события: {sorted(unknown)}")

        config = replace(
            base,
            seed=int(data.get("seed", base.seed)),
            layers=int(data.get("layers", base.layers)),
            heads=int(data.get("heads", base.heads)),
            feedforward=int(data.get("feedforward", base.feedforward)),
            dropout=float(data.get("dropout", base.dropout)),
            events_per_chunk=int(data.get("events_per_chunk", base.events_per_chunk)),
            device=str(data.get("device", base.device)),
        )

        config.validate()

        return config

    @staticmethod
    def load(path: Path | None) -> "EventConfig":

        if path is None:
            config = EventConfig()
            config.validate()
            return config

        return EventConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "DEVICES",
    "ConfigError",
    "EventConfig",
]
