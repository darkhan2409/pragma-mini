from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping


# ============================================================
# ИДЕЯ
# ============================================================
#
# Всё, что решает человек: глубина и ширина энкодера, основание
# лестницы частот RoPE и seed.
#
# Длины вектора d здесь нет: она приходит из весов входного слоя
# (data/06_embeddings/<group>/weights.pt).
#
# Капа на длину истории здесь нет: её ограничивает набор 05
# (MAX_EVENTS, MAX_TOKENS в src/dataset/settings.py), а внимание
# по оставшейся истории полное и точное.
# ============================================================


DEVICES = ("auto", "cpu", "cuda")


class ConfigError(ValueError):
    """
    Конфигурация энкодера истории невозможна.
    """


@dataclass(frozen=True)
class HistoryConfig:
    """
    Решения человека об энкодере истории.
    """

    # Seed розыгрыша весов энкодера.
    seed: int = 42

    # Блоков трансформера. В эталоне depth_history 1/2/6/18
    # против depth_event 2/5/16/45: история глубже анкеты, но
    # мельче события.
    layers: int = 2

    heads: int = 4

    # Ширина FFN. Эталон держит 4 * d.
    feedforward: int = 512

    # Dropout при обучении; при оценке модель стоит в eval.
    dropout: float = 0.1

    # Основание лестницы частот TimeRoPE. В эталоне помечено как
    # догадка; проверено, что при d/heads = 32 временная ось
    # живая — косинусная близость на шести месяцах 0.149 при
    # пороге 0.85.
    rope_base: float = 10000.0

    # Осталось от диагностического этапа 10 и обучением не читается:
    # устройство обучения задаёт MlmConfig. Хранится, потому что
    # входит в конфиг весов 07_backbone и чекпойнтов, а from_dict
    # строг к ключам.
    device: str = "auto"

    def validate(self) -> None:

        for name in ("layers", "heads", "feedforward"):

            value = getattr(self, name)

            if value < 1:
                raise ConfigError(f"{name} обязан быть положительным, получено {value}")

        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"dropout обязан лежать в [0, 1), получено {self.dropout}")

        if self.rope_base <= 1.0:
            raise ConfigError(f"rope_base обязан быть больше единицы, получено {self.rope_base}")

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

        if (dim // self.heads) % 2:
            raise ConfigError(
                f"на голову приходится {dim // self.heads} чисел, а TimeRoPE "
                "поворачивает их парами и требует чётности"
            )

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "layers": self.layers,
            "heads": self.heads,
            "feedforward": self.feedforward,
            "dropout": self.dropout,
            "rope_base": self.rope_base,
            "device": self.device,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "HistoryConfig":

        base = HistoryConfig()

        unknown = set(data) - set(base.as_dict())

        if unknown:
            raise ConfigError(f"неизвестные ключи конфига энкодера истории: {sorted(unknown)}")

        config = replace(
            base,
            seed=int(data.get("seed", base.seed)),
            layers=int(data.get("layers", base.layers)),
            heads=int(data.get("heads", base.heads)),
            feedforward=int(data.get("feedforward", base.feedforward)),
            dropout=float(data.get("dropout", base.dropout)),
            rope_base=float(data.get("rope_base", base.rope_base)),
            device=str(data.get("device", base.device)),
        )

        config.validate()

        return config

    @staticmethod
    def load(path: Path | None) -> "HistoryConfig":

        if path is None:
            config = HistoryConfig()
            config.validate()
            return config

        return HistoryConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "DEVICES",
    "ConfigError",
    "HistoryConfig",
]
