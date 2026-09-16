from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

from .activity import ActivityParams
from .amounts import AmountParams
from .calibration import CalibrationParams, CalibrationTarget
from .defects import DefectParams
from .fraud import FraudParams
from .geography import GeographyParams
from .income import IncomeParams
from .lifecycle import LifecycleParams
from .merchants import MerchantParams
from .population import PopulationParams
from .products import ProductParams
from .relationships import RelationshipParams
from .seasonality import SeasonalityParams
from .stress import StressParams
from .traits import TraitParams


# ============================================================
# ПАРАМЕТРЫ ГЕНЕРАТОРА
# ============================================================
#
# Значимые числа лежат здесь, а не в коде. Калибровка по
# банковским агрегатам меняет параметры, а не логику.
#
# Отпечаток параметров входит в ключ кэшей розыгрыша и в
# манифест: датасет нельзя спутать с собранным по другим
# правилам.
# ============================================================


@dataclass(frozen=True)
class GeneratorParams:

    population: PopulationParams = PopulationParams()
    traits: TraitParams = TraitParams()
    lifecycle: LifecycleParams = LifecycleParams()
    income: IncomeParams = IncomeParams()
    stress: StressParams = StressParams()
    relationships: RelationshipParams = RelationshipParams()
    fraud: FraudParams = FraudParams()
    activity: ActivityParams = ActivityParams()
    amounts: AmountParams = AmountParams()
    products: ProductParams = ProductParams()
    geography: GeographyParams = GeographyParams()
    merchants: MerchantParams = MerchantParams()
    seasonality: SeasonalityParams = SeasonalityParams()
    defects: DefectParams = DefectParams()
    calibration: CalibrationParams = CalibrationParams()

    # --------------------------------------------------------

    def as_dict(self) -> dict:
        return {
            section.name: _jsonable(getattr(self, section.name))
            for section in fields(self)
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def with_overrides(self, overrides: dict) -> GeneratorParams:
        """
        Неглубокое слияние: секция заменяется поле в поле.
        """

        known = {section.name for section in fields(self)}

        unknown = set(overrides) - known

        if unknown:
            raise ValueError(f"неизвестные секции параметров: {sorted(unknown)}")

        updates: dict[str, Any] = {}

        for name, values in overrides.items():

            section = getattr(self, name)

            if not isinstance(values, dict):
                raise ValueError(f"секция {name} должна быть объектом")

            allowed = {item.name for item in fields(section)}
            wrong = set(values) - allowed

            if wrong:
                raise ValueError(f"неизвестные параметры в секции {name}: {sorted(wrong)}")

            updates[name] = replace(section, **values)

        return replace(self, **updates)


def _jsonable(value: Any) -> Any:

    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    return value


# ============================================================
# ЗАГРУЗКА И АКТИВНЫЕ ПАРАМЕТРЫ
# ============================================================

DEFAULT = GeneratorParams()

_ACTIVE: dict[str, GeneratorParams] = {"params": DEFAULT}


def load(path: str | Path | None = None) -> GeneratorParams:
    """
    Параметры по умолчанию, при наличии файла с переопределениями
    слитые с ним.
    """

    if path is None:
        return DEFAULT

    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    return DEFAULT.with_overrides(raw)


def activate(params: GeneratorParams) -> None:
    _ACTIVE["params"] = params


def active() -> GeneratorParams:
    return _ACTIVE["params"]


__all__ = [
    "ActivityParams",
    "AmountParams",
    "CalibrationParams",
    "CalibrationTarget",
    "DEFAULT",
    "DefectParams",
    "FraudParams",
    "GeneratorParams",
    "GeographyParams",
    "IncomeParams",
    "LifecycleParams",
    "MerchantParams",
    "PopulationParams",
    "ProductParams",
    "RelationshipParams",
    "SeasonalityParams",
    "StressParams",
    "TraitParams",
    "activate",
    "active",
    "load",
]
