"""
Слой canonical: одна трассируемая запись на каждую строку RAW.

Здесь ничего не выбрасывается и ничего не обучается. Payload
разбирается в типизированные поля, версии и дубли размечаются,
сущности и связи выписываются как наблюдаемые упоминания.
"""

from .registry import FieldEntry, build_registry, registry_as_dict, registry_digest

__all__ = ["FieldEntry", "build_registry", "registry_as_dict", "registry_digest"]
