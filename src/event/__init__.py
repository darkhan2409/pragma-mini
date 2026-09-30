"""
Этап 8: энкодер события.

Одна команда на группу:

    python -m src.event.run train|val|test

Вход — выровненные батчи, видимые значения и веса входного слоя
этапа 06. Выход — вектор каждого настоящего события строкой на
событие в data/08_events/<group>/events.parquet и веса самого
энкодера.

Внимание ограничено одним событием: токены одного события не
видят токены другого. History Encoder, TimeRoPE, профиль и
MLM-голова — не здесь.
"""

from __future__ import annotations

from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


__all__ = ["FORMAT_VERSION", "IMPLEMENTATION_VERSION", "SCHEMA_VERSION"]
