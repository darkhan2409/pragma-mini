"""
Этап 6: временные позиции событий.

Одна команда на группу:

    python -m src.temporal.run train|val|test

Вход — data/05_dataset/<group>/samples.parquet, выход —
data/06_temporal/<group>/temporal.parquet: тот же пример плюс
event_time_log, расстояние до последнего события в сжатых
логарифмом секундах.

Этап ничего не отбирает, не переставляет и не кодирует: он
только добавляет к примеру один числовой канал.
"""

from __future__ import annotations

from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


__all__ = ["FORMAT_VERSION", "IMPLEMENTATION_VERSION", "SCHEMA_VERSION"]
