"""
Этап 6: временные позиции событий и вех анкеты.

Одна команда на группу:

    python -m src.temporal.run train|val|test

Вход — data/05_dataset/<group>/samples.parquet, выход —
data/06_temporal/<group>/temporal.parquet: тот же пример плюс два
канала в сжатых логарифмом секундах — event_time_log, расстояние
до последнего события, и profile_time_log, давность вехи анкеты
до cutoff (ноль у [USR] и Attributes).

Этап ничего не отбирает, не переставляет и не кодирует: он
только добавляет к примеру числовые каналы.
"""

from __future__ import annotations

from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


__all__ = ["FORMAT_VERSION", "IMPLEMENTATION_VERSION", "SCHEMA_VERSION"]
