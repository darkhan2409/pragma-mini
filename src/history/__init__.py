"""
Этап 12: энкодер истории.

Одна команда на группу:

    python -m src.history.run train|val|test

Вход — вектор анкеты этапа 11, векторы событий этапа 10 и
временные позиции этапа 07. Выход — итоговый вектор клиента в
data/12_history/<group>/history.parquet и веса энкодера.
Обновлённые векторы событий возвращаются вызывающему для будущей
MLM-головы, но на диск не пишутся.

Внимание двунаправленное и охватывает историю одного клиента.
Время входит через TimeRoPE по непрерывным log-секундам.
MLM-голова и обучение — не здесь.
"""

from __future__ import annotations

from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


__all__ = ["FORMAT_VERSION", "IMPLEMENTATION_VERSION", "SCHEMA_VERSION"]
