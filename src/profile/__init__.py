"""
Этап 11: энкодер анкеты.

Одна команда на группу:

    python -m src.profile.run train|val|test

Вход — токены анкеты из выровненных батчей с их временем и веса
входного слоя этапа 09. Выход — один вектор на клиента в
data/11_profiles/<group>/profiles.parquet и веса самого
энкодера.

Анкета это Attributes на cutoff (время 0) и вехи Lifelong раньше
него (время — давность вехи до cutoff, через TimeRoPE). Вектор
клиента это выход позиции [USR]. Событий и календаря этот этап
не видит вовсе.
"""

from __future__ import annotations

from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


__all__ = ["FORMAT_VERSION", "IMPLEMENTATION_VERSION", "SCHEMA_VERSION"]
