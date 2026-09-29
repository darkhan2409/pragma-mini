"""
Маскирование значений — не этап, а розыгрыш при чтении.

Маска на диске не хранится. Читатель входа модели разыгрывает её для
каждого клиента: choose решает, что скрыть и что предсказать, apply
заменяет выбранные значения на [MASK] или [UNK] и ставит метки.
Розыгрыш ключуется seed, группой и клиентом (choose.py), поэтому
маска val и test одна и та же при каждом чтении.

Саму модель и обучение пакет не содержит.
"""

from __future__ import annotations

from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION


__all__ = ["FORMAT_VERSION", "IMPLEMENTATION_VERSION"]
