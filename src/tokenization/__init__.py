"""
Токенизатор V1: смысловые значения → словари → пары числовых ID.

Словарь строится по шагам, и каждый шаг это отдельная команда с
одним видимым результатом в `data/tokenizer/`: ключи, категории,
числовые диапазоны, BPE и собранный из них `tokenizer.json`.

Учатся все словари только на `data/preprocessed/train`.
Кодирование группы (`encode`) применяет готовый словарь и ничего
не дообучает: значение, которого на train не было, получает
специальный токен.
"""

from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION

__all__ = ["FORMAT_VERSION", "IMPLEMENTATION_VERSION", "SCHEMA_VERSION"]
