"""
Токенизатор V1: смысловые значения → словари → пары числовых ID.

Словарь строится по шагам, и каждый шаг это отдельная команда с
одним видимым файлом в `data/vocab/`: служебные токены, ключи,
категории, числовые диапазоны, BPE и собранный из них
`final_vocab.json`.

Учатся все словари только на train. Кодирование группы (`encode`)
применяет готовый словарь и ничего не дообучает: значение, которого
на train не было, получает [UNK]. Результат ложится в `data/tokenized/<group>/`.
"""

from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION

__all__ = ["FORMAT_VERSION", "IMPLEMENTATION_VERSION", "SCHEMA_VERSION"]
