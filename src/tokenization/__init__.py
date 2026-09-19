"""
Токенизатор V1: смысловые значения → словари → пары числовых ID.

Вход один — смысловой слой препроцессинга (`semantic_as_of`).
Fit только на разрешённом train до fit_end, transform ничего не
дообучает.

Прежний пакет `src.tokenizer` удалён: он описывал контракт RAW
v3 и не импортировался уже ничем. Чем новый формат отличается от
него и что обязан сделать будущий потребитель — в
`compatibility_report.md`.
"""

from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION

__all__ = ["FORMAT_VERSION", "IMPLEMENTATION_VERSION", "SCHEMA_VERSION"]
