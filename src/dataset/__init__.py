"""
Сборка обучающего набора поверх замороженного словаря.

Одна команда на группу: `python -m src.dataset.run <group>`.
Закодированная группа превращается в один файл
`data/05_dataset/<group>/samples.parquet`, где строка это клиент:
токены событий и профиля, границы записей, каналы времени,
маска допустимых целей и вес примера.

Маскирование, MLM-голова и обучение сюда не входят.
"""

from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION

__all__ = ["FORMAT_VERSION", "IMPLEMENTATION_VERSION", "SCHEMA_VERSION"]
