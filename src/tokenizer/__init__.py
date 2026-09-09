"""
Tokenizer и runtime masker поверх preprocessing.

Слой превращает типизированные события в тройки
(key_id, value_id, field_position) и задаёт контракт входа для
будущей модели. Энкодеров и обучения здесь нет.
"""
