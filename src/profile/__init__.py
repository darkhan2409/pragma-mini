"""
Энкодер анкеты: внимание по всей анкете клиента, вектор анкеты из
позиции [USR]. Анкета — Attributes на cutoff (время 0) и вехи
Lifelong раньше него (давность вехи до cutoff, через TimeRoPE).

Часть модели (src/mlm/model.py); начальные веса разыгрывает
python -m src.mlm.init_backbone в data/07_backbone/profile.pt.
"""
