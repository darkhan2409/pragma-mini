from __future__ import annotations


# ============================================================
# ВЕРСИИ
# ============================================================
#
# FORMAT_VERSION — контракт энкодера анкеты: внимание по всей
# анкете клиента, один вектор из позиции [USR], время токенов
# через TimeRoPE — давность вех Lifelong до cutoff, ноль у [USR]
# и Attributes. Версия 2 — появился временной канал: веса версии
# 1 к энкодеру не подходят.
#
# IMPLEMENTATION_VERSION — версия кода пакета.
#
# SCHEMA_VERSION — схема profiles.parquet.
# ============================================================


FORMAT_VERSION = "2.0.0"

IMPLEMENTATION_VERSION = "2.0.0"

SCHEMA_VERSION = 1


__all__ = ["FORMAT_VERSION", "IMPLEMENTATION_VERSION", "SCHEMA_VERSION"]
