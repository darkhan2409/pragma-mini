from __future__ import annotations


# ============================================================
# ВЕРСИИ
# ============================================================
#
# FORMAT_VERSION — контракт энкодера анкеты: внимание по всей
# анкете клиента, один вектор из позиции [USR], временного канала
# нет. Его читает будущий History Encoder.
#
# IMPLEMENTATION_VERSION — версия кода пакета.
#
# SCHEMA_VERSION — схема profiles.parquet.
# ============================================================


FORMAT_VERSION = "1.0.0"

IMPLEMENTATION_VERSION = "1.1.0"

SCHEMA_VERSION = 1


__all__ = ["FORMAT_VERSION", "IMPLEMENTATION_VERSION", "SCHEMA_VERSION"]
