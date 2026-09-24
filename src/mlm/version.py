from __future__ import annotations


# ============================================================
# ВЕРСИИ
# ============================================================
#
# FORMAT_VERSION — контракт головы: контекст из трёх векторов,
# проекция 3d -> d, логиты связанными весами общей таблицы.
#
# IMPLEMENTATION_VERSION — версия кода пакета.
#
# SCHEMA_VERSION — схема targets.parquet.
# ============================================================


FORMAT_VERSION = "1.0.0"

IMPLEMENTATION_VERSION = "3.0.0"

SCHEMA_VERSION = 1


__all__ = ["FORMAT_VERSION", "IMPLEMENTATION_VERSION", "SCHEMA_VERSION"]
