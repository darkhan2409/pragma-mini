from __future__ import annotations


# ============================================================
# ВЕРСИИ
# ============================================================
#
# FORMAT_VERSION — контракт энкодера истории: последовательность
# [z_a, события], TimeRoPE по непрерывному времени, два выхода.
#
# IMPLEMENTATION_VERSION — версия кода пакета.
#
# SCHEMA_VERSION — схема history.parquet. Версия 2: в файле
# остался только итоговый вектор клиента; векторы событий живут
# в памяти и уходят MLM-голове.
# ============================================================


FORMAT_VERSION = "2.0.0"

IMPLEMENTATION_VERSION = "2.2.0"

SCHEMA_VERSION = 2


__all__ = ["FORMAT_VERSION", "IMPLEMENTATION_VERSION", "SCHEMA_VERSION"]
