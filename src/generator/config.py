from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path


# ============================================================
# PROJECT
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TOKENIZED_DIR = DATA_DIR / "tokenized"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
RUNS_DIR = DATA_DIR / "runs"


# ============================================================
# RANDOMNESS
# ============================================================

SEED = 42


# ============================================================
# TIME HORIZON
# ============================================================

# 24 месяца истории признаков.
HISTORY_START = datetime(2024, 6, 1)

# Глобальная верхняя граница feature history (момент "как есть").
FEATURE_END = datetime(2026, 6, 1)

# 90 дней строго будущего окна для downstream labels.
LABEL_WINDOW_DAYS = 90

LABEL_END = FEATURE_END + timedelta(days=LABEL_WINDOW_DAYS)


# ============================================================
# SOURCE AVAILABILITY
# ============================================================
#
# В реальном хранилище источники подключались в разные даты:
# витрина коммуникаций и GA4 живут заметно короче, чем
# транзакции и договоры.
#
# События раньше availability_start НЕ существуют в RAW,
# даже если поведение клиента их подразумевает.
# ============================================================

SOURCE_AVAILABILITY: dict[str, datetime] = {
    "profile": HISTORY_START,
    "transactions": HISTORY_START,
    # Реестр договоров старше окна наблюдения: в нём лежат
    # контракты, открытые до начала истории, но не раньше
    # даты миграции реестра.
    "product_events": datetime(2018, 1, 1),
    "app_operations": HISTORY_START,
    "banners": datetime(2024, 8, 1),
    "app_screens": datetime(2024, 12, 16),
    "communications": datetime(2024, 12, 22),
}

SOURCES = tuple(SOURCE_AVAILABILITY)


# ============================================================
# EVENT CONTRACT
# ============================================================
#
#     client_id | ts | event_type | payload
#
# EVENT_TYPES задаёт и множество типов, и приоритет
# детерминированного tie-break при одинаковом ts:
# индекс в кортеже = приоритет (меньше = раньше).
#
# Порядок отражает причинность:
# состояние -> договор -> деньги -> контакт банка ->
# баннер -> экран, открытый по баннеру -> операция на экране.
# ============================================================

EVENT_TYPES = (
    "profile_snapshot",
    "product_event",
    "transaction",
    "communication",
    "banner",
    "app_screen",
    "app_operation",
)

EVENT_TYPE_PRIORITY: dict[str, int] = {
    name: index for index, name in enumerate(EVENT_TYPES)
}

# Источник события -> тип события в timeline.
EVENT_TYPE_BY_SOURCE: dict[str, str] = {
    "profile": "profile_snapshot",
    "product_events": "product_event",
    "transactions": "transaction",
    "communications": "communication",
    "banners": "banner",
    "app_screens": "app_screen",
    "app_operations": "app_operation",
}


# ============================================================
# SEQUENCE BUDGETS
# ============================================================
#
# Два РАЗНЫХ параметра, их нельзя путать:
#
# MAX_TOKENS_PER_EVENT   сколько токенов занимает ОДНО событие
#                        (event_type + поля payload) в Event Encoder
#
# MAX_EVENTS_PER_HISTORY сколько событий History Encoder читает
#                        из ленты одного клиента
#
# Генератор их не применяет к RAW: RAW хранит историю целиком.
# Значения пишутся в manifest, чтобы preprocessing использовал
# те же лимиты и на синтетике, и на реальных данных.
# ============================================================

MAX_TOKENS_PER_EVENT = 12

MAX_EVENTS_PER_HISTORY = 4096


# ============================================================
# PROFILE
# ============================================================
#
# 20 полей профиля. Профиль это СОСТОЯНИЕ as-of, а не событие:
# он строится на конец каждого месяца и читается на cutoff.
# ============================================================

PROFILE_FIELDS = (
    "age",
    "gender",
    "family_status",
    "children",
    "education",
    "region",
    "housing_type",
    "pensioner",
    "income_type",
    "declared_income",
    "industry",
    "salary_day",
    "relationship_months",
    "contracts_count",
    "active_contracts",
    "holds_credit_card",
    "holds_debit_card",
    "holds_deposit",
    "credit_limit",
    "credit_utilization",
)

# Поля, меняющиеся во времени: только они уходят в timeline
# как компактный profile_snapshot. Полный срез живёт в profile.parquet.
PROFILE_DYNAMIC_FIELDS = (
    "declared_income",
    "relationship_months",
    "contracts_count",
    "active_contracts",
    "holds_credit_card",
    "holds_debit_card",
    "holds_deposit",
    "credit_limit",
    "credit_utilization",
)


# ============================================================
# DATASET PRESETS
# ============================================================

PRESETS = {
    "smoke": 100,
    "dev": 10_000,
    "eval": 50_000,
}
