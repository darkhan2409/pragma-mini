from __future__ import annotations

from ..categories import CATEGORIES


# ============================================================
# ИДЕЯ V2
# ============================================================
#
# Схемы, значения, горизонт, availability, as-of, шум и метка
# те же, что в v1. Меняются ПРАВИЛА: что зависит от чего.
#
# Здесь собраны только константы. Никакой логики.
# ============================================================


# ------------------------------------------------------------
# СЧЕТА ВМЕСТО СЛУЧАЙНЫХ ПЛАТЕЖЕЙ
# ------------------------------------------------------------
#
# В v1 квартплата, связь и госплатежи разыгрывались как
# обычные покупки: у клиента не было ни своего дня оплаты,
# ни повторяющейся суммы. В v2 эти группы полностью уходят
# из случайной смеси, и вместо них у клиента есть расписание
# счетов. Слот покупки, выпавший на такую группу, гасится:
# иначе связанные платежи просто добавили бы объём.
# ------------------------------------------------------------

BILL_GROUPS: tuple[str, ...] = ("utilities", "telecom", "government")

BILL_MCC_GROUP: dict[str, str] = {
    "4900": "utilities",
    "4814": "telecom",
    "4899": "telecom",
    "9222": "government",
    "9311": "government",
    "9399": "government",
}

# kind -> (mcc, базовая сумма, log_sigma)
BILL_KINDS: dict[str, tuple[str, int, float]] = {
    "utility": ("4900", 12_000, 0.30),
    "mobile": ("4814", 3_500, 0.20),
    "internet": ("4899", 6_000, 0.12),
    "fine": ("9222", 15_000, 0.55),
    "tax": ("9311", 30_000, 0.70),
    "service": ("9399", 8_000, 0.60),
}

# Не все счета клиента проходят через этот банк: у малоактивного
# клиента их видно меньше. Это и держит объём платежей на уровне
# v1, где доля таких покупок была пропорциональна активности.
BILL_BANK_SHARE = (0.20, 0.95)

# Доля клиентов с домашним интернетом отдельным счётом.
INTERNET_SHARE = 0.70

# Второй коммунальный счёт (свет и вода отдельными квитанциями).
SECOND_UTILITY_SHARE = 0.45

# Пуассоновские редкие счета в месяц.
FINE_RATE_PER_MONTH = 0.22
SERVICE_RATE_PER_MONTH = 0.15

# Налог: раз или два в год, в фиксированные месяцы клиента.
TAX_PER_YEAR = (1, 2)

# Смена тарифа связи за горизонт.
PLAN_CHANGE_SHARE = 0.35

# Сколько дней счёт остаётся к оплате в приложении, прежде чем
# клиент заплатит его мимо. Трёх дней мало: при одной сессии
# в день большая часть счетов не успевала попасть в приложение.
BILL_APP_WINDOW_DAYS = 10

# Задержка связанной транзакции после успешной операции.
LINKED_DELAY_SECONDS = (5, 60)


# ------------------------------------------------------------
# ОПЕРАЦИИ, ПОРОЖДАЮЩИЕ ТРАНЗАКЦИЮ
# ------------------------------------------------------------

OPERATION_MCC: dict[str, str] = {
    "pay_utility": "4900",
    "pay_mobile": "4814",
    "pay_internet": "4899",
    "pay_fine": "9222",
    "pay_tax": "9311",
    "market_order": "5999",
}

BILL_KIND_OPERATION: dict[str, str] = {
    "utility": "pay_utility",
    "mobile": "pay_mobile",
    "internet": "pay_internet",
    "fine": "pay_fine",
    "tax": "pay_tax",
    "service": "pay_tax",
}

# Обратное соответствие: гасить надо тот счёт, который клиент
# реально оплатил, а не тот, за которым изначально пришёл.
OPERATION_BILL_KINDS: dict[str, tuple[str, ...]] = {
    "pay_utility": ("utility",),
    "pay_mobile": ("mobile",),
    "pay_internet": ("internet",),
    "pay_fine": ("fine",),
    "pay_tax": ("tax", "service"),
}

# Покупка по QR: офлайн, привычная группа клиента.
QR_OPERATION = "pay_qr"

MARKET_GROUP = "marketplace"


# ------------------------------------------------------------
# ПРИВЫЧКИ ТРАТ
# ------------------------------------------------------------

HABIT_GROUPS_COUNT = (4, 9)          # сколько групп у клиента «свои»
HABIT_STICKINESS = (0.55, 0.90)      # шанс попасть в привычную точку
TASTE_SIGMA = 0.60                   # лог-нормальный разброс вкуса
HABIT_LOG_SIGMA_FACTOR = 0.45        # привычная сумма стабильнее случайной
PRICE_POINT_GROUPS = ("transit", "coffee", "fastfood", "taxi")
PRICE_POINT_COUNT = (1, 3)

DRIFT_POINTS = (1, 3)
DRIFT_BLEND_DAYS = (30, 60)
DRIFT_REDRAW_HABITS = (1, 2)

EPISODES = (0, 3)
EPISODE_DAYS = (5, 21)
EPISODE_KINDS = ("trip", "burst", "quiet")
EPISODE_WEIGHTS = (0.45, 0.30, 0.25)
TRIP_FOREIGN_SHARE = 0.55
TRIP_GROUP_BOOST = {"travel": 3.0, "hotel": 4.0, "restaurant": 2.0, "airline": 2.5}
BURST_GROUP_BOOST = {"home_goods": 6.0, "furniture": 5.0, "electronics": 3.0}
QUIET_RATE_FACTOR = 0.50

# Зарплата: у клиента стабильная сумма, а не новый розыгрыш каждый месяц.
SALARY_FACTOR = (0.88, 1.06)
SALARY_NOISE = 0.02
SALARY_RAISE_SHARE = 0.30
SALARY_RAISE_FACTOR = (1.06, 1.20)

# Подписки: начало, остановка, редкая смена суммы.
SUBSCRIPTION_LATE_START_SHARE = 0.35
SUBSCRIPTION_STOP_SHARE = 0.35
SUBSCRIPTION_PRICE_CHANGE_SHARE = 0.30
SUBSCRIPTION_PRICE_CHANGE = (0.10, 0.30)


# ------------------------------------------------------------
# АВТОРИЗАЦИЯ
# ------------------------------------------------------------
#
# Защищённое действие возможно только после успешного входа
# либо при действующей авторизации. Приложение помнит вход
# ограниченное время; после этого нужен новый.
# ------------------------------------------------------------

AUTH_TTL_HOURS = 40
AUTH_MAX_ATTEMPTS = 3
AUTH_RECOVERY_SHARE = 0.45      # доля попыток восстановить доступ после провалов
AUTH_RETRY_SECONDS = (3, 25)
AUTH_SCREEN_GAP_SECONDS = (1, 6)


# ------------------------------------------------------------
# СВОБОДНЫЙ ПЛАТЁЖ
# ------------------------------------------------------------
#
# Клиент платит не только по счетам из расписания. Такой платёж
# имеет своё назначение и свою сумму и ничей счёт не гасит.
# ------------------------------------------------------------

FREE_PAYMENT_AMOUNT = {
    "utility": (3_000, 0.50),
    "mobile": (2_000, 0.45),
    "internet": (4_000, 0.35),
    "fine": (12_000, 0.60),
    "tax": (20_000, 0.80),
}


# ------------------------------------------------------------
# ПЕРЕВОДЫ
# ------------------------------------------------------------
#
# Перевод между своими счетами требует, чтобы своих счетов
# было хотя бы два. Это правило продукта, а не случайность.
# ------------------------------------------------------------

ACCOUNT_PRODUCTS: tuple[str, ...] = ("debit_card", "credit_card", "deposit")

TRANSFER_OWN_MIN_ACCOUNTS = 2


# ------------------------------------------------------------
# СЕССИИ И ОПЕРАЦИИ
# ------------------------------------------------------------

MAX_SCREENS_PER_SESSION = 20

# Сессия может нести не одно намерение: посмотрел баланс, потом
# заплатил. Это и даёт разную глубину при одном сценарии.
CONTINUE_SHARE = 0.20
MAX_INTENTS_PER_SESSION = 3
MAX_ATTEMPTS_PER_INTENT = 3
MAX_SUPPORT_HOPS = 1
MIN_STEP_SECONDS = 1

# Сбой сервиса: общий для всех клиентов, в RAW не пишется.
OUTAGE_DAY_PROBABILITY = 0.06
OUTAGE_DOMAINS = ("transfers", "payments", "auth", "cards")
OUTAGE_HOURS = (1, 4)
OUTAGE_FAILURE_BOOST = 0.35

# Поддержка после сбоя доступна всем, но у неё разный порог.
SUPPORT_ADOPTED_FACTOR = 1.0
SUPPORT_FOREIGN_FACTOR = 0.35

# Память клиента: сколько времени событие влияет на выбор.
RECENT_FAILURE_DAYS = 3
RECENT_OFFER_DAYS = 7
RECENT_REMINDER_DAYS = 3
RECENT_REJECTION_DAYS = 60
RECENT_VIEW_DAYS = 14
UNFINISHED_DAYS = 7

# Досрочное закрытие возможно не раньше этого срока владения.
EARLY_CLOSE_MIN_MONTHS = 3

# Блокировка карты: сколько дней клиент терпит до разблокировки.
CARD_BLOCK_MAX_DAYS = 21


# ------------------------------------------------------------
# ЗАЯВКИ
# ------------------------------------------------------------

# Интерес к продукту после осознанного изучения раздела.
APPLY_AFTER_EXPLORE = 0.013
EXPLORE_DEPTH_FACTOR = {"root": 1.0, "calc": 2.0, "terms": 3.5}
EXPLORE_REPEAT_FACTOR = 0.5
EXPLORE_REPEAT_CAP = 3.0
REJECTION_FACTOR = 0.3
OFFER_FACTOR = 1.8
UNFINISHED_FACTOR = 1.8

# Баннер по уже имеющемуся продукту показывается редко.
OWNED_BANNER_FACTOR = 0.10

# Задержка между успешным deposit_open и договором.
DEPOSIT_OPEN_DELAY = (60, 600)


# ------------------------------------------------------------
# LATENT
# ------------------------------------------------------------
#
# Имена, которых не должно быть ни в одной колонке RAW и ни
# в одном ключе payload. Проверяется тестом.
# ------------------------------------------------------------

V2_LATENT_NAMES: frozenset[str] = frozenset(
    {
        "scenario",
        "target",
        "intent",
        "attempt",
        "outage",
        "completed",
        "abandoned",
        "support_used",
        "habit",
        "habits",
        "taste",
        "stickiness",
        "episode",
        "drift",
        "price_points",
        "bill",
        "bill_kind",
        "paid_via",
        "authorized",
        "feasible",
        "bill_key",
        "op_key",
        "card_blocked",
        "explore_depth",
        "operation_triggers",
        "salary_factor",
        "in_app_bills_share",
    }
)


def is_bill_group(group: str) -> bool:
    return group in BILL_GROUPS


# ------------------------------------------------------------
# КОНФИГУРАЦИЯ ДАТАСЕТА
# ------------------------------------------------------------


def configuration() -> dict:
    """
    Правила, с которыми собран датасет. Пишется в манифест,
    чтобы набор нельзя было спутать с собранным по другим
    константам.
    """

    return {
        "auth": {
            "ttl_hours": AUTH_TTL_HOURS,
            "max_attempts": AUTH_MAX_ATTEMPTS,
            "recovery_share": AUTH_RECOVERY_SHARE,
        },
        "bills": {
            "groups": list(BILL_GROUPS),
            "bank_share": list(BILL_BANK_SHARE),
            "app_window_days": BILL_APP_WINDOW_DAYS,
            "internet_share": INTERNET_SHARE,
            "second_utility_share": SECOND_UTILITY_SHARE,
            "fine_rate_per_month": FINE_RATE_PER_MONTH,
            "service_rate_per_month": SERVICE_RATE_PER_MONTH,
            "plan_change_share": PLAN_CHANGE_SHARE,
            "linked_delay_seconds": list(LINKED_DELAY_SECONDS),
        },
        "habits": {
            "groups_count": list(HABIT_GROUPS_COUNT),
            "taste_sigma": TASTE_SIGMA,
            "log_sigma_factor": HABIT_LOG_SIGMA_FACTOR,
            "drift_points": list(DRIFT_POINTS),
            "drift_blend_days": list(DRIFT_BLEND_DAYS),
            "episodes": list(EPISODES),
            "episode_days": list(EPISODE_DAYS),
        },
        "sessions": {
            "max_screens": MAX_SCREENS_PER_SESSION,
            "max_attempts_per_intent": MAX_ATTEMPTS_PER_INTENT,
            "max_support_hops": MAX_SUPPORT_HOPS,
            "max_intents": MAX_INTENTS_PER_SESSION,
            "continue_share": CONTINUE_SHARE,
        },
        "outcomes": {
            "outage_day_probability": OUTAGE_DAY_PROBABILITY,
            "outage_domains": list(OUTAGE_DOMAINS),
            "outage_hours": list(OUTAGE_HOURS),
        },
        "products": {
            "early_close_min_months": EARLY_CLOSE_MIN_MONTHS,
            "card_block_max_days": CARD_BLOCK_MAX_DAYS,
            "transfer_own_min_accounts": TRANSFER_OWN_MIN_ACCOUNTS,
        },
        "applications": {
            "apply_after_explore": APPLY_AFTER_EXPLORE,
            "explore_depth_factor": dict(EXPLORE_DEPTH_FACTOR),
            "rejection_factor": REJECTION_FACTOR,
            "offer_factor": OFFER_FACTOR,
            "owned_banner_factor": OWNED_BANNER_FACTOR,
        },
        "subscriptions": {
            "late_start_share": SUBSCRIPTION_LATE_START_SHARE,
            "stop_share": SUBSCRIPTION_STOP_SHARE,
            "price_change_share": SUBSCRIPTION_PRICE_CHANGE_SHARE,
        },
    }


BILL_GROUP_INDEX: tuple[int, ...] = tuple(
    index for index, group in enumerate(CATEGORIES) if group in BILL_GROUPS
)
