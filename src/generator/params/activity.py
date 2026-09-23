from __future__ import annotations

from dataclasses import dataclass, field


# ============================================================
# АКТИВНОСТЬ
# ============================================================
#
# Интенсивности задаются режимом активности, состоянием
# жизненного цикла и ролью банка. Мягкие ограничители не дают
# хвосту превратиться в бесконечный цикл повторов.
# ============================================================


@dataclass(frozen=True)
class ActivityParams:

    # Покупок в день по режиму активности.
    #
    # Это ПОПЫТКИ, а не наблюдаемые события: часть их уходит мимо
    # банка (hidden_purchase_share), поэтому в выгрузке покупок
    # заметно меньше.
    purchases_per_day: dict = field(
        default_factory=lambda: {
            "silent": 0.10,
            "rare": 1.20,
            "regular": 4.60,
            "high": 8.00,
            "extreme": 15.00,
        }
    )

    # Сессий приложения в день.
    #
    # Прежние значения давали «регулярному» клиенту 1,4 сессии в
    # месяц: приложение стояло почти у всех, а следов в данных
    # почти не было. Теперь после onboarding клиент заходит в
    # него регулярно — от раза в две недели у молчунов до
    # нескольких раз в день у самых активных.
    sessions_per_day: dict = field(
        default_factory=lambda: {
            "silent": 0.04,
            "rare": 0.15,
            "regular": 0.55,
            "high": 1.30,
            "extreme": 3.20,
        }
    )

    # Множитель по состоянию жизненного цикла.
    #
    # Только состояния, которые заданы НЕ лентой: срок с
    # регистрации, скрытый стресс и нехватка денег на платёж.
    # Ярлыки, выведенные из объёма действий (active, growing,
    # stable, dormant, churn_risk, churned, returned,
    # closed_relationship), поведения не меняют: ярлык описывает
    # поведение, а не вызывает его. Иначе множитель на них
    # замыкал круг — ноль у dormant не давал клиенту ни одного
    # действия, чтобы выйти из молчания, спад у churn_risk
    # углублял сам себя, рост у growing разгонял сам себя.
    # Настоящее молчание задают скрытые паузы (silenced_streams).
    state_factor: dict = field(
        default_factory=lambda: {
            "prospect": 0.0,
            "onboarding": 0.55,
            "new_client": 0.85,
            "financial_stress": 0.80,
            "delinquent": 0.65,
        }
    )

    # Множитель по роли банка: видимая доля оборота.
    role_factor: dict = field(
        default_factory=lambda: {
            "primary": 1.30,
            "secondary": 0.85,
            "credit_only": 0.45,
            "deposit_only": 0.35,
            "episodic": 0.40,
        }
    )

    # Пауза по видам: что именно замолкает.
    pause_silences: dict = field(
        default_factory=lambda: {
            "full": ("purchases", "sessions", "transfers", "cash", "bills"),
            "app_only": ("sessions",),
            "cards_only": ("purchases", "cash"),
            "other_bank": ("purchases", "transfers", "cash", "bills"),
            "seasonal": ("purchases", "sessions", "transfers", "cash", "bills"),
        }
    )

    other_bank_residual: float = 0.12

    # Нехватка денег на счёте в банке почти никогда не выглядит
    # как отказ: потребность закрывается наличными, деньгами в
    # другом банке или просто откладывается. Наблюдаемый отказ
    # это редкое событие, и после пары отказов за день клиент
    # перестаёт пробовать.
    hidden_purchase_share: float = 0.62
    decline_attempt_share: float = 0.10
    max_declines_per_day: int = 2
    autopay_attempt_share: float = 0.30

    # За сколько дней до платежа банк начинает напоминать.
    due_reminder_days: int = 5

    # Обращение по одному и тому же поводу не повторяется
    # каждый день.
    support_cooldown_days: int = 14

    # Согласие на рассылку отзывают.
    consent_withdrawal_per_year: float = 0.06

    # Сколько сообщений банк отправляет клиенту в месяц ДО
    # затуханий по усталости, согласию и состоянию клиента.
    # Значение пришпилено к измеренному якорю отчёта банка
    # (communications_per_client_month = 3.8): после затуханий
    # наблюдаемая частота выходит примерно на него.
    communications_base_per_month: float = 7.4

    # Оплата по QR: в Казахстане это основной способ платить в
    # рознице, и раздел приложения для неё уже существовал.
    qr_share_of_pos: float = 0.22
    qr_min_digital_affinity: float = 0.25

    # Сколько экранов сверх обязательного пути смотрят в
    # сессии. Раньше сессия всегда была ровно три-четыре
    # экрана.
    session_extra_screens: dict = field(
        default_factory=lambda: {
            "balance": 1.4,
            "payment": 0.8,
            "transfer": 0.9,
            "cards": 1.1,
            "explore": 1.6,
            "loan": 1.0,
            "deposit": 1.0,
            "market": 1.5,
            "profile": 0.6,
            "support": 0.7,
        }
    )

    weekend_factor_purchases: float = 1.10
    weekend_factor_sessions: float = 0.92

    # Часовые профили: будни и выходные различаются формой.
    hour_profile_weekday: tuple = (
        0.006, 0.003, 0.002, 0.002, 0.003, 0.008,
        0.022, 0.048, 0.062, 0.058, 0.052, 0.058,
        0.078, 0.070, 0.055, 0.052, 0.062, 0.085,
        0.098, 0.082, 0.055, 0.030, 0.014, 0.008,
    )

    hour_profile_weekend: tuple = (
        0.010, 0.006, 0.004, 0.003, 0.003, 0.005,
        0.010, 0.020, 0.035, 0.055, 0.072, 0.080,
        0.082, 0.078, 0.072, 0.068, 0.065, 0.068,
        0.072, 0.064, 0.050, 0.032, 0.020, 0.012,
    )

    session_hour_profile: tuple = (
        0.010, 0.005, 0.003, 0.003, 0.003, 0.008,
        0.025, 0.055, 0.080, 0.085, 0.075, 0.070,
        0.075, 0.075, 0.065, 0.065, 0.070, 0.085,
        0.100, 0.100, 0.085, 0.060, 0.035, 0.015,
    )

    # Ночной сегмент: кто вообще покупает ночью.
    night_segment_share: float = 0.045
    night_hours: tuple = (0, 1, 2, 3, 4, 5)
    night_boost: float = 1.8

    # Мягкие ограничители.
    max_sessions_per_day: int = 9
    max_screens_per_session: int = 20
    max_purchases_per_day: int = 18

    # Внешние переводы и наличные.
    transfers_per_month: dict = field(
        default_factory=lambda: {
            "silent": 0.2,
            "rare": 1.0,
            "regular": 4.0,
            "high": 8.0,
            "extreme": 14.0,
        }
    )

    # Вероятность показать баннеры на экране витрины.
    banner_screen_share: float = 0.35

    cash_withdrawals_per_month: dict = field(
        default_factory=lambda: {
            "silent": 0.4,
            "rare": 1.0,
            "regular": 2.2,
            "high": 3.0,
            "extreme": 4.0,
        }
    )

    cash_withdrawal_share_of_income: tuple = (0.05, 0.35)

    # Продолжительность сессии и шага.
    session_step_seconds: tuple = (4, 95)

    # Сбой сервиса: общий для всех клиентов, в RAW не пишется.
    outage_day_probability: float = 0.05
    outage_domains: tuple = ("transfers", "payments", "auth", "cards")
    outage_hours: tuple = (1, 4)
    outage_failure_boost: float = 0.35
