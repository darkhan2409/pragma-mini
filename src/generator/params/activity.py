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
            "silent": 0.01,
            "rare": 0.90,
            "regular": 5.60,
            "high": 9.00,
            "extreme": 15.00,
        }
    )

    # Сессий приложения в день.
    #
    # После onboarding клиент заходит в приложение регулярно: от
    # пары раз в год у молчуна до раза в день у самых активных. В
    # среднем на месяц с приложением выходит 6–20 сессий.
    sessions_per_day: dict = field(
        default_factory=lambda: {
            "silent": 0.005,
            "rare": 0.05,
            "regular": 0.40,
            "high": 0.55,
            "extreme": 1.10,
        }
    )

    # Множитель по стадии клиента на день (behaviour/engagement.stage).
    #
    # Только стадии, которые заданы НЕ лентой: срок с прихода,
    # скрытый стресс и просрочка. Ярлыков, выведенных из объёма
    # действий, нет вовсе: ярлык описывает поведение, а не вызывает
    # его, и множитель на нём замыкал бы круг. Молчание задают
    # отношения клиента с банком (behaviour/engagement).
    state_factor: dict = field(
        default_factory=lambda: {
            "prospect": 0.0,
            "onboarding": 0.55,
            "new_client": 0.85,
            "financial_stress": 0.80,
            "delinquent": 0.65,
        }
    )

    # Множитель частоты покупок и сессий по роли банка. Покупки
    # дополнительно режет видимая доля оборота (visible_share),
    # поэтому роль здесь задаёт прежде всего заходы в приложение:
    # клиент с одним кредитом тоже смотрит график и платит.
    role_factor: dict = field(
        default_factory=lambda: {
            "primary": 1.30,
            "secondary": 1.15,
            "credit_only": 1.15,
            "deposit_only": 1.00,
            "episodic": 1.00,
        }
    )

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
    communications_base_per_month: float = 11.5

    # Множитель частоты рассылок по режиму активности клиента.
    communications_mode_factor: dict = field(
        default_factory=lambda: {
            "silent": 0.05,
            "rare": 0.35,
            "regular": 1.05,
            "high": 1.30,
            "extreme": 1.40,
        }
    )

    # Множитель частоты рассылок клиенту, замолчавшему в глазах
    # банка: остаются сервисные сообщения и попытки вернуть.
    communications_silent_factor: float = 0.3

    # Оплата по QR: в Казахстане это основной способ платить в
    # рознице, и раздел приложения для неё уже существовал.
    qr_share_of_pos: float = 0.22
    qr_min_digital_affinity: float = 0.25

    # Сколько экранов сверх обязательного пути смотрят в
    # сессии, по цели сессии (behaviour/sessions.GOAL_*). Раньше
    # сессия всегда была ровно три-четыре экрана.
    session_extra_screens: dict = field(
        default_factory=lambda: {
            "balance_check": 5.0,
            "payment": 5.0,
            "transfer": 5.0,
            "card_management": 5.5,
            "product_explore": 7.0,
            "loan_service": 5.5,
            "deposit_service": 5.5,
            "market": 7.0,
            "profile_settings": 4.0,
            "support": 4.5,
        }
    )

    # Доля регулярных списаний — счетов (коммуналка, связь,
    # интернет…) и подписок, — которые клиент проводит через этот
    # банк, по режиму активности.
    recurring_in_bank_share: dict = field(
        default_factory=lambda: {
            "silent": 0.05,
            "rare": 0.40,
            "regular": 1.0,
            "high": 1.0,
            "extreme": 1.0,
        }
    )

    # Вес цели «перевод» у сессии клиента, подключившего переводы
    # (у «баланса» — 3.0).
    transfer_goal_weight: float = 0.6

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
            "silent": 0.1,
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
            "silent": 0.15,
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
