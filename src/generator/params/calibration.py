from __future__ import annotations

from dataclasses import dataclass


# ============================================================
# КАЛИБРОВОЧНЫЕ ЭТАЛОНЫ
# ============================================================
#
# Метрика без реального эталона хранится со статусом
# no_reference и пустым значением. Выдуманное точное число
# здесь запрещено: отчёт обязан честно показать, что эталона
# нет, а не сравнить генератор сам с собой.
# ============================================================


STATUS_REFERENCE = "reference"
STATUS_HYPOTHESIS = "hypothesis"
STATUS_NO_REFERENCE = "no_reference"

STATUSES = (STATUS_REFERENCE, STATUS_HYPOTHESIS, STATUS_NO_REFERENCE)


@dataclass(frozen=True)
class CalibrationTarget:
    metric: str
    group: str
    unit: str
    status: str
    value: float | None = None
    low: float | None = None
    high: float | None = None
    source: str | None = None
    period: str | None = None
    geography: str | None = None
    confidence: str = "low"
    tolerance: float = 0.25
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "metric": self.metric,
            "group": self.group,
            "unit": self.unit,
            "status": self.status,
            "value": self.value,
            "low": self.low,
            "high": self.high,
            "source": self.source,
            "period": self.period,
            "geography": self.geography,
            "confidence": self.confidence,
            "tolerance": self.tolerance,
            "note": self.note,
        }


# Решение об объёме: лента клиента сделана плотнее прежнего
# плана ради обучающего материала. Данными банка эта величина не
# подтверждена и подтверждена быть не может — в отчёте банка
# такой метрики нет.
_VOLUME_DECISION = (
    "решение владельца проекта о плотности ленты, а не измерение банка "
    "и не полоса из плана"
)

# Сколько событий в месяц ожидается от клиента КАЖДОГО режима.
#
# Полоса — это ожидание, а не ограничитель. Клиент, выпавший из
# своей полосы, остаётся в данных целиком: отчёт реализма
# называет его и считает долю таких клиентов, но ленту никто не
# обрезает. Обрезка превратила бы наблюдение в подгонку.
# Полоса silent начинается не с нуля: даже молчун получает
# зарплату, ежемесячную выписку, начисление процентов и
# сервисные уведомления. Это события БАНКА, частотами активности
# они не управляются, и требовать от такого клиента пустой ленты
# значит требовать, чтобы банк перестал работать.
EVENTS_PER_MONTH_BY_MODE: dict[str, tuple[int, int]] = {
    "silent": (0, 15),
    "rare": (6, 30),
    "regular": (50, 110),
    "high": (140, 320),
    "extreme": (450, 1100),
}


def _reference(metric, group, unit, value, source, period, geography, confidence, tolerance=0.20, note=""):
    return CalibrationTarget(metric, group, unit, STATUS_REFERENCE, value=value, source=source,
                             period=period, geography=geography, confidence=confidence,
                             tolerance=tolerance, note=note)


def _hypothesis(metric, group, unit, low, high, note=""):
    """
    Полоса-ориентир из плана. Это НЕ эталон: данными банка она не
    подтверждена, выход за неё ошибкой генератора не является и
    подгонки не требует.
    """

    mark = "гипотеза из плана, данными банка не подтверждена"

    return CalibrationTarget(metric, group, unit, STATUS_HYPOTHESIS, low=low, high=high,
                             source="план для генератора.txt, раздел 17.2", period=None,
                             geography="KZ", confidence="low",
                             note=f"{note}; {mark}" if note else mark)


def _absent(metric, group, unit, note=""):
    return CalibrationTarget(metric, group, unit, STATUS_NO_REFERENCE, note=note)


_BANK_REPORT = "Отчет по данным NBO/NBC/NBT pragmatiq, ред. 2026-09-07"
_BANK_PERIOD = "2024-12-16 .. 2026-08-24"


DEFAULT_TARGETS: tuple = (
    # --- реальные якоря банка ---
    _reference("communications_per_client_month", "channels", "events", 3.8,
               _BANK_REPORT, _BANK_PERIOD, "KZ", "high", 0.25),
    _reference("delivery_rate_call", "channels", "share", 0.062,
               _BANK_REPORT, _BANK_PERIOD, "KZ", "high", 0.30),
    _reference("delivery_rate_sms", "channels", "share", 0.813,
               _BANK_REPORT, _BANK_PERIOD, "KZ", "high", 0.15),
    _reference("delivery_rate_push", "channels", "share", 0.426,
               _BANK_REPORT, _BANK_PERIOD, "KZ", "high", 0.30),
    _hypothesis("app_sessions_per_client_month", "activity", "events", 6.0, 20.0,
                note=(
                    "прежний эталон банка (1 сессия в месяц) считался по всей базе, "
                    "включая тех, кто приложение не ставил, и покрытие GA4 в отчёте "
                    "названо сомнительным. Полоса взята для клиента, у которого "
                    "приложение есть, и эталоном не является"
                )),
    _reference("banner_ctr", "channels", "share", 0.0235,
               _BANK_REPORT, _BANK_PERIOD, "KZ", "high", 0.40),
    _reference("app_domain_share_auth", "activity", "share", 0.849,
               _BANK_REPORT, _BANK_PERIOD, "KZ", "medium", 0.25),
    _reference("app_domain_share_cards", "activity", "share", 0.601,
               _BANK_REPORT, _BANK_PERIOD, "KZ", "medium", 0.25),
    _reference("app_domain_share_transfers", "activity", "share", 0.509,
               _BANK_REPORT, _BANK_PERIOD, "KZ", "medium", 0.25),
    _reference("app_domain_share_loans", "activity", "share", 0.442,
               _BANK_REPORT, _BANK_PERIOD, "KZ", "medium", 0.25),
    _reference("app_domain_share_payments", "activity", "share", 0.296,
               _BANK_REPORT, _BANK_PERIOD, "KZ", "medium", 0.25),

    # --- объём событий на клиента в месяц ---
    #
    # Это НЕ гипотеза плана и не измерение банка, а решение
    # владельца проекта: сделать ленту клиента плотнее, чтобы
    # модели было на чём учиться. Прежние полосы (среднее 65–85)
    # взяты из плана и здесь сознательно заменены.
    _hypothesis("events_per_client_month_mean", "activity", "events", 110, 150,
                note=_VOLUME_DECISION),
    _hypothesis("events_per_client_month_p10", "activity", "events", 0, 4),
    _hypothesis("events_per_client_month_p25", "activity", "events", 6, 30),
    _hypothesis("events_per_client_month_median", "activity", "events", 60, 100),
    _hypothesis("events_per_client_month_p75", "activity", "events", 130, 200),
    _hypothesis("events_per_client_month_p90", "activity", "events", 230, 330),
    _hypothesis("events_per_client_month_p95", "activity", "events", 340, 500),
    _hypothesis("events_per_client_month_p99", "activity", "events", 650, 1000),
    # --- кредитный риск и денежный поток ---
    _hypothesis("dpd90_client_share", "credit", "share", 0.03, 0.06,
                note="доля клиентов, дошедших до просрочки 90+ за окно"),
    _hypothesis("installment_missed_share", "credit", "share", 0.06, 0.12,
                note="доля платежей графика, не оплаченных в льготный срок"),
    _hypothesis("approval_rate_credit", "credit", "share", 0.40, 0.50,
                note="одобрение по кредитным семействам; по остальным оно близко к единице"),
    _hypothesis("loan_amount_to_income_median", "credit", "ratio", 1.0, 3.0,
                note="выдача кредита к месячному доходу"),
    _hypothesis("inbound_transfers_per_client_month", "channels", "events", 0.5, 2.0,
                note="входящие переводы от внешних отправителей"),

    # --- продукты, мошенничество, поведение ---
    _hypothesis("contracts_per_client_median", "products", "contracts", 2, 4,
                note="сколько договоров держит обычный клиент, включая открытые до окна"),
    _hypothesis("fraud_episodes_per_client_year", "fraud", "events", 0.03, 0.08,
                note="эпизод мошенничества у клиента в год"),
    _hypothesis("night_purchase_share", "activity", "share", 0.02, 0.05,
                note="покупки между полуночью и шестью утра"),
    _hypothesis("out_of_hours_pos_share", "activity", "share", 0.0, 0.03,
                note="покупки в точке вне её часов работы"),
    _hypothesis("support_chat_share", "channels", "share", 0.35, 0.75,
                note="доля обращений в поддержку через чат"),
    _hypothesis("trait_behaviour_min_correlation", "activity", "correlation", 0.2, 1.0,
                note="слабейшая из связей «черта и поведение, которым она управляет»"),

    _hypothesis("zero_month_share", "zero_months", "share", 0.10, 0.15,
                note="месяцы ВООБЩЕ без записей; полоса из плана, реального эталона нет"),
    _absent("no_client_action_month_share", "zero_months", "share",
            note="месяцы, где банк что-то записал, а клиент не делал ничего; "
                 "показатель другой и полосу zero_month_share к нему применять нельзя"),
    _hypothesis("segment_share_silent", "zero_months", "share", 0.10, 0.15),
    _hypothesis("segment_share_sleepy", "activity", "share", 0.15, 0.20),
    _hypothesis("segment_share_moderate", "activity", "share", 0.25, 0.30),
    _hypothesis("segment_share_regular", "activity", "share", 0.25, 0.30),
    _hypothesis("segment_share_high", "activity", "share", 0.08, 0.12),
    _hypothesis("segment_share_extreme", "activity", "share", 0.01, 0.03),

    # --- метрики без эталона ---
    _absent("mcc_share_distribution", "mcc", "share", "нужна выгрузка транзакций банка"),
    _absent("category_share_distribution", "mcc", "share", "нужна выгрузка транзакций банка"),
    _absent("purchase_amount_quantiles", "amounts", "tenge", "нужна выгрузка транзакций банка"),
    _absent("balance_quantiles", "amounts", "tenge", "нужны остатки по счетам"),
    _absent("income_quantiles", "amounts", "tenge", "нужны зачисления зарплаты"),
    _absent("online_share", "online_offline", "share", "нужен флаг e-commerce в выгрузке"),
    _absent("hour_of_day_profile", "time", "share", "нужны часы операций"),
    _absent("day_of_week_profile", "time", "share", "нужны даты операций"),
    _absent("city_share_distribution", "cities", "share", "нужен город точки"),
    _absent("product_penetration", "products", "share", "нужен реестр договоров"),
    _absent("application_conversion", "products", "share", "нужна воронка заявок"),
    _absent("dpd30_share", "dpd", "share", "нужна витрина просрочек"),
    _absent("dpd90_share", "dpd", "share", "нужна витрина просрочек"),
    _absent("pause_length_distribution", "pauses", "days", "нужна помесячная активность клиентов"),
    _absent("return_after_pause_share", "pauses", "share", "нужна помесячная активность клиентов"),
    _absent("returned_by_window_end_share", "pauses", "share",
            "доля пауз, закончившихся возвращением К КОНЦУ НАБЛЮДЕНИЯ; "
            "это наблюдаемый результат на дату конца датасета, а не вероятность возвращения"),
    _absent("pause_ongoing_at_window_end_share", "pauses", "share",
            "доля пауз, которые на дату конца датасета ещё длятся; "
            "уходом из банка это не является, будущее клиента неизвестно"),
    _absent("confirmed_closure_share", "pauses", "share",
            "доля клиентов с ПОДТВЕРЖДЁННЫМ закрытием отношений; "
            "молчание на конце окна сюда не входит"),
)


@dataclass(frozen=True)
class CalibrationParams:

    targets: tuple = DEFAULT_TARGETS

    def by_group(self) -> dict:
        grouped: dict = {}
        for target in self.targets:
            grouped.setdefault(target.group, []).append(target)
        return grouped

    def as_list(self) -> list:
        return [target.as_dict() for target in self.targets]
