from __future__ import annotations

from dataclasses import dataclass, field


# ============================================================
# ЖИЗНЕННЫЙ ЦИКЛ
# ============================================================
#
# Стадия клиента на день и жизненные события. Неактивность
# заранее не планируется: она вырастает из отношений клиента с
# банком (params/engagement.py, behaviour/engagement.py).
# ============================================================


@dataclass(frozen=True)
class LifecycleParams:

    # Стадия клиента на день (behaviour/engagement.stage): сколько
    # месяцев он новый после прихода, с какого уровня стресса и с
    # какой просрочки поведение сбавляет частоты.
    onboarding_months: int = 1
    new_client_months: int = 3
    stress_stage_level: float = 0.45
    delinquent_dpd: int = 30
    # Сколько дней после просрочки delinquent_dpd и больше банк не
    # шлёт клиенту предложений (синтетическое допущение).
    offer_cooloff_days: int = 90

    # Сколько дней молчания банк ждёт, прежде чем звать клиента
    # назад. Скрытое состояние банк не видит, он видит только
    # отсутствие операций, и замечает его не в первый же день.
    winback_after_silence_days: int = 30

    # ------------------------------------------------------------
    # ЖИЗНЕННЫЕ СОБЫТИЯ
    # ------------------------------------------------------------
    #
    # Событие меняет несколько потоков сразу: доход, бюджет,
    # категории трат, привычки, интерес к продуктам и вероятность
    # просрочки. Банк узнаёт о нём позже и не всегда.
    # ------------------------------------------------------------

    life_event_rate_per_year: dict = field(
        default_factory=lambda: {
            "job_change": 0.16,
            "job_loss": 0.055,
            "income_up": 0.24,
            "income_down": 0.07,
            "move": 0.09,
            "big_purchase": 0.20,
            "illness": 0.13,
            "vacation": 0.55,
            "renovation": 0.07,
            "education": 0.06,
            # Браков около 6.1 и разводов около 2.0 на 1 000 жителей
            # (БНС 2024): на 1 000 взрослых — около 18 вступивших в брак
            # и 6 разведённых в год, развод на три свадьбы. Было 0.03 и
            # 0.018: развод почти на каждую свадьбу (аудит 2026-11-05, M14).
            "wedding": 0.04,
            "divorce": 0.006,
        }
    )

    life_event_stage_factor: dict = field(
        default_factory=lambda: {
            "young_adult": {"job_change": 2.1, "education": 3.2, "wedding": 2.4,
                            "big_purchase": 0.7, "renovation": 0.4, "divorce": 0.3, "illness": 0.6},
            "early_career": {"job_change": 1.6, "wedding": 2.2, "big_purchase": 1.2,
                             "education": 1.4, "move": 1.5},
            "family": {"renovation": 1.6, "big_purchase": 1.3, "divorce": 1.5,
                       "education": 0.8, "job_change": 0.8},
            "mature": {"job_change": 0.5, "illness": 1.6, "renovation": 1.2,
                       "wedding": 0.3, "education": 0.4, "move": 0.7},
            "retired": {"job_change": 0.15, "job_loss": 0.2, "illness": 2.6,
                        "income_up": 0.3, "education": 0.1, "wedding": 0.1, "move": 0.5,
                        "big_purchase": 0.6, "vacation": 0.8},
        }
    )

    # Событие требует состояния: свадьба только у несемейных,
    # развод только у семейных.
    life_event_requires: dict = field(
        default_factory=lambda: {
            "wedding": ("single", "divorced", "widow"),
            "divorce": ("married", "civil_marriage"),
        }
    )

    # Сколько раз событие может повториться за горизонт.
    life_event_max: dict = field(
        default_factory=lambda: {
            "job_loss": 2,
            "child_birth": 2,
            "move": 2,
            "wedding": 1,
            "divorce": 1,
            "job_change": 3,
        }
    )

    # РОЖДЕНИЕ РЕБЁНКА
    #
    # Частота — по возрасту матери, а не по стадии жизни: прежде
    # рожали в 61 год, 15% матерей были старше 45, а второй ребёнок
    # появлялся через 2 дня после первого (аудит 2026-11-05, M14).
    #
    # Рождений на женщину в год с возраста группы. Синтетическое
    # допущение по БНС 2024 (research.md): доли рождений по возрасту
    # матери (до 20 лет — 11.9 тыс. из 365.9 тыс., 25–29 — 27.5%,
    # 30–34 — 26.2%, средний возраст 29.8), нормированные на СКР 2.8.
    birth_rate_by_mother_age: tuple = (
        (15, 0.018), (20, 0.106), (25, 0.154), (30, 0.147),
        (35, 0.095), (40, 0.034), (45, 0.003), (50, 0.0),
    )
    # У клиента-мужчины рожает партнёрша: она в среднем моложе.
    partner_age_gap: int = 3
    # В браке рожают чаще: при составе клиентов 25–44 лет средний
    # множитель около 1, внебрачных рождений около 14%.
    birth_family_factor: dict = field(
        default_factory=lambda: {
            "married": 1.45, "civil_marriage": 1.2, "single": 0.25, "divorced": 0.25, "widow": 0.1,
        }
    )
    # Не раньше чем через 15 месяцев после прошлого рождения.
    birth_min_interval_days: int = 456
    # Кандидаты в рождения идут с этой частотой и принимаются с
    # долей частота_на_дату / эта: прореживание держит процесс
    # шагами вперёд по времени (горизонт только обрезает будущее).
    # Не ниже max(birth_rate) * max(birth_family_factor).
    birth_candidate_rate: float = 0.25
    # Когда мать выходит на работу после рождения, дней: до года,
    # до полутора лет (оплачиваемый отпуск) или позже, до трёх.
    return_to_work_days: tuple = ((180, 365, 0.20), (365, 548, 0.50), (548, 1095, 0.30))

    # Траты жизненных событий (life/events.event_purchases): категория,
    # сколько покупок от и до (None — одна в месяц события), первый и
    # последний день от даты события (None — до конца события), во
    # сколько раз чек больше обычного от и до. Синтетическое
    # допущение: данных банка о составе таких трат нет. Ни один сдвиг
    # не раньше events.LOOKAHEAD.
    event_baskets: dict = field(
        default_factory=lambda: {
            "wedding": (
                ("clothing", 1, 3, -45, -3, 2.0, 5.0),
                ("shoes", 1, 2, -45, -3, 1.5, 3.0),
                ("cosmetics", 0, 2, -20, -1, 1.5, 3.0),
                ("beauty", 1, 2, -5, 0, 1.5, 3.0),
                ("restaurant", 1, 1, -2, 0, 10.0, 30.0),
            ),
            "illness": (
                ("pharmacy", 2, 5, 0, None, 1.0, 2.5),
                ("medical", 1, 2, 0, None, 1.0, 3.0),
                ("lab", 0, 2, 0, None, 1.0, 2.0),
            ),
            "renovation": (
                ("home_goods", 3, 10, 0, None, 2.0, 6.0),
                ("furniture", 0, 2, 0, None, 1.0, 2.5),
                ("appliances", 0, 1, 0, None, 1.0, 2.0),
            ),
            "education": (
                ("education", None, None, 0, None, 1.0, 1.0),
                ("books", 0, 2, 0, None, 1.0, 2.0),
            ),
            "child_birth": (
                ("medical", 1, 3, -150, -10, 1.0, 2.5),
                ("kids", 2, 4, -90, -7, 2.0, 5.0),
                ("pharmacy", 2, 4, -3, 60, 1.0, 2.0),
                ("kids", 2, 5, 0, 90, 1.0, 2.5),
            ),
            "move": (
                ("home_goods", 1, 3, 0, 30, 1.5, 4.0),
                ("furniture", 0, 1, 0, 45, 1.0, 2.0),
            ),
        }
    )

    # Переезд: в другое поселение или внутри своего.
    move_to_other_settlement_share: float = 0.42

    # Банк узнаёт об изменении атрибута с задержкой и не всегда.
    profile_change_known_share: dict = field(
        default_factory=lambda: {
            "move": 0.62,
            "job_change": 0.48,
            "job_loss": 0.22,
            "income_up": 0.35,
            "income_down": 0.20,
            "child_birth": 0.34,
            "wedding": 0.40,
            "divorce": 0.30,
        }
    )

    profile_change_delay_days: tuple = (3, 240)

    # Изменение подтверждается документом не всегда.
    profile_change_confirmed_share: float = 0.58
