from __future__ import annotations

from dataclasses import dataclass, field


# ============================================================
# НАСЕЛЕНИЕ
# ============================================================
#
# Совместные распределения важнее независимого выбора: тип
# поселения задаёт занятость и доход, возраст задаёт этап
# жизни, этап задаёт семью и продуктовые потребности.
# ============================================================


# Режим активности клиента. Он один задаёт, сколько следов
# человек оставляет в банке за месяц:
#
#   silent    почти ничего: зарплата и пара обязательных списаний
#   rare      редкие покупки, приложение открывается изредка
#   regular   обычный клиент: карта, приложение, платежи
#   high      банк основной, оборот идёт через него
#   extreme   карта и приложение под рукой каждый день
#
# Объявленные полосы событий на клиента в месяц лежат в
# calibration.EVENTS_PER_MONTH_BY_MODE. Это ожидание, а не
# ограничитель: ленту клиента, выпавшего из полосы, никто не
# обрезает. Сверка с полосами — проверкой калибровки после
# генерации (README, раздел 1).
ACTIVITY_MODES = ("silent", "rare", "regular", "high", "extreme")


@dataclass(frozen=True)
class PopulationParams:

    # Возраст: доли по жизненным этапам и границы внутри этапа.
    stage_weights: dict = field(
        default_factory=lambda: {
            "young_adult": 0.13,
            "early_career": 0.28,
            "family": 0.31,
            "mature": 0.19,
            "retired": 0.09,
        }
    )

    stage_age_range: dict = field(
        default_factory=lambda: {
            "young_adult": (18, 25),
            "early_career": (25, 35),
            "family": (35, 50),
            "mature": (50, 63),
            "retired": (63, 82),
        }
    )

    gender_weights: dict = field(default_factory=lambda: {"M": 0.47, "F": 0.53})

    family_status_by_stage: dict = field(
        default_factory=lambda: {
            "young_adult": {"single": 0.62, "civil_marriage": 0.16, "married": 0.19, "divorced": 0.02, "widow": 0.01},
            "early_career": {"single": 0.28, "civil_marriage": 0.14, "married": 0.50, "divorced": 0.07, "widow": 0.01},
            "family": {"single": 0.13, "civil_marriage": 0.08, "married": 0.66, "divorced": 0.11, "widow": 0.02},
            "mature": {"single": 0.09, "civil_marriage": 0.05, "married": 0.68, "divorced": 0.12, "widow": 0.06},
            "retired": {"single": 0.06, "civil_marriage": 0.03, "married": 0.58, "divorced": 0.10, "widow": 0.23},
        }
    )

    children_rate_by_stage: dict = field(
        default_factory=lambda: {
            "young_adult": 0.35,
            "early_career": 1.20,
            "family": 2.10,
            "mature": 2.30,
            "retired": 2.40,
        }
    )

    # Дети старше 18 из домохозяйства уходят: в профиле остаются
    # только несовершеннолетние.
    children_at_home_factor: dict = field(
        default_factory=lambda: {
            "young_adult": 1.0,
            "early_career": 1.0,
            "family": 0.85,
            "mature": 0.35,
            "retired": 0.05,
        }
    )

    max_children: int = 6

    education_by_settlement_type: dict = field(
        default_factory=lambda: {
            "metropolis": {"higher": 0.46, "incomplete_higher": 0.12, "vocational": 0.21, "secondary": 0.19, "academic_degree": 0.02},
            "major_city": {"higher": 0.38, "incomplete_higher": 0.11, "vocational": 0.25, "secondary": 0.25, "academic_degree": 0.01},
            "regional_centre": {"higher": 0.33, "incomplete_higher": 0.10, "vocational": 0.27, "secondary": 0.29, "academic_degree": 0.01},
            "industrial_town": {"higher": 0.24, "incomplete_higher": 0.08, "vocational": 0.34, "secondary": 0.335, "academic_degree": 0.005},
            "small_town": {"higher": 0.22, "incomplete_higher": 0.08, "vocational": 0.32, "secondary": 0.375, "academic_degree": 0.005},
            "district_centre": {"higher": 0.18, "incomplete_higher": 0.07, "vocational": 0.31, "secondary": 0.435, "academic_degree": 0.005},
            "rural": {"higher": 0.12, "incomplete_higher": 0.05, "vocational": 0.28, "secondary": 0.548, "academic_degree": 0.002},
        }
    )

    housing_by_settlement_type: dict = field(
        default_factory=lambda: {
            "metropolis": {"own_apartment": 0.38, "rented": 0.34, "with_parents": 0.16, "own_house": 0.09, "municipal": 0.02, "office_housing": 0.01},
            "major_city": {"own_apartment": 0.44, "rented": 0.25, "with_parents": 0.16, "own_house": 0.12, "municipal": 0.02, "office_housing": 0.01},
            "regional_centre": {"own_apartment": 0.46, "rented": 0.19, "with_parents": 0.15, "own_house": 0.17, "municipal": 0.02, "office_housing": 0.01},
            "industrial_town": {"own_apartment": 0.49, "rented": 0.15, "with_parents": 0.15, "own_house": 0.18, "municipal": 0.02, "office_housing": 0.01},
            "small_town": {"own_apartment": 0.41, "rented": 0.12, "with_parents": 0.15, "own_house": 0.29, "municipal": 0.02, "office_housing": 0.01},
            "district_centre": {"own_apartment": 0.32, "rented": 0.10, "with_parents": 0.15, "own_house": 0.40, "municipal": 0.02, "office_housing": 0.01},
            "rural": {"own_apartment": 0.12, "rented": 0.06, "with_parents": 0.16, "own_house": 0.64, "municipal": 0.01, "office_housing": 0.01},
        }
    )

    # С этого возраста «живёт с родителями» почти не встречается:
    # вес варианта умножается на множитель.
    with_parents_until_age: int = 50
    with_parents_after_age_factor: float = 0.1

    income_type_by_stage: dict = field(
        default_factory=lambda: {
            "young_adult": {"employed": 0.44, "student": 0.24, "self_employed": 0.13, "state_employee": 0.08, "unemployed": 0.08, "business_owner": 0.03, "pensioner": 0.00},
            "early_career": {"employed": 0.60, "self_employed": 0.15, "state_employee": 0.11, "business_owner": 0.07, "unemployed": 0.05, "student": 0.02, "pensioner": 0.00},
            "family": {"employed": 0.59, "self_employed": 0.15, "state_employee": 0.12, "business_owner": 0.08, "unemployed": 0.05, "student": 0.00, "pensioner": 0.01},
            "mature": {"employed": 0.53, "self_employed": 0.13, "state_employee": 0.13, "business_owner": 0.07, "pensioner": 0.09, "unemployed": 0.05, "student": 0.00},
            "retired": {"pensioner": 0.82, "employed": 0.08, "self_employed": 0.05, "state_employee": 0.03, "business_owner": 0.01, "unemployed": 0.01, "student": 0.00},
        }
    )

    # Занятость в сельской местности смещена к самозанятости.
    income_type_settlement_factor: dict = field(
        default_factory=lambda: {
            "metropolis": {"employed": 1.15, "business_owner": 1.35, "self_employed": 1.10},
            "major_city": {"employed": 1.05, "business_owner": 1.10},
            "industrial_town": {"employed": 1.20, "state_employee": 1.10, "business_owner": 0.70},
            "small_town": {"employed": 0.85, "state_employee": 1.15, "self_employed": 1.20, "business_owner": 0.65},
            "district_centre": {"employed": 0.70, "state_employee": 1.25, "self_employed": 1.35, "business_owner": 0.55},
            "rural": {"employed": 0.50, "state_employee": 1.20, "self_employed": 1.80, "business_owner": 0.45},
        }
    )

    industry_weights: dict = field(
        default_factory=lambda: {
            "trade": 0.150, "construction": 0.110, "transport": 0.095, "education": 0.085,
            "healthcare": 0.080, "government": 0.075, "manufacturing": 0.070, "oil_and_gas": 0.055,
            "agriculture": 0.050, "it": 0.045, "finance": 0.040, "hospitality": 0.035,
            "mining": 0.030, "utilities": 0.025, "telecom": 0.020, "security": 0.012,
            "logistics": 0.010, "real_estate": 0.006, "media": 0.004, "science": 0.003,
        }
    )

    industry_income_types: tuple = ("employed", "state_employee")

    # Медианный месячный доход отрасли в тенге.
    income_median_by_industry: dict = field(
        default_factory=lambda: {
            "trade": 280_000, "construction": 340_000, "transport": 310_000, "education": 230_000,
            "healthcare": 270_000, "government": 300_000, "manufacturing": 350_000, "oil_and_gas": 720_000,
            "agriculture": 210_000, "it": 620_000, "finance": 520_000, "hospitality": 240_000,
            "mining": 560_000, "utilities": 320_000, "telecom": 400_000, "security": 220_000,
            "logistics": 300_000, "real_estate": 380_000, "media": 300_000, "science": 280_000,
        }
    )

    income_median_by_type: dict = field(
        default_factory=lambda: {
            "self_employed": 300_000,
            "business_owner": 650_000,
            "pensioner": 110_000,
            "student": 90_000,
            "unemployed": 80_000,
        }
    )

    income_sigma: float = 0.52

    income_settlement_factor: dict = field(
        default_factory=lambda: {
            "metropolis": 1.35,
            "major_city": 1.12,
            "regional_centre": 1.00,
            "industrial_town": 1.05,
            "small_town": 0.84,
            "district_centre": 0.74,
            "rural": 0.66,
        }
    )

    income_stage_factor: dict = field(
        default_factory=lambda: {
            "young_adult": 0.62,
            "early_career": 0.98,
            "family": 1.18,
            "mature": 1.10,
            "retired": 0.55,
        }
    )

    income_education_factor: dict = field(
        default_factory=lambda: {
            "secondary": 0.84,
            "vocational": 0.94,
            "incomplete_higher": 0.96,
            "higher": 1.14,
            "academic_degree": 1.30,
        }
    )

    income_bounds: tuple = (75_000, 4_500_000)

    # --- Анкета → поведение и доход (аудит 2026-11-05, волна 4) ---

    # Средняя зарплата региона к средней по стране (БНС 2025:
    # Мангистау 623, Атырау 613, Астана 594, Алматы 556, Шымкент 344,
    # Туркестанская 319 тыс. ₸ при 443 по стране; прочие — синтетическое
    # приближение). Прежде регион на доход не влиял: Шымкент выходил
    # богаче Атырау (аудит, M8, persona-15).
    income_region_factor: dict = field(
        default_factory=lambda: {
            "Mangystau": 1.41, "Atyrau": 1.38, "Astana": 1.34, "Almaty": 1.26, "Shymkent": 0.78,
            "Turkistan": 0.72, "Karaganda": 0.92, "Pavlodar": 0.93, "East Kazakhstan": 0.90,
            "Ulytau": 1.05, "Aktobe": 0.88, "West Kazakhstan": 0.92, "Kostanay": 0.82,
            "North Kazakhstan": 0.80, "Akmola": 0.84, "Abai": 0.82, "Zhetisu": 0.76,
            "Almaty region": 0.85, "Zhambyl": 0.74, "Kyzylorda": 0.80,
        }
    )
    # Разрыв в оплате: зарплата женщин около 0.78 мужской (БНС;
    # уверенность средняя). Прежде медианы совпадали (persona-01).
    income_gender_factor: dict = field(default_factory=lambda: {"M": 1.12, "F": 0.88})

    # Пенсия — государственная выплата: не зависит ни от поселения, ни
    # от этапа жизни, средняя около 131–143 тыс. ₸ (research.md).
    # Прежде медиана 110 тыс. умножалась на множители поселения и этапа,
    # и 53% пенсионеров сидели на нижней границе 75 тыс. (persona-18).
    pension_income_median: int = 135_000
    pension_income_sigma: float = 0.30

    # Отрасль по региону: нефтегаз — в нефтяных регионах, горнодобыча —
    # в промышленных; бюджетник работает в бюджетной сфере
    # (persona-16: бюджетники в нефтегазе и IT).
    industry_region_factor: dict = field(
        default_factory=lambda: {
            "oil_and_gas": ({"Atyrau", "Mangystau", "West Kazakhstan", "Kyzylorda", "Aktobe"}, 5.0, 0.08),
            "mining": ({"Karaganda", "Pavlodar", "East Kazakhstan", "Ulytau", "Kostanay", "Abai"}, 3.0, 0.15),
        }
    )
    state_industries: tuple = ("education", "healthcare", "government", "science", "utilities", "security")

    # Сколько клиентов банка из сёл и регионов: веса поселений к
    # населению (БНС: сельских около 37%, Туркестанская 10.5% и
    # Алматинская 7.6% населения). Клиентская база банка городская
    # сильнее населения, поэтому поправка частичная.
    rural_weight_factor: float = 1.5
    region_weight_factor: dict = field(
        default_factory=lambda: {"Turkistan": 1.8, "Almaty region": 2.0, "Zhetisu": 1.3, "Zhambyl": 1.1,
                                 "Almaty": 0.85}
    )

    # Признаки участия: доля людей, которые вообще так тратят, по полу и
    # возрасту (синтетическое допущение; курение — по GATS Казахстан:
    # мужчины около 36%, женщины около 6%). Без признака категории нет
    # вовсе — структурный ноль. Прежде табак покупали 84% клиентов, а
    # ставки делали 59%, у женщин 65+ — 67% и 47% (аудит, C6).
    habit_shares: dict = field(
        default_factory=lambda: {
            "smokes": {"M": 0.36, "F": 0.06},
            "gambles": {"M": 0.12, "F": 0.025},
            "scooter": {"M": 0.30, "F": 0.22},
            "cinema": {"M": 0.45, "F": 0.50},
        }
    )
    # Как признак меняется с возрастом: (возраст от, множитель).
    habit_age_factor: dict = field(
        default_factory=lambda: {
            "smokes": ((18, 0.8), (25, 1.0), (55, 0.8), (70, 0.5)),
            "gambles": ((18, 1.3), (30, 1.0), (45, 0.5), (60, 0.15)),
            "scooter": ((18, 1.6), (30, 0.8), (40, 0.35), (55, 0.05)),
            "cinema": ((18, 1.4), (30, 1.0), (45, 0.6), (60, 0.25)),
        }
    )

    # Машина — по доходу: вероятность из транспортной смеси поселения
    # умножается на (доход / 320 тыс.)^0.45 в пределах [0.35, 1.8].
    # Прежде владение машиной падало с ростом дохода (persona-19).
    car_income_elasticity: float = 0.45

    # Структура трат по полу и возрасту (синтетическое допущение по
    # направлению эффектов; прежде пол и возраст не меняли ни одной
    # доли категорий — аудит, C6, persona-02).
    category_gender_factor: dict = field(
        default_factory=lambda: {
            "F": {"beauty": 2.0, "cosmetics": 2.4, "clothing": 1.25, "shoes": 1.2, "kids": 1.3,
                  "pharmacy": 1.15, "home_goods": 1.2},
            "M": {"beauty": 0.5, "cosmetics": 0.35, "car_service": 1.5, "electronics": 1.3, "sports": 1.3,
                  "fuel": 1.2, "fastfood": 1.15},
        }
    )
    # (возраст от, множители): берётся ступень с наибольшим «от» не
    # больше возраста.
    category_age_factor: tuple = (
        (18, {"education": 1.6, "fastfood": 1.3, "coffee": 1.2, "entertainment": 1.2, "sports": 1.2,
              "pharmacy": 0.8, "medical": 0.7}),
        (30, {}),
        (45, {"pharmacy": 1.3, "medical": 1.3, "lab": 1.2, "fastfood": 0.75, "coffee": 0.8,
              "entertainment": 0.7, "education": 0.5}),
        (60, {"pharmacy": 1.7, "medical": 1.6, "lab": 1.5, "fastfood": 0.45, "coffee": 0.5,
              "entertainment": 0.4, "marketplace": 0.6, "ecom": 0.6, "electronics": 0.6, "sports": 0.4,
              "education": 0.2, "kids": 0.3}),
    )
    # Где категория вообще есть: кикшеринг, прокат авто и кино — в
    # городах (прежде кикшеринг и прокат были у 95–100% клиентов
    # райцентров; аудит, context-12).
    category_settlement_types: dict = field(
        default_factory=lambda: {
            "micromobility": ("metropolis", "major_city", "regional_centre"),
            "car_rental": ("metropolis", "major_city"),
            "cinema": ("metropolis", "major_city", "regional_centre", "industrial_town", "small_town"),
        }
    )
    # Машинные траты — у водящих: парковка, автосервис, штрафы ГАИ
    # (прежде парковку платили 93.5% клиентов без машины; context-13).
    car_categories: dict = field(
        default_factory=lambda: {"fuel": 0.02, "parking": 0.03, "car_service": 0.03, "fines": 0.10}
    )

    # Обязательные расходы как доля дохода домохозяйства.
    mandatory_share_by_stage: dict = field(
        default_factory=lambda: {
            "young_adult": (0.35, 0.60),
            "early_career": (0.40, 0.65),
            "family": (0.48, 0.74),
            "mature": (0.42, 0.68),
            "retired": (0.45, 0.72),
        }
    )

    rent_share_of_income: tuple = (0.18, 0.34)

    # Роль Home Credit в жизни клиента.
    hcb_role_weights: dict = field(
        default_factory=lambda: {
            "primary": 0.26,
            "secondary": 0.34,
            "credit_only": 0.22,
            "deposit_only": 0.08,
            "episodic": 0.10,
        }
    )

    # Доля денежного оборота клиента, видимая этому банку.
    hcb_visible_share: dict = field(
        default_factory=lambda: {
            "primary": (0.72, 0.95),
            "secondary": (0.30, 0.60),
            "credit_only": (0.08, 0.30),
            "deposit_only": (0.05, 0.22),
            "episodic": (0.03, 0.18),
        }
    )

    activity_mode_weights: dict = field(
        default_factory=lambda: {
            "silent": 0.12,
            "rare": 0.20,
            "regular": 0.38,
            "high": 0.24,
            "extreme": 0.06,
        }
    )

    # Режим активности коррелирует с ролью банка.
    activity_mode_role_factor: dict = field(
        default_factory=lambda: {
            "primary": {"regular": 1.5, "high": 1.9, "extreme": 2.2, "rare": 0.35, "silent": 0.15},
            "secondary": {"regular": 1.1, "rare": 0.9, "silent": 0.7},
            "credit_only": {"silent": 1.6, "rare": 1.8, "high": 0.5, "extreme": 0.3},
            "deposit_only": {"silent": 2.4, "rare": 2.2, "high": 0.3, "extreme": 0.15},
            "episodic": {"silent": 2.8, "rare": 2.6, "regular": 0.3, "high": 0.15, "extreme": 0.05},
        }
    )

    # Клиент пришёл в банк до начала окна: стаж в месяцах.
    tenure_months_gamma: tuple = (2.0, 26.0)
    tenure_months_max: int = 168

    # Стаж на первой известной работе к её точке отсчёта — не
    # больше стольких месяцев и не раньше совершеннолетия.
    job_tenure_months_max: int = 240

    # Доля клиентов, регистрирующихся внутри окна наблюдения.
    registration_in_window_share: float = 0.18

    # Регистрация не позже, чем за столько дней до конца окна.
    registration_margin_days: int = 31

    # Доля тех, кто зарегистрировался и не начал пользоваться.
    registered_and_vanished_share: float = 0.14

