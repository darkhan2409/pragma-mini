from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from src.generator.config import DATA_DIR
from src.preprocessing.artifacts import dumps_json, sha256_bytes

from .version import SCHEMA_VERSION


# ============================================================
# ИДЕЯ
# ============================================================
#
# Всё, что решает человек, а не данные, живёт здесь: какие
# значения считаются одним смыслом, каким методом резать каждое
# число, какие границы заданы бизнесом, что делать с нулём и
# минусом, как учить BPE.
#
# Отпечаток конфига входит в манифест артефактов, поэтому смена
# любого решения делает прежний словарь неактуальным явно, а не
# молча.
#
# Здесь нет ни одного значения, посчитанного по данным: границы
# квантилей считает этап 3 на train, а fallback ниже это
# заранее объявленная шкала на случай, когда наблюдений мало.
# Подбирать её по test запрещено.
# ============================================================


ARTIFACTS_DIR = DATA_DIR / "artifacts"
TOKENIZED_DIR = DATA_DIR / "tokenized"

# Каталог артефактов словаря внутри набора.
VOCAB_DIRNAME = "tokenizer"


# ------------------------------------------------------------
# ЧИСЛОВЫЕ ЭНКОДЕРЫ
# ------------------------------------------------------------

METHOD_QUANTILE = "quantile"
METHOD_FIXED = "fixed"
METHOD_UNFITTED = "unfitted"

METHODS: tuple[str, ...] = (METHOD_QUANTILE, METHOD_FIXED, METHOD_UNFITTED)

# Ноль: обычное значение диапазона или отдельный бакет. Отдельным
# он нужен там, где ноль это событие («платёж не внесён», «нет
# просрочки»), а не маленькое число.
ZERO_IN_RANGE = "in_range"
ZERO_SEPARATE = "separate"

ZERO_POLICIES: tuple[str, ...] = (ZERO_IN_RANGE, ZERO_SEPARATE)

# Минус: разрешён доменом или невозможен. Невозможное значение не
# попадает в крайний бакет, оно помечается [INVALID].
NEGATIVE_ALLOWED = "allowed"
NEGATIVE_INVALID = "invalid"

NEGATIVE_POLICIES: tuple[str, ...] = (NEGATIVE_ALLOWED, NEGATIVE_INVALID)


class ConfigError(ValueError):
    """
    Конфигурация токенизатора противоречива.
    """


@dataclass(frozen=True)
class NumericEncoder:
    """
    Правило кодирования одного числового ключа.

    method=quantile   границы считает train, bins это желаемое
                      число корзин, fallback применяется, когда
                      наблюдений меньше порога;
    method=fixed      границы заданы бизнесом и по данным не
                      двигаются;
    method=unfitted   кодировщика нет осознанно: значение
                      получает числовое [UNK] и попадает в
                      диагностику.
    """

    method: str
    bins: int | None = None
    boundaries: tuple[float, ...] = ()
    fallback: tuple[float, ...] = ()
    zero_policy: str = ZERO_IN_RANGE
    negative_policy: str = NEGATIVE_INVALID
    # Ключ, по значениям которого учатся границы. Нужен там, где
    # прежнее и новое значение поля профиля обязаны делить шкалу
    # с самим полем: иначе один доход получил бы три разные сетки.
    fit_source: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        # Границы хранятся дробными всегда, даже когда записаны
        # целыми числами. Иначе объект, прочитанный из
        # собственного файла, отличался бы от исходного одной
        # только записью, и контрольная сумма конфигурации
        # переставала бы сходиться.
        object.__setattr__(self, "boundaries", tuple(float(value) for value in self.boundaries))
        object.__setattr__(self, "fallback", tuple(float(value) for value in self.fallback))

    def validate(self, key: str) -> None:

        if self.method not in METHODS:
            raise ConfigError(f"ключ {key}: метод {self.method!r} не из {METHODS}")

        if self.zero_policy not in ZERO_POLICIES:
            raise ConfigError(f"ключ {key}: политика нуля {self.zero_policy!r} не из {ZERO_POLICIES}")

        if self.negative_policy not in NEGATIVE_POLICIES:
            raise ConfigError(f"ключ {key}: политика минуса {self.negative_policy!r} не из {NEGATIVE_POLICIES}")

        if self.method == METHOD_QUANTILE:
            if not self.bins or self.bins < 2:
                raise ConfigError(f"ключ {key}: у квантильного метода должно быть не меньше двух корзин")
            if self.boundaries:
                raise ConfigError(f"ключ {key}: квантильные границы считает train, задавать их руками нельзя")

        if self.method == METHOD_FIXED:
            if not self.boundaries:
                raise ConfigError(f"ключ {key}: у фиксированного метода обязаны быть границы")
            if self.fallback:
                raise ConfigError(f"ключ {key}: fixed не нуждается в fallback, границы уже заданы")

        if self.method == METHOD_UNFITTED and (self.boundaries or self.fallback or self.bins):
            raise ConfigError(f"ключ {key}: unfitted означает отсутствие шкалы, границ у него нет")

        for name, values in (("boundaries", self.boundaries), ("fallback", self.fallback)):
            if values and list(values) != sorted(set(values)):
                raise ConfigError(f"ключ {key}: {name} обязаны строго возрастать без повторов")

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "bins": self.bins,
            "boundaries": list(self.boundaries),
            "fallback": list(self.fallback),
            "zero_policy": self.zero_policy,
            "negative_policy": self.negative_policy,
            "fit_source": self.fit_source,
            "reason": self.reason,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "NumericEncoder":

        unknown = set(data) - set(NumericEncoder(method=METHOD_UNFITTED).as_dict())

        if unknown:
            raise ConfigError(f"неизвестные поля кодировщика: {sorted(unknown)}")

        return NumericEncoder(
            method=str(data["method"]),
            bins=None if data.get("bins") is None else int(data["bins"]),
            boundaries=tuple(float(value) for value in data.get("boundaries", ())),
            fallback=tuple(float(value) for value in data.get("fallback", ())),
            zero_policy=str(data.get("zero_policy", ZERO_IN_RANGE)),
            negative_policy=str(data.get("negative_policy", NEGATIVE_INVALID)),
            fit_source=data.get("fit_source"),
            reason=str(data.get("reason", "")),
        )


def _quantile(bins: int, fallback: tuple[float, ...], reason: str,
              zero: str = ZERO_SEPARATE, negative: str = NEGATIVE_INVALID,
              fit_source: str | None = None) -> NumericEncoder:
    return NumericEncoder(
        method=METHOD_QUANTILE,
        bins=bins,
        fallback=fallback,
        zero_policy=zero,
        negative_policy=negative,
        fit_source=fit_source,
        reason=reason,
    )


def _fixed(boundaries: tuple[float, ...], reason: str,
           zero: str = ZERO_IN_RANGE, negative: str = NEGATIVE_INVALID,
           fit_source: str | None = None) -> NumericEncoder:
    return NumericEncoder(
        method=METHOD_FIXED,
        boundaries=boundaries,
        zero_policy=zero,
        negative_policy=negative,
        fit_source=fit_source,
        reason=reason,
    )


# Заранее объявленные шкалы на случай, когда train-наблюдений
# меньше порога. Это не подгонка под данные: шкалы взяты из
# смысла величины (тенге, доли, часы) и от набора не зависят.
FALLBACK_KZT: tuple[float, ...] = (1_000, 5_000, 20_000, 50_000, 200_000, 1_000_000)
FALLBACK_RATIO: tuple[float, ...] = (0.01, 0.05, 0.2, 0.5, 1.0, 2.0)
FALLBACK_HOURS: tuple[float, ...] = (1, 6, 24, 72, 168, 720)


def default_numeric_encoders() -> dict[str, NumericEncoder]:
    """
    Кодировщик каждого числового ключа смыслового реестра.

    Деньги, отношения и интервалы режутся по train: их разброс
    свойство популяции. Сроки, возраст, ставка, доли и счётчики
    заданы бизнесом: их границы известны заранее и от выборки
    зависеть не должны.
    """

    money = "сумма в тенге: разброс свойство популяции, границы считает train"
    ratio = "безразмерное отношение: границы считает train"
    interval = "интервал между событиями в часах: границы считает train"

    encoders: dict[str, NumericEncoder] = {
        # --- деньги события ---
        "transaction_amount": _quantile(16, FALLBACK_KZT, money),
        "amount_or_limit": _quantile(12, FALLBACK_KZT, money),
        "amount_due": _quantile(12, FALLBACK_KZT, money),
        "amount_paid": _quantile(12, FALLBACK_KZT, money),
        "principal_outstanding": _quantile(12, FALLBACK_KZT, money),
        "requested_amount": _quantile(12, FALLBACK_KZT, money),
        "approved_amount": _quantile(12, FALLBACK_KZT, money),
        "balance_after": _quantile(
            16, FALLBACK_KZT,
            "остаток счёта: минус законен, это долг по карте или овердрафт",
            negative=NEGATIVE_ALLOWED,
        ),
        # --- деньги профиля ---
        "profile_declared_income": _quantile(12, FALLBACK_KZT, money),
        "profile_declared_income_old": _quantile(
            12, FALLBACK_KZT,
            "прежний доход делит шкалу с самим доходом: иначе одно и то же число попало бы в разные корзины",
            fit_source="profile_declared_income",
        ),
        "profile_declared_income_new": _quantile(
            12, FALLBACK_KZT,
            "новый доход делит шкалу с самим доходом",
            fit_source="profile_declared_income",
        ),
        "profile_credit_limit": _quantile(12, FALLBACK_KZT, money),
        # --- отношения ---
        "amount_to_declared_income": _quantile(12, FALLBACK_RATIO, ratio),
        "amount_to_limit": _quantile(12, FALLBACK_RATIO, ratio),
        "amount_to_balance_after": _quantile(
            12, FALLBACK_RATIO,
            "отношение к остатку: минус законен, остаток бывает отрицательным",
            negative=NEGATIVE_ALLOWED,
        ),
        "amount_to_client_average": _quantile(12, FALLBACK_RATIO, ratio),
        # --- интервалы ---
        "since_previous_hours": _quantile(12, FALLBACK_HOURS, interval, zero=ZERO_IN_RANGE),
        "since_same_type_hours": _quantile(12, FALLBACK_HOURS, interval, zero=ZERO_IN_RANGE),
        "since_last_income_hours": _quantile(12, FALLBACK_HOURS, interval, zero=ZERO_IN_RANGE),
        # --- время в днях: шкалы бизнеса ---
        # Граница 1 здесь была бы лишней: ноль уже отдельный
        # диапазон, а между нулём и единицей целых дней нет.
        "days_past_due": _fixed(
            (30, 60, 90, 120, 180),
            "полосы просрочки банка: ноль это отдельное состояние «просрочки нет»",
            zero=ZERO_SEPARATE,
        ),
        "days_to_due": _fixed(
            (-30, -7, 0, 1, 7, 30),
            "дней до планового платежа: минус это просрочка, и она законна",
            negative=NEGATIVE_ALLOWED,
        ),
        "days_since_related_event": _fixed(
            (1, 7, 30, 90),
            "расстояние до события-причины: секунды, сутки, неделя, месяц",
            zero=ZERO_SEPARATE,
        ),
        "age_of_history_days": _fixed(
            (30, 90, 180, 365, 730, 1095),
            "возраст наблюдаемой истории клиента",
        ),
        # --- сроки ---
        "term": _fixed((3, 6, 12, 24, 36, 60), "типовые сроки договоров в месяцах"),
        "requested_term": _fixed((3, 6, 12, 24, 36, 60), "запрошенный срок в месяцах"),
        "approved_term": _fixed((3, 6, 12, 24, 36, 60), "одобренный срок в месяцах"),
        "profile_relationship_months": _fixed(
            (6, 12, 24, 36, 60, 120), "длительность отношений с банком в месяцах"
        ),
        # --- возраст и доли ---
        "profile_age": _fixed((25, 30, 35, 45, 55, 65), "возрастные группы"),
        "profile_credit_utilization": _fixed(
            (0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
            "использование лимита долей: ноль это отдельное состояние «лимитом не пользуются»",
            zero=ZERO_SEPARATE,
        ),
        "rate": _fixed((0.05, 0.10, 0.15, 0.20, 0.25, 0.35), "годовая ставка долей"),
        # --- счётчики ---
        # У счётчиков ноль уже отдельный диапазон, поэтому первая
        # граница начинается с двойки: интервал между нулём и
        # единицей пуст по природе целого числа.
        "installment_no": _fixed((2, 4, 7, 13, 25), "номер платежа в графике", zero=ZERO_SEPARATE),
        "profile_children": _fixed((2, 3, 4), "число детей", zero=ZERO_SEPARATE),
        "profile_contracts_count": _fixed((2, 3, 5), "число договоров", zero=ZERO_SEPARATE),
        "profile_active_contracts": _fixed((2, 3, 5), "число действующих договоров", zero=ZERO_SEPARATE),
        # --- без шкалы ---
        "original_amount": NumericEncoder(
            method=METHOD_UNFITTED,
            reason=(
                "единица значения лежит в соседнем ключе original_currency: квантили по смеси валют "
                "смысла не имеют, а пересчёта по курсу в данных нет"
            ),
        ),
    }

    for name in ("profile_children_old", "profile_children_new"):
        encoders[name] = _fixed(
            (2, 3, 4),
            "прежнее и новое число детей делят шкалу с самим полем профиля",
            zero=ZERO_SEPARATE,
            fit_source="profile_children",
        )

    return encoders


# ------------------------------------------------------------
# ДОМЕНЫ КАТЕГОРИАЛЬНЫХ ЗНАЧЕНИЙ
# ------------------------------------------------------------


@dataclass(frozen=True)
class ValueDomain:
    """
    Несколько ключей, значения которых берутся из одного
    множества и потому делят коды.
    """

    name: str
    keys: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict:
        return {"name": self.name, "keys": list(self.keys), "reason": self.reason}

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "ValueDomain":

        unknown = set(data) - {"name", "keys", "reason"}

        if unknown:
            raise ConfigError(f"неизвестные поля домена: {sorted(unknown)}")

        return ValueDomain(str(data["name"]), tuple(str(key) for key in data["keys"]), str(data.get("reason", "")))


def default_value_domains() -> tuple[ValueDomain, ...]:
    """
    Объединения, у каждого из которых названа причина.

    По умолчанию домен это сам ключ: одинаковое написание ещё
    ничего не значит. Здесь перечислено только то, где множество
    значений доказуемо одно.
    """

    return (
        ValueDomain(
            "event_type_domain",
            ("event_type", "related_event_type"),
            "тип события и тип события-причины берутся из одного перечня типов ленты",
        ),
    )


# Объединения, которые напрашиваются, но в V1 не делаются.
# Список ведётся руками: он объясняет решение, а не прячет его.
DECLINED_DOMAINS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("profile_<поле>", "profile_<поле>_old", "profile_<поле>_new"),
        "смысл один, но физический тип разный: у профиля булево и число, у изменения профиля строка. "
        "Пока смысловой слой не типизирует old_value и new_value по самому полю, общий домен склеил бы "
        "true и \"true\" в разные коды под одним смыслом",
    ),
    (
        ("merchant_city", "profile_city"),
        "город точки это место покупки, а город профиля место жизни: объявлено несовместимым в реестре",
    ),
)


# ------------------------------------------------------------
# BPE
# ------------------------------------------------------------


@dataclass(frozen=True)
class BpeConfig:
    """
    Байтовый BPE: полный алфавит из 256 байт, обучение только на
    разрешённых train-текстах.
    """

    vocab_size: int = 4096
    min_frequency: int = 2
    # Пробел перед первым куском не добавляется: иначе decode
    # вернул бы не тот текст, что подавали.
    add_prefix_space: bool = False
    # Разбиение по словам до слияний: без него куски склеиваются
    # через пробел и перестают быть частями слова.
    use_regex: bool = True

    def as_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "add_prefix_space": self.add_prefix_space,
            "use_regex": self.use_regex,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "BpeConfig":

        unknown = set(data) - set(BpeConfig().as_dict())

        if unknown:
            raise ConfigError(f"неизвестные поля BPE: {sorted(unknown)}")

        base = BpeConfig()

        return BpeConfig(
            vocab_size=int(data.get("vocab_size", base.vocab_size)),
            min_frequency=int(data.get("min_frequency", base.min_frequency)),
            add_prefix_space=bool(data.get("add_prefix_space", base.add_prefix_space)),
            use_regex=bool(data.get("use_regex", base.use_regex)),
        )


# ------------------------------------------------------------
# КОНФИГ
# ------------------------------------------------------------


@dataclass(frozen=True)
class TokenizerConfig:

    schema_version: int = SCHEMA_VERSION

    # Группа, на которой разрешено учиться. Другой у V1 нет:
    # разрешение выдаёт разделение препроцессинга.
    fit_group: str = "train"

    # Сколько значений числового ключа держать в выборке для
    # квантилей. Выборка bottom-k по хэшу единицы дедупликации:
    # она не зависит ни от порядка чтения, ни от seed.
    quantile_sample_k: int = 200_000
    quantile_algorithm: str = "inverted_cdf"

    # Меньше этого наблюдений — границы по train не считаются, и
    # ключ уходит на объявленный fallback.
    numeric_min_values: int = 50
    numeric_min_clients: int = 5

    # Порог редкости категорий. None значит «все наблюдавшиеся
    # категории сохраняются целиком»; поиска «лучшего порога» нет.
    rare_min_count: int | None = None

    # Предел точного подсчёта различных числовых значений.
    distinct_cap: int = 100_000

    # Ключи, которые реестр называет текстом, а конфигурация
    # приказывает кодировать целиком. Пусто по умолчанию:
    # токенизатор следует реестру и лишь называет противоречие.
    text_keys_as_categorical: tuple[str, ...] = ()

    value_domains: tuple[ValueDomain, ...] = field(default_factory=default_value_domains)

    numeric_encoders: dict[str, NumericEncoder] = field(default_factory=default_numeric_encoders)

    bpe: BpeConfig = field(default_factory=BpeConfig)

    # Больше этого кусков в одном значении — явная ошибка, а не
    # молчаливая обрезка текста.
    max_pieces_per_value: int = 256

    # --------------------------------------------------------

    def validate(self) -> None:

        for key, encoder in sorted(self.numeric_encoders.items()):
            encoder.validate(key)

            if encoder.fit_source is not None and encoder.fit_source not in self.numeric_encoders:
                raise ConfigError(f"ключ {key}: источник шкалы {encoder.fit_source!r} не объявлен")

        seen: dict[str, str] = {}

        for domain in self.value_domains:

            if len(domain.keys) < 2:
                raise ConfigError(f"домен {domain.name}: объединять нечего")

            for key in domain.keys:
                if key in seen:
                    raise ConfigError(f"ключ {key} объявлен в двух доменах: {seen[key]} и {domain.name}")
                seen[key] = domain.name

        if self.rare_min_count is not None and self.rare_min_count < 2:
            raise ConfigError("порог редкости меньше двух не имеет смысла")

        if self.quantile_sample_k < 1000:
            raise ConfigError("выборка для квантилей меньше тысячи значений не даёт устойчивых границ")

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "fit_group": self.fit_group,
            "quantile_sample_k": self.quantile_sample_k,
            "quantile_algorithm": self.quantile_algorithm,
            "numeric_min_values": self.numeric_min_values,
            "numeric_min_clients": self.numeric_min_clients,
            "rare_min_count": self.rare_min_count,
            "distinct_cap": self.distinct_cap,
            "text_keys_as_categorical": list(self.text_keys_as_categorical),
            "value_domains": [domain.as_dict() for domain in self.value_domains],
            "declined_domains": [{"keys": list(keys), "reason": reason} for keys, reason in DECLINED_DOMAINS],
            "numeric_encoders": {key: encoder.as_dict() for key, encoder in sorted(self.numeric_encoders.items())},
            "bpe": self.bpe.as_dict(),
            "max_pieces_per_value": self.max_pieces_per_value,
        }

    def sha256(self) -> str:
        return sha256_bytes(dumps_json(self.as_dict()).encode("utf-8"))

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "TokenizerConfig":

        base = TokenizerConfig()
        declared = base.as_dict()

        unknown = set(data) - set(declared)

        if unknown:
            raise ConfigError(f"неизвестные ключи конфига: {sorted(unknown)}")

        # declined_domains это запись решения, а не настройка.
        # Прочитать её из собственного артефакта можно — иначе
        # записанный нами же файл не читался бы обратно, — а
        # поменять нельзя.
        if "declined_domains" in data and data["declined_domains"] != declared["declined_domains"]:
            raise ConfigError(
                "declined_domains это запись принятого решения, а не настройка: "
                "через файл конфигурации она не меняется"
            )

        encoders = dict(base.numeric_encoders)

        for key, item in data.get("numeric_encoders", {}).items():
            encoders[str(key)] = NumericEncoder.from_dict(item)

        domains = (
            tuple(ValueDomain.from_dict(item) for item in data["value_domains"])
            if "value_domains" in data
            else base.value_domains
        )

        config = replace(
            base,
            fit_group=str(data.get("fit_group", base.fit_group)),
            quantile_sample_k=int(data.get("quantile_sample_k", base.quantile_sample_k)),
            quantile_algorithm=str(data.get("quantile_algorithm", base.quantile_algorithm)),
            numeric_min_values=int(data.get("numeric_min_values", base.numeric_min_values)),
            numeric_min_clients=int(data.get("numeric_min_clients", base.numeric_min_clients)),
            rare_min_count=(
                None if data.get("rare_min_count") is None else int(data["rare_min_count"])
            ),
            distinct_cap=int(data.get("distinct_cap", base.distinct_cap)),
            text_keys_as_categorical=tuple(str(key) for key in data.get("text_keys_as_categorical", ())),
            value_domains=domains,
            numeric_encoders=encoders,
            bpe=BpeConfig.from_dict(data["bpe"]) if "bpe" in data else base.bpe,
            max_pieces_per_value=int(data.get("max_pieces_per_value", base.max_pieces_per_value)),
        )

        config.validate()

        return config

    @staticmethod
    def load(path: Path | None) -> "TokenizerConfig":

        if path is None:
            config = TokenizerConfig()
            config.validate()
            return config

        return TokenizerConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def vocab_dir(name: str) -> Path:
    return ARTIFACTS_DIR / name / VOCAB_DIRNAME


def tokenized_dir(name: str) -> Path:
    return TOKENIZED_DIR / name


__all__ = [
    "ARTIFACTS_DIR",
    "BpeConfig",
    "ConfigError",
    "DECLINED_DOMAINS",
    "METHOD_FIXED",
    "METHOD_QUANTILE",
    "METHOD_UNFITTED",
    "NEGATIVE_ALLOWED",
    "NEGATIVE_INVALID",
    "NumericEncoder",
    "TOKENIZED_DIR",
    "TokenizerConfig",
    "VOCAB_DIRNAME",
    "ValueDomain",
    "ZERO_IN_RANGE",
    "ZERO_SEPARATE",
    "default_numeric_encoders",
    "default_value_domains",
    "tokenized_dir",
    "vocab_dir",
]
