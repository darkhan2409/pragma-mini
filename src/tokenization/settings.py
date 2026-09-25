from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from src.generator.config import DATA_DIR



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


# Пути стандартны и в командах не задаются.
#
#   data/03_vocab/               чем кодируются данные: шесть файлов словаря
#   data/04_tokenized/<group>/   результат кодирования, два файла
#
# Словарь один на весь конвейер и учится только на train,
# поэтому лежит отдельно от групп.
VOCAB_DIR = DATA_DIR / "03_vocab"
TOKENIZED_DIR = DATA_DIR / "04_tokenized"

SPECIAL_TOKENS_FILE = "special_tokens.json"
KEY_VOCAB_FILE = "key_vocab.json"
VALUE_VOCAB_FILE = "value_vocab.json"
BUCKETS_FILE = "buckets.json"
BPE_FILE = "bpe.json"
FINAL_VOCAB_FILE = "final_vocab.json"


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

# Минус: разрешён доменом или невозможен. Невозможное значение в
# крайний бакет не попадает: шкала на нём не учится, а при
# кодировании оно становится [UNK].
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


def default_numeric_encoders() -> dict[str, NumericEncoder]:
    """
    Кодировщик каждого числового ключа смыслового реестра.

    Деньги режутся по train: их разброс свойство популяции.
    Ставка, доли и длительность отношений заданы бизнесом: их
    границы известны заранее и от выборки зависеть не должны.

    Сроков, счётчиков и возраста здесь нет вовсе: у них важно
    точное значение, а не порядок величины, поэтому реестр объявил
    их категориями и каждое число получает свой токен.
    """

    money = "сумма в тенге: разброс свойство популяции, границы считает train"

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
        # --- время в днях: шкалы бизнеса ---
        # Граница 1 здесь была бы лишней: ноль уже отдельный
        # диапазон, а между нулём и единицей целых дней нет.
        "days_past_due": _fixed(
            (30, 60, 90, 120, 180),
            "полосы просрочки банка: ноль это отдельное состояние «просрочки нет»",
            zero=ZERO_SEPARATE,
        ),
        # --- сроки ---
        "profile_relationship_months": _fixed(
            (6, 12, 24, 36, 60, 120), "длительность отношений с банком в месяцах"
        ),
        # --- доли ---
        "profile_credit_utilization": _fixed(
            (0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
            "использование лимита долей: ноль это отдельное состояние «лимитом не пользуются»",
            zero=ZERO_SEPARATE,
        ),
        # Ставки объявлены каталогом продуктов и лежат в [0.06, 0.24],
        # сгущаясь к 0.17. Прежние границы (0.05 … 0.35) сводили
        # почти все объявленные ставки в одну корзину [0.15, 0.20).
        "rate": _fixed((0.10, 0.14, 0.16, 0.17, 0.18, 0.20), "годовая ставка долей"),
        # --- без шкалы ---
        "original_amount": NumericEncoder(
            method=METHOD_UNFITTED,
            reason=(
                "единица значения лежит в соседнем ключе original_currency: квантили по смеси валют "
                "смысла не имеют, а пересчёта по курсу в данных нет"
            ),
        ),
    }

    return encoders


# ------------------------------------------------------------
# ДОМЕНЫ КАТЕГОРИАЛЬНЫХ ЗНАЧЕНИЙ
# ------------------------------------------------------------


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

    # Предел точного подсчёта различных числовых значений.
    distinct_cap: int = 100_000

    # Ключи, которые реестр называет текстом, а конфигурация
    # приказывает кодировать целиком. Пусто по умолчанию:
    # токенизатор следует реестру и лишь называет противоречие.
    text_keys_as_categorical: tuple[str, ...] = ()

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

        if self.quantile_sample_k < 1000:
            raise ConfigError("выборка для квантилей меньше тысячи значений не даёт устойчивых границ")

    def as_dict(self) -> dict:
        return {
            "fit_group": self.fit_group,
            "quantile_sample_k": self.quantile_sample_k,
            "quantile_algorithm": self.quantile_algorithm,
            "numeric_min_values": self.numeric_min_values,
            "numeric_min_clients": self.numeric_min_clients,
            "distinct_cap": self.distinct_cap,
            "text_keys_as_categorical": list(self.text_keys_as_categorical),
            "numeric_encoders": {key: encoder.as_dict() for key, encoder in sorted(self.numeric_encoders.items())},
            "bpe": self.bpe.as_dict(),
            "max_pieces_per_value": self.max_pieces_per_value,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "TokenizerConfig":

        base = TokenizerConfig()
        declared = base.as_dict()

        unknown = set(data) - set(declared)

        if unknown:
            raise ConfigError(f"неизвестные ключи конфига: {sorted(unknown)}")

        encoders = dict(base.numeric_encoders)

        for key, item in data.get("numeric_encoders", {}).items():
            encoders[str(key)] = NumericEncoder.from_dict(item)

        config = replace(
            base,
            fit_group=str(data.get("fit_group", base.fit_group)),
            quantile_sample_k=int(data.get("quantile_sample_k", base.quantile_sample_k)),
            quantile_algorithm=str(data.get("quantile_algorithm", base.quantile_algorithm)),
            numeric_min_values=int(data.get("numeric_min_values", base.numeric_min_values)),
            numeric_min_clients=int(data.get("numeric_min_clients", base.numeric_min_clients)),
            distinct_cap=int(data.get("distinct_cap", base.distinct_cap)),
            text_keys_as_categorical=tuple(str(key) for key in data.get("text_keys_as_categorical", ())),
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


def vocab_path(name: str) -> Path:
    """
    Путь к файлу словаря в data/03_vocab.
    """

    return VOCAB_DIR / name


def tokenized_dir(group: str) -> Path:
    """
    Каталог закодированной группы.
    """

    return TOKENIZED_DIR / group


__all__ = [
    "BPE_FILE",
    "BUCKETS_FILE",
    "KEY_VOCAB_FILE",
    "SPECIAL_TOKENS_FILE",
    "TOKENIZED_DIR",
    "VOCAB_DIR",
    "FINAL_VOCAB_FILE",
    "VALUE_VOCAB_FILE",
    "BpeConfig",
    "ConfigError",
    "METHOD_FIXED",
    "METHOD_QUANTILE",
    "METHOD_UNFITTED",
    "NEGATIVE_ALLOWED",
    "NEGATIVE_INVALID",
    "NumericEncoder",
    "TokenizerConfig",
    "ZERO_IN_RANGE",
    "ZERO_SEPARATE",
    "default_numeric_encoders",
    "tokenized_dir",
    "vocab_path",
]
