from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.generator.config import ARTIFACTS_DIR, PROCESSED_DIR, TOKENIZED_DIR


# ============================================================
# ИДЕЯ
# ============================================================
#
# Единое пространство ID: сначала special-токены, затем ключи
# полей, затем значения. Порядок назначения детерминированный и
# не зависит ни от частот, ни от порядка чтения файлов, поэтому
# добавление клиентов не переставляет уже выданные ID.
#
#   [0 .. 5]                      special
#   [6 .. 6 + n_keys)             ключи (namespace__field)
#   [6 + n_keys .. size)          значения, сгруппированные по ключу
#
# Значение принадлежит конкретному полю: "true" у
# transaction__is_online и "true" у communication__delivered это
# разные ID.
# ============================================================


FORMAT_VERSION = 1


# ------------------------------------------------------------
# SPECIAL TOKENS
# ------------------------------------------------------------
#
# ID фиксированы спецификацией и не меняются никогда.

PAD = "[PAD]"
UNK = "[UNK]"
MASK = "[MASK]"
EVT = "[EVT]"
USR = "[USR]"
MISSING = "[MISSING]"

SPECIAL_TOKENS: tuple[str, ...] = (PAD, UNK, MASK, EVT, USR, MISSING)

SPECIAL_IDS: dict[str, int] = {name: index for index, name in enumerate(SPECIAL_TOKENS)}

PAD_ID = SPECIAL_IDS[PAD]
UNK_ID = SPECIAL_IDS[UNK]
MASK_ID = SPECIAL_IDS[MASK]
EVT_ID = SPECIAL_IDS[EVT]
USR_ID = SPECIAL_IDS[USR]
MISSING_ID = SPECIAL_IDS[MISSING]

N_SPECIAL = len(SPECIAL_TOKENS)

SPECIAL_NOTES: dict[str, str] = {
    PAD: "зарезервирован; collate отдаёт плоский batch со смещениями, padding не используется",
    UNK: "неизвестный ключ или значение, которого нет во frozen vocab",
    MASK: "ставится только runtime-masker'ом; в сохранённых датасетах не встречается",
    EVT: "начало события, позиция 0; тот же ID и в key_ids, и в value_ids",
    USR: "начало профиля, позиция 0; тот же ID и в key_ids, и в value_ids",
    MISSING: "явный пропуск значения; key_id при этом настоящий ключ поля",
}


# ------------------------------------------------------------
# ИМЕНА ФАЙЛОВ
# ------------------------------------------------------------

KEY_VOCAB_FILE = "key_vocab.json"
VALUE_VOCAB_FILE = "value_vocab.json"
SPECIAL_TOKENS_FILE = "special_tokens.json"
FIELD_VALUE_IDS_FILE = "field_value_ids.json"
CONFIG_FILE = "tokenizer_config.json"
STATS_FILE = "tokenizer_stats.json"
GOLDEN_FILE = "golden_examples.json"

# Имя намеренно отличается от manifest.json у RAW.
DATASET_MANIFEST_FILE = "tokenized_manifest.json"

VOCAB_FILES: tuple[str, ...] = (
    SPECIAL_TOKENS_FILE,
    KEY_VOCAB_FILE,
    VALUE_VOCAB_FILE,
    FIELD_VALUE_IDS_FILE,
)

VOCAB_SUBDIR = "tokenizer"


# ------------------------------------------------------------
# ПУТИ
# ------------------------------------------------------------


def processed_dir(name: str) -> Path:
    return PROCESSED_DIR / name


def artifacts_dir(name: str) -> Path:
    return ARTIFACTS_DIR / name


def vocab_dir(name: str) -> Path:
    return ARTIFACTS_DIR / name / VOCAB_SUBDIR


def tokenized_dir(name: str) -> Path:
    return TOKENIZED_DIR / name


# ------------------------------------------------------------
# ОШИБКИ
# ------------------------------------------------------------


class IncompatibleArtifactsError(RuntimeError):
    """
    Artifacts не сходятся друг с другом: другой preprocessing,
    другая версия формата или правленый словарь.
    """


# ------------------------------------------------------------
# НАСТРОЙКИ ЗАПУСКА
# ------------------------------------------------------------


@dataclass(frozen=True)
class TokenizerSettings:
    """
    Лимиты берутся из split_manifest (то есть из manifest RAW),
    флаги CLI их переопределяют. Лимиты НЕ применяются к данным:
    превышения только считаются.
    """

    max_tokens_per_event: int | None = None
    max_events_per_history: int | None = None

    def resolve(self, manifest_limits: dict[str, int]) -> "ResolvedLimits":
        return ResolvedLimits(
            max_tokens_per_event=(
                self.max_tokens_per_event
                if self.max_tokens_per_event is not None
                else int(manifest_limits["max_tokens_per_event"])
            ),
            max_events_per_history=(
                self.max_events_per_history
                if self.max_events_per_history is not None
                else int(manifest_limits["max_events_per_history"])
            ),
            source=(
                "cli"
                if (self.max_tokens_per_event is not None or self.max_events_per_history is not None)
                else "split_manifest.raw"
            ),
        )


@dataclass(frozen=True)
class ResolvedLimits:
    max_tokens_per_event: int
    max_events_per_history: int
    source: str

    def as_dict(self) -> dict:
        return {
            "max_tokens_per_event": self.max_tokens_per_event,
            "max_events_per_history": self.max_events_per_history,
            "source": self.source,
            "rule": (
                "лимит события считается включая [EVT]; лимит истории считается в событиях. "
                "Ничего не обрезается: превышения попадают в статистику"
            ),
        }


DEFAULT_SETTINGS = TokenizerSettings()
