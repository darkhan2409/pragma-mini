from __future__ import annotations

from pathlib import Path

from src.preprocessing.artifacts import read_json, sha256_bytes, sha256_file, write_json
from src.preprocessing.build import events_schema, profile_schema
from src.preprocessing.buckets import RULE as BUCKET_RULE
from src.preprocessing.config import SCHEMA_VERSION

from .config import (
    CONFIG_FILE,
    FORMAT_VERSION,
    N_SPECIAL,
    SPECIAL_IDS,
    SPECIAL_NOTES,
    SPECIAL_TOKENS,
    VOCAB_FILES,
    IncompatibleArtifactsError,
    ResolvedLimits,
)
from .encode import (
    EVENT_FORMAT,
    EVENT_WIDTH,
    PROFILE_FORMAT,
    PROFILE_WIDTH,
    tokenized_events_schema,
    tokenized_profile_schema,
)
from .masking import MaskingConfig
from .semantics import read_modes, registry_as_dict, registry_digest
from .vocab import (
    FIELD_ID_RULE,
    KEY_FORMAT,
    NO_FIELD,
    ORDER_RULES,
    VALUE_RULES,
    CandidateIndex,
    FitReport,
    Vocab,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# tokenizer_config.json описывает всё, от чего зависят ID:
# версию формата, специальные ID, правила порядка и хэши тех
# artifacts preprocessing, на которых словарь обучался.
#
# При загрузке хэши пересчитываются. Несовпадение это ошибка, а
# не предупреждение: словарь, обученный на других границах
# корзин, молча даёт неверные значения.
# ============================================================


PREPROCESSING_ARTIFACTS: tuple[str, ...] = (
    "bucket_edges.json",
    "field_stats.json",
    "split_manifest.json",
)


def preprocessing_digests(artifacts_dir: Path) -> dict:
    """
    Отпечаток preprocessing, на котором обучен словарь.
    """

    artifacts_dir = Path(artifacts_dir)

    return {
        "schema_version": SCHEMA_VERSION,
        "sha256": {name: sha256_file(artifacts_dir / name) for name in PREPROCESSING_ARTIFACTS},
        "schemas_sha256": {
            "events": sha256_bytes(str(events_schema()).encode("utf-8")),
            "profile": sha256_bytes(str(profile_schema()).encode("utf-8")),
        },
    }


def vocab_digests(vocab_dir: Path) -> dict[str, str]:
    return {name: sha256_file(Path(vocab_dir) / name) for name in VOCAB_FILES}


def build_config(
    vocab: Vocab,
    vocab_dir: Path,
    artifacts_dir: Path,
    limits: ResolvedLimits,
    fit: FitReport,
) -> dict:
    """
    Собирается ПОСЛЕ записи файлов словаря: в него входят их хэши.
    """

    return {
        "format_version": FORMAT_VERSION,
        "special_tokens": {
            "ids": dict(SPECIAL_IDS),
            "order": list(SPECIAL_TOKENS),
            "notes": dict(SPECIAL_NOTES),
        },
        "id_layout": {
            "rule": "единое пространство: сначала special, затем ключи, затем значения",
            "special": [0, N_SPECIAL],
            "keys": [N_SPECIAL, vocab.first_value_id],
            "values": [vocab.first_value_id, vocab.size],
            "size": vocab.size,
        },
        "key_format": KEY_FORMAT,
        "modes": vocab.modes,
        "field_id_space": {
            "n_fields": vocab.n_fields,
            "no_field": NO_FIELD,
            "rule": FIELD_ID_RULE,
        },
        "semantic_registry": {
            "sha256": registry_digest(),
            **registry_as_dict(),
        },
        "sharing": vocab.sharing_report(),
        "order_rules": ORDER_RULES,
        "value_rules": {**VALUE_RULES, "bucket_rule": BUCKET_RULE},
        "event_format": {**EVENT_FORMAT, "width_by_type": dict(sorted(EVENT_WIDTH.items()))},
        "profile_format": {**PROFILE_FORMAT, "width": PROFILE_WIDTH},
        "limits": limits.as_dict(),
        "masking_defaults": MaskingConfig().as_dict(),
        "dtypes": {
            "events": str(tokenized_events_schema()),
            "profile": str(tokenized_profile_schema()),
        },
        "fit": fit.as_dict(),
        "preprocessing": preprocessing_digests(artifacts_dir),
        "vocab": {
            "n_fields": vocab.n_fields,
            "n_keys": vocab.n_key_tokens,
            "n_values": vocab.n_values,
            "size": vocab.size,
            "sha256": vocab_digests(vocab_dir),
        },
    }


def write_config(
    vocab: Vocab,
    vocab_dir: Path,
    artifacts_dir: Path,
    limits: ResolvedLimits,
    fit: FitReport,
) -> dict:

    config = build_config(vocab, vocab_dir, artifacts_dir, limits, fit)

    write_json(Path(vocab_dir) / CONFIG_FILE, config)

    return config


# ============================================================
# ЗАГРУЗКА
# ============================================================


class Tokenizer:
    """
    Frozen словарь плюс его config. Загрузка проверяет, что
    artifacts не разошлись.
    """

    def __init__(self, vocab: Vocab, config: dict):
        self.vocab = vocab
        self.config = config
        self.candidates: CandidateIndex = vocab.candidates()

    @property
    def size(self) -> int:
        return self.vocab.size

    @staticmethod
    def load(vocab_dir: Path, artifacts_dir: Path | None = None) -> "Tokenizer":

        vocab_dir = Path(vocab_dir)

        config_path = vocab_dir / CONFIG_FILE

        if not config_path.exists():
            raise IncompatibleArtifactsError(f"нет {CONFIG_FILE} в {vocab_dir}")

        config = read_json(config_path)

        if config.get("format_version") != FORMAT_VERSION:
            raise IncompatibleArtifactsError(
                f"tokenizer_config версии {config.get('format_version')}, ожидалась {FORMAT_VERSION}"
            )

        if config.get("special_tokens", {}).get("ids") != SPECIAL_IDS:
            raise IncompatibleArtifactsError("special-токены в config не совпадают с контрактом")

        actual = vocab_digests(vocab_dir)

        declared = config.get("vocab", {}).get("sha256", {})

        for name in VOCAB_FILES:
            if actual[name] != declared.get(name):
                raise IncompatibleArtifactsError(f"{name} изменён после записи словаря")

        if artifacts_dir is not None:

            fresh = preprocessing_digests(artifacts_dir)

            stored = config.get("preprocessing", {})

            if fresh["sha256"] != stored.get("sha256"):
                changed = sorted(
                    name
                    for name in PREPROCESSING_ARTIFACTS
                    if fresh["sha256"][name] != stored.get("sha256", {}).get(name)
                )
                raise IncompatibleArtifactsError(
                    f"artifacts preprocessing изменились после обучения словаря: {changed}"
                )

            if fresh["schemas_sha256"] != stored.get("schemas_sha256"):
                raise IncompatibleArtifactsError("схемы processed изменились после обучения словаря")

        vocab = Vocab.load(vocab_dir)

        if vocab.size != config.get("id_layout", {}).get("size"):
            raise IncompatibleArtifactsError("размер словаря не совпадает с config")

        key_mode, value_mode = read_modes(config)

        if (vocab.key_mode, vocab.value_mode) != (key_mode, value_mode):
            raise IncompatibleArtifactsError(
                f"режимы словаря {vocab.key_mode}/{vocab.value_mode} не совпадают с config "
                f"{key_mode}/{value_mode}"
            )

        # Правила склейки это часть словаря: их правка обязана
        # делать прежние artifacts несовместимыми, а не менять
        # смысл ID молча.
        stored_registry = config.get("semantic_registry", {}).get("sha256")

        if stored_registry is not None and stored_registry != registry_digest():
            raise IncompatibleArtifactsError(
                "реестр семантической склейки изменился после обучения словаря"
            )

        return Tokenizer(vocab, config)
