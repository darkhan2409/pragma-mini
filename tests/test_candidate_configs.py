from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from src.masking.settings import MaskingConfig
from src.mlm.settings import MlmConfig
from src.tokenization.settings import TokenizerConfig


# ============================================================
# ИДЕЯ
# ============================================================
#
# Конфиги кандидатов аудита 2026-11-03 лежат в audit/…/configs и
# грузятся теми же from_dict, что и обучение. Каждый обязан быть
# валидным и отличаться от значений по умолчанию ровно своим
# переключателем: лишний ключ в конфиге кандидата сделал бы сравнение
# с эталоном нечестным. Группы кандидата 3 сверяются с заранее
# заданным списком. Тест читает только audit/, не data/.
# ============================================================


AUDIT = Path(__file__).resolve().parents[1] / "audit" / "2026-11-03-pretrain-fixes"
CONFIGS = AUDIT / "configs"

GROUPS = (
    ("direction", "transaction_amount"),
    ("app_domain", "firebase_screen", "operation"),
    ("campaign_code", "offer", "template", "purpose"),
    ("is_online", "operation_channel"),
    ("mcc", "merchant_category", "merchant_name"),
)

TRAIN = {
    "train-shuffle.json": "shuffle_order",
    "train-hide-keys.json": "hide_event_keys",
}


def changed(config, default) -> set[str]:
    """Ключи, значения которых отличаются от значений по умолчанию."""

    base = default.as_dict()

    return {key for key, value in config.as_dict().items() if value != base[key]}


def test_every_config_file_is_covered() -> None:
    """Новый файл в configs/ без проверки здесь не остаётся незамеченным."""

    expected = {*TRAIN, "masking-groups.json", "masking-event-type-only.json", "tokenizer-amount-by-type.json"}

    assert {path.name for path in CONFIGS.glob("*.json")} == expected


@pytest.mark.parametrize("name, flag", sorted(TRAIN.items()))
def test_train_config_enables_exactly_its_flag(name: str, flag: str) -> None:

    config = MlmConfig.load(CONFIGS / name)

    assert getattr(config, flag) is True
    assert changed(config, MlmConfig()) == {flag}
    assert set(json.loads((CONFIGS / name).read_text(encoding="utf-8"))) == {flag}


def test_masking_config_has_candidate_groups_and_event_type_flag() -> None:

    config = MaskingConfig.load(CONFIGS / "masking-groups.json")

    assert config.field_groups == GROUPS
    assert config.event_type_event_only is True
    assert changed(config, MaskingConfig()) == {"field_groups", "event_type_event_only"}


def test_event_type_only_masking_config_enables_exactly_its_flag() -> None:
    """Пара кандидату 4, если кандидат 3 отвергнут (спек §4)."""

    config = MaskingConfig.load(CONFIGS / "masking-event-type-only.json")

    assert config.event_type_event_only is True
    assert changed(config, MaskingConfig()) == {"event_type_event_only"}


def test_tokenizer_config_splits_amount_by_event_type() -> None:

    config = TokenizerConfig.load(CONFIGS / "tokenizer-amount-by-type.json")
    default = TokenizerConfig()

    assert default.numeric_encoders["transaction_amount"].split_by == "direction"
    assert config.numeric_encoders["transaction_amount"].split_by == "event_type"

    # Шкала суммы отличается от умолчания только split_by и пояснением reason к нему.
    def without_split(encoder) -> dict:
        return {key: value for key, value in asdict(encoder).items() if key not in ("split_by", "reason")}

    assert "event_type" in config.numeric_encoders["transaction_amount"].reason

    assert without_split(config.numeric_encoders["transaction_amount"]) == without_split(
        default.numeric_encoders["transaction_amount"]
    )

    # Остальные шкалы и все прочие поля — как по умолчанию.
    assert {k: v for k, v in config.numeric_encoders.items() if k != "transaction_amount"} == {
        k: v for k, v in default.numeric_encoders.items() if k != "transaction_amount"
    }
    assert changed(config, default) == {"numeric_encoders"}


def test_accepted_flags_file_names_real_config_fields() -> None:
    """
    accepted.json — принятые владельцем флаги, их сверяет chain.sh с действующими
    конфигами. Ключ не из полей конфига сделал бы каждый запуск отказом.
    """

    accepted = json.loads((AUDIT / "accepted.json").read_text(encoding="utf-8"))

    assert set(accepted) == {"mlm", "masking"}
    assert set(accepted["mlm"]) <= set(MlmConfig().as_dict())
    assert set(accepted["masking"]) <= set(MaskingConfig().as_dict())
