from __future__ import annotations

from datetime import date

import pytest

from tests.test_profile_state import (
    AGE_VALUES,
    EARLY,
    QUIET_SNAPSHOT,
    prepare,
    write_profile_vocab,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Возраст — точное целое число полных лет на cutoff, и в словаре
# каждый возраст это своё значение ключа profile_age, а не
# диапазон:
#
#   birth_date 1991-12-01, cutoff val (1 апреля 2026 у банка)
#     -> age 34 -> токен value:profile_age=34
#
# Значения словарь берёт только из train, как у любой категории:
# полного диапазона возрастов заранее не заводится, и невиданный
# на train возраст на val и test кодируется [UNK]. Остальные числа
# анкеты и событий по-прежнему режутся на диапазоны.
#
# Даты рождения среди токенов нет, пенсионера — тоже.
# ============================================================


def age_token(stage, born: date):
    """
    Возраст клиента на cutoff val и номер его значения в анкете.
    """

    from src.tokenization.encode import encode_profile
    from src.tokenization.finalvocab import FrozenArtifacts

    artifacts = FrozenArtifacts.load()

    history = prepare(stage, EARLY, dict(QUIET_SNAPSHOT, birth_date=born))

    record, _ = encode_profile(artifacts, history, 4)

    place = record.key_ids.index(artifacts.key_id("profile_age"))

    return artifacts, history.profile["profile_age"], record, record.value_ids[place]


def test_age_is_a_category_with_a_unit_not_a_number():

    from src.preprocessing.keys import CATEGORICAL, PROFILE_KEYS

    key = PROFILE_KEYS["age"]

    assert (key.key, key.kind, key.unit) == ("profile_age", CATEGORICAL, "years")


def test_age_has_no_bucket_edges_and_the_other_numbers_keep_theirs():
    """
    Шкалы объявлены ровно у прежних числовых ключей, кроме
    возраста. Доход по-прежнему режется по train квантилями.
    """

    from src.tokenization.settings import METHOD_QUANTILE, default_numeric_encoders

    encoders = default_numeric_encoders()

    assert "profile_age" not in encoders

    assert set(encoders) == {
        "transaction_amount", "amount_or_limit", "amount_due", "amount_paid",
        "principal_outstanding", "requested_amount", "approved_amount", "balance_after",
        "profile_declared_income", "profile_declared_income_old", "profile_declared_income_new",
        "profile_credit_limit", "days_past_due", "profile_relationship_months",
        "profile_credit_utilization", "rate", "original_amount",
    }

    assert encoders["profile_declared_income"].method == METHOD_QUANTILE


def test_neighbouring_ages_get_different_values(stage):
    """
    34 и 35, 35 и 39 — разные значения словаря: ни соседние годы,
    ни годы одного бывшего диапазона [35, 45) не схлопываются.
    """

    write_profile_vocab(stage)

    _, age34, _, id34 = age_token(stage, date(1991, 12, 1))
    _, age35, _, id35 = age_token(stage, date(1990, 12, 1))
    artifacts, age39, _, id39 = age_token(stage, date(1986, 12, 1))

    assert (age34, age35, age39) == (34, 35, 39)
    assert len({id34, id35, id39}) == 3

    assert artifacts.describe(id34) == "value:profile_age=34"
    assert artifacts.describe(id35) == "value:profile_age=35"
    assert artifacts.describe(id39) == "value:profile_age=39"


def test_age_stays_an_exact_integer(stage):
    """
    После препроцессинга возраст — int, после кодирования —
    значение ровно этого числа, а не диапазона.
    """

    write_profile_vocab(stage)

    artifacts, age, record, value = age_token(stage, date(1990, 1, 1))

    assert age == 36 and type(age) is int
    assert value == artifacts.categorical_id("profile_age", "36")
    assert record.positions[record.value_ids.index(value)] == 0

    names = [artifacts.describe(token) for token in record.value_ids + record.key_ids]

    assert not [name for name in names if "bucket" in name and "age" in name]


def test_age_unseen_on_train_becomes_unk(stage):

    from src.tokenization.specials import UNK

    write_profile_vocab(stage)

    assert "50" not in AGE_VALUES

    artifacts, age, _, value = age_token(stage, date(1976, 1, 1))

    assert age == 50
    assert value == artifacts.special(UNK)


def test_birth_date_and_pensioner_never_reach_the_tokens(stage):

    write_profile_vocab(stage)

    artifacts, _, record, _ = age_token(stage, date(1950, 1, 1))

    names = [artifacts.describe(token) for token in record.key_ids + record.value_ids]

    assert not [name for name in names if "birth" in name or "pensioner" in name]


def test_age_vocab_is_learned_on_train_only(stage):
    """
    В train клиентке 36, в val — 50. Словарь, собранный теми же
    функциями, что и fit, знает возраст 36 и не знает 50, а шкалы
    у возраста нет; у дохода она есть.
    """

    from src.tokenization.categorical import build_value_vocab
    from src.tokenization.fit import read_train
    from src.tokenization.keyvocab import build_key_vocab
    from src.tokenization.numeric import build_buckets
    from src.tokenization.schema import SemanticSchema
    from src.tokenization.settings import TokenizerConfig
    from src.tokenization.specials import build_special_tokens

    prepare(stage, EARLY, dict(QUIET_SNAPSHOT, birth_date=date(1990, 1, 1)), group="train")
    prepare(stage, EARLY, dict(QUIET_SNAPSHOT, birth_date=date(1976, 1, 1)))

    config = TokenizerConfig.load(None)
    schema = SemanticSchema.open()

    train = read_train(config, schema)

    key_vocab = build_key_vocab(build_special_tokens(), schema)
    value_vocab = build_value_vocab(train, key_vocab, config, schema)
    buckets, _ = build_buckets(train, value_vocab, config, schema)

    assert train.group == "train"
    assert list(value_vocab["profile_age"]) == ["36"]

    assert "profile_age" not in buckets
    assert "profile_declared_income" in buckets


def test_vocab_with_an_age_scale_is_refused(stage):
    """
    Словарь прежнего кода кодировал возраст диапазонами. Кодировать
    им нельзя: возраст молча стал бы диапазоном снова.
    """

    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.settings import TokenizerConfig
    from src.tokenization.transform import TransformError, encode_group

    write_profile_vocab(stage, age_scale=True)

    prepare(stage, EARLY, dict(QUIET_SNAPSHOT, birth_date=date(1990, 1, 1)))

    with pytest.raises(TransformError, match="profile_age"):
        encode_group(FrozenArtifacts.load(), "val", TokenizerConfig.load(None))
