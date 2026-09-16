"""
Свойство: режим словаря меняет ТОЛЬКО token_id.

field_id, кандидаты полей, маски и локальные цели обязаны быть
одинаковыми во всех четырёх режимах — иначе арки сравнивали бы
разные задачи, а не разные словари.

Проверяется на настоящем маленьком наборе: все четыре словаря
строятся из одного прохода по train и каждым токенизируется свой
датасет, как это будет на реальном запуске.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.model.backbone import build_backbone
from src.model.config import config_from_tokenizer
from src.model.history_batching import (
    metadata_from_examples,
    prepare_history_batch,
    to_model_inputs,
)
from src.model.mlm_batching import build_targets, representations
from src.model.mlm_head import FieldTable, MLMHead, TargetError
from src.model.losses import mlm_loss
from src.tokenizer.artifacts import Tokenizer, read_modes
from src.tokenizer.config import FIELD_VALUE_IDS_FILE, KEY_VOCAB_FILE, N_SPECIAL, VALUE_VOCAB_FILE
from src.tokenizer.dataset import TokenizedDataset, collate
from src.tokenizer.masking import Masker, MaskingConfig
from src.tokenizer.run import MODE_TAGS, run as tokenizer_run
from src.tokenizer.semantics import (
    AMBIGUOUS_KEYS,
    DEFAULT_KEY_MODE,
    DEFAULT_VALUE_MODE,
    SEMANTIC_KEY_GROUPS,
    SemanticRegistryError,
    validate_registry,
)
from src.tokenizer.vocab import NO_FIELD, fit_counters


BASELINE = "baseline"
SEMKEYS = "semkeys"
SHARED = "shared"
BOTH = "semkeys_shared"

TAGS = tuple(tag for tag, _, _ in MODE_TAGS)


# ============================================================
# ЧЕТЫРЕ НАСТОЯЩИХ СЛОВАРЯ
# ============================================================


@pytest.fixture(scope="module")
def modes(tok_run, tmp_path_factory) -> dict:
    """
    Все четыре режима на одном и том же processed.

    Счётчики считаются один раз: они по полям и от режима не
    зависят. Каждый словарь токенизирует свой датасет — именно
    так арки и готовятся.
    """

    root = tmp_path_factory.mktemp("vocab_modes")

    counters = fit_counters(tok_run["processed"], tok_run["artifacts"])

    built: dict[str, dict] = {}

    for tag, key_mode, value_mode in MODE_TAGS:

        vocab_out = root / f"tokenizer__{tag}"
        out_dir = root / f"tokenized__{tag}"

        tokenizer_run(
            processed_in=tok_run["processed"],
            artifacts_in=tok_run["artifacts"],
            out_dir=out_dir,
            vocab_out=vocab_out,
            quiet=True,
            key_mode=key_mode,
            value_mode=value_mode,
            counters=counters,
        )

        tokenizer = Tokenizer.load(vocab_out, tok_run["artifacts"])

        built[tag] = {
            "tag": tag,
            "vocab_dir": vocab_out,
            "tokenized": out_dir,
            "tokenizer": tokenizer,
            "vocab": tokenizer.vocab,
            "table": FieldTable.load(tokenizer.vocab, vocab_out),
        }

    return built


def field_id(item: dict, key: str) -> int:
    return item["vocab"].field_entry(key).field_id


def key_token(item: dict, key: str) -> int:
    return item["vocab"].field_entry(key).key_token_id


def value_token(item: dict, key: str, value: str) -> int:
    return item["vocab"].value_token(field_id(item, key), value)


# ============================================================
# 1. BASELINE НЕ СЛОМАН
# ============================================================


def test_baseline_reproduces_the_previous_layout(modes, tok_run):
    """
    В baseline key token это N_SPECIAL + field_id, а кандидаты
    поля идут подряд. На этом стоит чтение датасетов и словарей,
    записанных до расщепления.
    """

    vocab = modes[BASELINE]["vocab"]

    assert vocab.is_baseline
    assert vocab.modes == {
        "key_mode": DEFAULT_KEY_MODE,
        "categorical_value_mode": DEFAULT_VALUE_MODE,
    }

    for entry in vocab.fields:
        assert entry.key_token_id == N_SPECIAL + entry.field_id
        assert entry.is_contiguous

    # Режим baseline ничего не склеивает: раскладка совпадает с
    # той, что даёт словарь по умолчанию.
    reference = Tokenizer.load(tok_run["vocab"], tok_run["artifacts"]).vocab

    assert vocab.size == reference.size
    assert [entry.key for entry in vocab.fields] == [entry.key for entry in reference.fields]
    assert [entry.candidates for entry in vocab.fields] == [
        entry.candidates for entry in reference.fields
    ]


def test_modes_are_read_from_the_config():
    """
    Режимы приходят из tokenizer_config и нигде не угадываются:
    конфиг без них это не baseline, а испорченный конфиг.
    """

    assert read_modes({"modes": {"key_mode": "semantic", "categorical_value_mode": "shared"}}) == (
        "semantic",
        "shared",
    )

    with pytest.raises(KeyError):
        read_modes({})


# ============================================================
# 2-3. КЛЮЧИ
# ============================================================


def test_semantic_group_shares_one_key_token_but_not_the_field(modes):

    for tag in (SEMKEYS, BOTH):

        item = modes[tag]

        left = "profile__declared_income"
        right = "profile_snapshot__declared_income"

        assert key_token(item, left) == key_token(item, right), tag
        assert field_id(item, left) != field_id(item, right), tag

        # Общий key token не делает поля одним полем.
        assert item["vocab"].field_entry(left).predictable is False
        assert item["vocab"].field_entry(right).predictable is True

    # В baseline они по-прежнему разные токены.
    for tag in (BASELINE, SHARED):
        item = modes[tag]
        assert key_token(item, "profile__declared_income") != key_token(
            item, "profile_snapshot__declared_income"
        ), tag


def test_every_registry_group_is_merged_and_nothing_else(modes):

    for tag in (SEMKEYS, BOTH):

        item = modes[tag]

        merged = {
            token.token_id: tuple(sorted(token.fields))
            for token in item["vocab"].key_tokens
            if token.shared
        }

        assert len(merged) == len(SEMANTIC_KEY_GROUPS), tag

        for _, members in SEMANTIC_KEY_GROUPS:
            tokens = {key_token(item, key) for key in members}
            assert len(tokens) == 1, (tag, members)


def test_ambiguous_keys_are_never_merged(modes):

    for tag in TAGS:

        item = modes[tag]

        for members, reason in AMBIGUOUS_KEYS:

            tokens = [key_token(item, key) for key in members]

            assert len(set(tokens)) == len(tokens), (tag, members, reason)


def test_registry_typo_is_an_error():
    with pytest.raises(SemanticRegistryError, match="которого нет в реестре"):
        validate_registry(["transaction__amount"])


# ============================================================
# 4-5. ЗНАЧЕНИЯ
# ============================================================


def test_same_categorical_value_shares_one_token(modes):

    left, right = "transaction__is_online", "communication__delivered"

    for tag in (SHARED, BOTH):

        item = modes[tag]

        assert value_token(item, left, "true") == value_token(item, right, "true"), tag

        # Поля при этом остаются разными, и локальный индекс у
        # каждого свой собственный.
        assert field_id(item, left) != field_id(item, right)

    for tag in (BASELINE, SEMKEYS):
        item = modes[tag]
        assert value_token(item, left, "true") != value_token(item, right, "true"), tag


def test_shared_value_is_one_entry_without_a_field(modes):
    """
    Общий токен это ОДИН объект словаря. field_id у записи
    значения нет: принадлежность это отношение многие-ко-многим.
    """

    item = modes[BOTH]

    token = value_token(item, "transaction__is_online", "true")

    entry = item["vocab"].values[token - item["vocab"].first_value_id]

    assert entry.shared is True
    assert not hasattr(entry, "field_id")
    assert len(entry.fields) > 1

    names = {item["vocab"].fields[index].key for index in entry.fields}

    assert {"transaction__is_online", "communication__delivered"} <= names


def test_numeric_buckets_never_merge(modes):

    for tag in TAGS:

        item = modes[tag]

        age = value_token(item, "profile__age", "7")
        amount = value_token(item, "transaction__amount", "7")

        assert age is not None and amount is not None, tag
        assert age != amount, tag

        # Целочисленные категории это тоже коды поля.
        dow = value_token(item, "communication__day_of_week", "1")
        children = value_token(item, "profile__children", "1")

        assert dow != children, tag


def test_numeric_local_index_stays_the_bucket_index(modes):

    for tag in TAGS:

        item = modes[tag]

        entry = item["vocab"].field_entry("transaction__amount")

        fields = np.full(entry.n_values, entry.field_id, dtype=np.int64)

        local = item["table"].to_local(fields, np.asarray(entry.candidates))

        assert np.array_equal(local, np.arange(entry.n_values)), tag


# ============================================================
# 6-7. РЕЖИМЫ И ARTIFACTS
# ============================================================


@pytest.mark.parametrize("tag", TAGS)
def test_every_mode_builds_a_consistent_vocab(modes, tag):

    vocab = modes[tag]["vocab"]

    assert vocab.n_fields == modes[BASELINE]["vocab"].n_fields

    # Токены покрывают своё пространство без дыр и повторов.
    assert [token.token_id for token in vocab.key_tokens] == list(
        range(N_SPECIAL, vocab.first_value_id)
    )
    assert [entry.id for entry in vocab.values] == list(range(vocab.first_value_id, vocab.size))

    covered = {token for entry in vocab.fields for token in entry.candidates}

    assert covered == set(range(vocab.first_value_id, vocab.size))


def test_sizes_shrink_exactly_by_what_was_merged(modes):

    base = modes[BASELINE]["vocab"]

    for tag in TAGS:

        vocab = modes[tag]["vocab"]
        report = vocab.sharing_report()

        merged_keys = sum(
            len(token.fields) - 1 for token in vocab.key_tokens if token.shared
        )
        merged_values = sum(
            len(entry.fields) - 1 for entry in vocab.values if len(entry.fields) > 1
        )

        assert vocab.n_key_tokens == base.n_key_tokens - merged_keys, tag
        assert vocab.n_values == base.n_values - merged_values, tag
        assert vocab.size == base.size - merged_keys - merged_values, tag

        if tag == BASELINE:
            assert report["n_merged_key_tokens"] == 0
            assert report["n_merged_value_tokens"] == 0
        else:
            assert report["n_merged_key_tokens"] + report["n_merged_value_tokens"] > 0, tag


@pytest.mark.parametrize("tag", TAGS)
def test_artifacts_round_trip(modes, tag):

    item = modes[tag]

    config = json.loads((item["vocab_dir"] / "tokenizer_config.json").read_text(encoding="utf-8"))

    assert read_modes(config) == (item["vocab"].key_mode, item["vocab"].value_mode)
    assert config["field_id_space"]["n_fields"] == item["vocab"].n_fields
    assert config["field_id_space"]["no_field"] == NO_FIELD
    assert config["semantic_registry"]["sha256"]

    keys = json.loads((item["vocab_dir"] / KEY_VOCAB_FILE).read_text(encoding="utf-8"))
    values = json.loads((item["vocab_dir"] / VALUE_VOCAB_FILE).read_text(encoding="utf-8"))

    assert keys["n_keys"] == item["vocab"].n_key_tokens
    assert keys["n_fields"] == item["vocab"].n_fields
    assert len(keys["key_tokens"]) == item["vocab"].n_key_tokens

    # У записи значения поля field_id нет ни в одном режиме.
    assert all("field_id" not in entry for entry in values["values"])

    stored = json.loads((item["vocab_dir"] / FIELD_VALUE_IDS_FILE).read_text(encoding="utf-8"))

    for entry in item["vocab"].fields:
        assert stored["fields"][entry.key]["value_ids"] == list(entry.candidates)

    # Перечитанный словарь это тот же словарь.
    again = Tokenizer.load(item["vocab_dir"]).vocab

    assert again.size == item["vocab"].size
    assert again.modes == item["vocab"].modes
    assert [e.candidates for e in again.fields] == [e.candidates for e in item["vocab"].fields]


# ============================================================
# 8. ДОМЕНЫ MLM НЕ СМЕШАЛИСЬ
# ============================================================


def test_candidate_domains_are_identical_in_every_mode(modes):
    """
    Кандидаты живут на поле, поэтому склейка токенов не имеет
    права их менять: ни состав, ни размер, ни набор голов.
    """

    base = modes[BASELINE]["table"]

    for tag in TAGS:

        table = modes[tag]["table"]

        assert np.array_equal(table.n_candidates, base.n_candidates), tag
        assert np.array_equal(table.predictable, base.predictable), tag
        assert table.trainable_field_ids == base.trainable_field_ids, tag
        assert table.degenerate_field_ids == base.degenerate_field_ids, tag


def test_heads_are_the_same_fields_in_every_mode(modes):

    base = modes[BASELINE]

    names = None

    for tag in TAGS:

        item = modes[tag]

        config = config_from_tokenizer(item["tokenizer"], d_model=16, n_heads=2, dim_feedforward=32)

        head = MLMHead(config, item["table"])

        sizes = {name: layer.out_features for name, layer in head.heads.items()}

        if names is None:
            names = sizes
        else:
            assert sizes == names, tag

        # Непредсказуемое поле головы не получает даже тогда,
        # когда делит key token с предсказуемым.
        hidden = item["vocab"].field_entry("profile__declared_income").field_id

        assert str(hidden) not in head.heads, tag

    assert names
    assert base["table"].size_of(base["table"].trainable_field_ids[0]) >= 2


def test_value_of_another_field_is_rejected_in_shared_mode(modes):
    """
    В shared-режиме чужое значение может лежать «внутри
    диапазона» поля. Отсев идёт по таблице кандидатов, а не по
    границам, поэтому оно всё равно отвергается.
    """

    item = modes[BOTH]

    table = item["table"]

    mcc = item["vocab"].field_entry("transaction__mcc")
    status = item["vocab"].field_entry("app_operation__status")

    foreign = np.array([mcc.candidates[0]], dtype=np.int64)

    with pytest.raises(TargetError, match="не входит в"):
        table.to_local(np.array([status.field_id], dtype=np.int64), foreign)


# ============================================================
# 9. ЗАДАЧА ОДНА, СЛОВАРИ РАЗНЫЕ
# ============================================================


def batch_of(item: dict, limit: int = 4, step: int = 0):

    data = TokenizedDataset(item["tokenized"], "train", vocab_dir=item["vocab_dir"])

    examples = [data.load(index) for index in range(limit)]

    masker = Masker(item["vocab"], MaskingConfig(mode="field_balanced", seed=11))

    history = prepare_history_batch(
        collate(examples), metadata_from_examples(examples), 64, masker=masker, step=step
    )

    return history, build_targets(history, item["table"])


def test_masks_and_local_targets_are_identical_in_every_mode(modes):
    """
    Ради этого расщепление и делалось.

    Masker работает по field_id, а он от режима не зависит,
    поэтому замаскированные позиции, поля целей и локальные
    ответы совпадают у всех четырёх. Вокабулярный отпечаток при
    этом обязан разойтись: token id у них разные.
    """

    invariant: dict[str, str] = {}
    vocabulary: dict[str, str] = {}

    for tag in TAGS:

        _, targets = batch_of(modes[tag])

        assert targets.n > 0, tag

        invariant[tag] = targets.field_digest()
        vocabulary[tag] = targets.digest()

    assert len(set(invariant.values())) == 1, invariant

    assert vocabulary[BASELINE] != vocabulary[BOTH]


def test_forward_and_backward_in_the_combined_mode(modes):

    item = modes[BOTH]

    history, targets = batch_of(item)

    config = config_from_tokenizer(
        item["tokenizer"], d_model=16, n_heads=2, dim_feedforward=32, dropout=0.0
    )

    backbone = build_backbone(config, seed=5)
    head = MLMHead(config, item["table"])

    tensors = targets.tensors()

    out = backbone(to_model_inputs(history, config), gather=targets.gather())

    field_logits = head(*representations(out, tensors), tensors["field_ids"])

    loss = mlm_loss(field_logits, tensors["local_targets"])

    assert torch.isfinite(loss.field_balanced)

    loss.field_balanced.backward()

    weight = backbone.pair.embeddings.token.weight

    assert weight.grad is not None
    assert torch.isfinite(weight.grad).all()

    # У склеенного токена градиент ненулевой: он действительно
    # участвует в предсказании.
    shared_token = value_token(item, "transaction__is_online", "true")

    assert float(weight.grad[shared_token].abs().sum()) > 0.0


def test_vocab_size_does_not_shift_the_rest_of_the_init(modes):
    """
    Таблица токенов разыгрывается из своего потока, поэтому
    размер словаря не двигает инициализацию остального. Иначе
    арки различались бы не только словарём.
    """

    made = {}

    for tag in (BASELINE, BOTH):

        config = config_from_tokenizer(
            modes[tag]["tokenizer"], d_model=16, n_heads=2, dim_feedforward=32
        )

        made[tag] = build_backbone(config, seed=21).state_dict()

    left, right = made[BASELINE], made[BOTH]

    assert set(left) == set(right)

    token = "pair.embeddings.token.weight"

    assert left[token].shape != right[token].shape

    for name in left:
        if name == token:
            continue
        assert torch.equal(left[name], right[name]), name
