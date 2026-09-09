"""
MLM head: кандидаты полей, адреса замаскированных позиций и три
представления из ОДНОГО прохода.

Главное, что здесь проверяется, это соответствие. Ошибка в
адресации не падает и не заметна по loss: голова просто учится
не тому.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta

import numpy as np
import pytest
import torch

from src.tokenizer.artifacts import Tokenizer
from src.tokenizer.config import (
    EVT_ID,
    FIELD_VALUE_IDS_FILE,
    MISSING_ID,
    USR_ID,
    IncompatibleArtifactsError,
)
from src.tokenizer.dataset import Events, Example, TokenizedDataset, collate
from src.tokenizer.encode import encode_pairs
from src.tokenizer.masking import Masker, MaskingConfig
from src.tokenizer.vocab import KeyEntry, ValueEntry, Vocab
from src.model.backbone import build_backbone
from src.model.batching import BatchError
from src.model.config import ModelConfig, config_from_tokenizer
from src.model.history_batching import (
    metadata_from_examples,
    prepare_history_batch,
    to_model_inputs,
)
from src.model.losses import mlm_loss
from src.model.mlm_batching import build_targets, representations
from src.model.mlm_head import FieldTable, MLMHead, TargetError

from tests.test_tok_encode import toy_vocab


BASE = datetime(2025, 1, 1)

COLORS = ("red", "blue")


# ============================================================
# СИНТЕТИКА С ДВУМЯ ПРЕДСКАЗУЕМЫМИ ПОЛЯМИ
# ============================================================


def mlm_config(vocab=None) -> ModelConfig:
    vocab = vocab or toy_vocab()
    return ModelConfig(vocab_size=vocab.size, max_position_embeddings=16, dropout=0.0)


def mlm_example(
    vocab,
    client_id: int,
    hours: list[float],
    cutoff_hours: float,
    colors: list[str] | None = None,
    sizes: list[int] | None = None,
) -> Example:
    """
    Событие это [EVT] плюс два предсказуемых поля.
    """

    count = len(hours)

    colors = colors or [COLORS[index % len(COLORS)] for index in range(count)]
    sizes = sizes or [index % 2 for index in range(count)]

    records = [
        encode_pairs(vocab, [("toy__color", color), ("toy__size", size)], EVT_ID)
        for color, size in zip(colors, sizes)
    ]

    widths = np.array([len(record.key_ids) for record in records], dtype=np.int64)

    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    np.cumsum(widths, out=offsets[1:])

    events = Events(
        key_ids=np.concatenate([record.key_ids for record in records]),
        value_ids=np.concatenate([record.value_ids for record in records]),
        positions=np.concatenate([record.positions for record in records]),
        offsets=offsets,
        event_type=np.array(["toy"] * len(records), dtype=object),
        ts=np.array([np.datetime64(BASE + timedelta(hours=h), "us") for h in hours]),
        seq=np.arange(len(records), dtype=np.int64),
    )

    return Example(
        client_id=client_id,
        cutoff=BASE + timedelta(hours=cutoff_hours),
        dataset="toy",
        client_group="train",
        seq_end=len(records),
        snapshot_ts=BASE - timedelta(hours=1),
        profile=encode_pairs(vocab, [("toy__flag", True)], USR_ID),
        events=events,
    )


def prepared(histories, cutoff_hours=48, max_events=10, vocab=None, rate=1.0, step=0):
    """
    Batch с масками, цели и вход модели.
    """

    vocab = vocab or toy_vocab()

    examples = [
        mlm_example(vocab, index, hours, cutoff_hours) for index, hours in enumerate(histories)
    ]

    masker = Masker(vocab, MaskingConfig(mode="token", seed=7, token_rate=rate))

    history = prepare_history_batch(
        collate(examples), metadata_from_examples(examples), max_events, masker=masker, step=step
    )

    table = FieldTable(vocab)

    return history, build_targets(history, table), table


def narrow_vocab() -> Vocab:
    """
    Предсказуемое поле с двумя значениями и предсказуемое с одним.
    """

    keys = [
        KeyEntry(6, "s__wide", "s", "wide", "categorical", True, "string", 8, 10),
        KeyEntry(7, "s__narrow", "s", "narrow", "categorical", True, "string", 10, 11),
    ]

    values = [
        ValueEntry(8, 6, "s__wide", "a", 3),
        ValueEntry(9, 6, "s__wide", "b", 2),
        ValueEntry(10, 7, "s__narrow", "only", 5),
    ]

    return Vocab(keys, values)


# ============================================================
# ТАБЛИЦА ПОЛЕЙ
# ============================================================


def test_trainable_fields_are_predictable_with_two_candidates(tok_run):
    tokenizer = Tokenizer.load(tok_run["vocab"], tok_run["artifacts"])

    table = FieldTable.load(tokenizer.vocab, tok_run["vocab"])

    for key_id in table.trainable_key_ids:
        entry = tokenizer.vocab.key_entry_by_id(key_id)
        assert entry.predictable
        assert entry.n_values >= 2

    for key_id in table.degenerate_key_ids:
        assert tokenizer.vocab.key_entry_by_id(key_id).n_values < 2

    predictable = sorted(entry.id for entry in tokenizer.vocab.keys if entry.predictable)

    assert sorted(table.trainable_key_ids + table.degenerate_key_ids) == predictable


def test_local_and_global_are_inverse(tok_run):
    tokenizer = Tokenizer.load(tok_run["vocab"], tok_run["artifacts"])

    table = FieldTable.load(tokenizer.vocab, tok_run["vocab"])

    keys = []
    values = []

    for key_id in table.trainable_key_ids:
        entry = tokenizer.vocab.key_entry_by_id(key_id)
        for value in range(entry.value_start, entry.value_end):
            keys.append(entry.id)
            values.append(value)

    keys = np.array(keys, dtype=np.int64)
    values = np.array(values, dtype=np.int64)

    local = table.to_local(keys, values)

    assert local.min() == 0
    assert np.array_equal(table.to_global(keys, local), values)


def test_special_value_is_not_a_target():
    table = FieldTable(toy_vocab())

    with pytest.raises(TargetError, match="special"):
        table.to_local(np.array([6]), np.array([MISSING_ID]))


def test_value_of_another_field_is_rejected():
    table = FieldTable(toy_vocab())

    # 11 это первое значение toy__size, а ключ передан toy__color.
    with pytest.raises(TargetError, match="не принадлежит полю"):
        table.to_local(np.array([6]), np.array([11]))


def test_non_predictable_field_is_rejected():
    table = FieldTable(toy_vocab())

    with pytest.raises(TargetError, match="predictable"):
        table.to_local(np.array([8]), np.array([13]))


def test_degenerate_field_has_no_head():
    table = FieldTable(narrow_vocab())

    assert table.trainable_key_ids == (6,)
    assert table.degenerate_key_ids == (7,)

    head = MLMHead(ModelConfig(vocab_size=11, max_position_embeddings=8), table)

    assert set(head.heads) == {"6"}


def test_broken_artifact_is_rejected(tok_run, tmp_path):
    tokenizer = Tokenizer.load(tok_run["vocab"], tok_run["artifacts"])

    directory = tmp_path / "vocab"

    shutil.copytree(tok_run["vocab"], directory)

    path = directory / FIELD_VALUE_IDS_FILE

    artifact = json.loads(path.read_text(encoding="utf-8"))

    name = next(iter(artifact["fields"]))

    artifact["fields"][name]["value_ids"] = artifact["fields"][name]["value_ids"][:-1]

    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(IncompatibleArtifactsError, match="кандидаты поля"):
        FieldTable.load(tokenizer.vocab, directory)


# ============================================================
# АДРЕСА ЦЕЛЕЙ
# ============================================================


def test_targets_point_at_the_right_token():
    history, targets, table = prepared([[0, 1, 2]])

    # Каждое событие это [EVT], color, size: маскируются позиции 1 и 2.
    assert targets.flat.tolist() == [1, 2, 4, 5, 7, 8]
    assert targets.event_row.tolist() == [0, 0, 1, 1, 2, 2]
    assert targets.col.tolist() == [1, 2, 1, 2, 1, 2]
    assert targets.example.tolist() == [0] * 6

    assert targets.key_ids.tolist() == [6, 7] * 3

    # Цели это исходные значения, а не [MASK].
    assert np.array_equal(targets.global_targets, table.to_global(targets.key_ids, targets.local_targets))


def test_targets_map_back_to_the_untruncated_batch():
    history, targets, _ = prepared([[0, 1, 2, 3, 4]], max_events=3)

    assert history.info.kept_events.tolist() == [2, 3, 4]

    # Токены событий 2, 3 и 4 в исходном batch это 6..14.
    assert targets.original_events.tolist() == [2, 2, 3, 3, 4, 4]
    assert targets.original_tokens.tolist() == [7, 8, 10, 11, 13, 14]


def test_two_examples_keep_their_own_addresses():
    history, targets, _ = prepared([[0, 1], [0, 1, 2]])

    assert targets.example.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    assert targets.event_row.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

    # Номера строк не убывают: иначе microbatch переставил бы выходы.
    assert (np.diff(targets.event_row) >= 0).all()


def test_degenerate_positions_are_dropped_and_counted():
    vocab = narrow_vocab()

    record = encode_pairs(vocab, [("s__wide", "a"), ("s__narrow", "only")], EVT_ID)

    events = Events(
        key_ids=record.key_ids,
        value_ids=record.value_ids,
        positions=record.positions,
        offsets=np.array([0, len(record.key_ids)], dtype=np.int64),
        event_type=np.array(["toy"], dtype=object),
        ts=np.array([np.datetime64(BASE, "us")]),
        seq=np.zeros(1, dtype=np.int64),
    )

    example = Example(
        client_id=0,
        cutoff=BASE + timedelta(hours=5),
        dataset="toy",
        client_group="train",
        seq_end=1,
        snapshot_ts=BASE - timedelta(hours=1),
        profile=encode_pairs(vocab, [], USR_ID),
        events=events,
    )

    masker = Masker(vocab, MaskingConfig(mode="token", seed=1, token_rate=1.0))

    history = prepare_history_batch(
        collate([example]), metadata_from_examples([example]), 4, masker=masker, step=0
    )

    targets = build_targets(history, FieldTable(vocab))

    assert targets.n_masked == 2
    assert targets.n_degenerate == 1
    assert targets.n == 1
    assert targets.key_ids.tolist() == [6]


def test_batch_without_masker_has_no_targets():
    vocab = toy_vocab()

    examples = [mlm_example(vocab, 0, [0, 1], 24)]

    history = prepare_history_batch(collate(examples), metadata_from_examples(examples), 5)

    with pytest.raises(TargetError, match="без masker"):
        build_targets(history, FieldTable(vocab))


def test_digest_depends_on_the_targets():
    left = prepared([[0, 1, 2]])[1]
    right = prepared([[0, 1, 2]])[1]
    other = prepared([[0, 1, 2, 3]])[1]

    assert left.digest() == right.digest()
    assert left.digest() != other.digest()


# ============================================================
# ТРИ ПРЕДСТАВЛЕНИЯ ИЗ ОДНОГО ПРОХОДА
# ============================================================


def test_local_hidden_matches_a_direct_forward():
    history, targets, _ = prepared([[0, 1, 2], [0, 5]])

    config = mlm_config()

    inputs = to_model_inputs(history, config)

    backbone = build_backbone(config, seed=5).eval()

    tensors = targets.tensors()

    with torch.no_grad():

        out = backbone(inputs, event_microbatch=None, gather=targets.gather())

        direct = backbone.pair.event(
            inputs.events.key_ids,
            inputs.events.value_ids,
            inputs.events.positions,
            inputs.events.padding_mask,
            return_hidden=True,
        )

    torch.testing.assert_close(
        out.local_hidden, direct.hidden[tensors["event_row"], tensors["col"]]
    )


def test_event_and_client_representations_are_indexed_consistently():
    history, targets, _ = prepared([[0, 1, 2], [0, 5]])

    config = mlm_config()

    inputs = to_model_inputs(history, config)

    backbone = build_backbone(config, seed=6).eval()

    tensors = targets.tensors()

    with torch.no_grad():
        out = backbone(inputs, gather=targets.gather())

    h_local, h_event, h_usr = representations(out, tensors)

    assert h_local.shape == h_event.shape == h_usr.shape == (targets.n, config.d_model)

    torch.testing.assert_close(h_event, out.event_embeddings[tensors["event_row"]])
    torch.testing.assert_close(h_usr, out.client_embedding[tensors["example"]])

    # Позиции одного события делят вектор события и вектор клиента.
    first, second = 0, 1

    assert targets.event_row[first] == targets.event_row[second]

    torch.testing.assert_close(h_event[first], h_event[second])
    assert not torch.allclose(h_local[first], h_local[second])


def test_microbatch_does_not_change_gathered_states():
    history, targets, _ = prepared([[0, 1, 2, 3, 4], [0, 1]])

    config = mlm_config()

    inputs = to_model_inputs(history, config)

    backbone = build_backbone(config, seed=8).eval()

    with torch.no_grad():
        whole = backbone(inputs, event_microbatch=None, gather=targets.gather())
        pieces = backbone(inputs, event_microbatch=2, gather=targets.gather())

    torch.testing.assert_close(whole.local_hidden, pieces.local_hidden)
    torch.testing.assert_close(whole.event_embeddings, pieces.event_embeddings)


def test_forward_without_gather_has_no_local_states():
    history, targets, _ = prepared([[0, 1, 2]])

    config = mlm_config()

    backbone = build_backbone(config, seed=9).eval()

    with torch.no_grad():
        out = backbone(to_model_inputs(history, config))

    assert out.local_hidden is None

    with pytest.raises(TargetError, match="без gather"):
        representations(out, targets.tensors())


def test_shuffled_rows_are_rejected():
    history, targets, _ = prepared([[0, 1, 2]])

    config = mlm_config()

    backbone = build_backbone(config, seed=10).eval()

    rows, cols = targets.gather()

    with pytest.raises(BatchError, match="неубывающих"):
        backbone(to_model_inputs(history, config), gather=(rows.flip(0), cols.flip(0)))


def test_gather_beyond_the_record_is_rejected():
    history, targets, _ = prepared([[0, 1]])

    config = mlm_config()

    backbone = build_backbone(config, seed=11).eval()

    rows, cols = targets.gather()

    with pytest.raises(BatchError, match="padding"):
        backbone(to_model_inputs(history, config), gather=(rows, torch.full_like(cols, 5)))


# ============================================================
# ГОЛОВА
# ============================================================


def head_and_inputs(seed: int = 3, histories=None, dropout: float = 0.0):
    histories = histories or [[0, 1, 2], [0, 5]]

    history, targets, table = prepared(histories)

    config = mlm_config()

    inputs = to_model_inputs(history, config)

    backbone = build_backbone(config, seed=seed)
    head = MLMHead(config, table)

    return backbone, head, inputs, targets, table, config


def test_logits_cover_exactly_the_field_candidates():
    backbone, head, inputs, targets, table, _ = head_and_inputs()

    backbone.eval()
    head.eval()

    tensors = targets.tensors()

    with torch.no_grad():
        out = backbone(inputs, gather=targets.gather())
        field_logits = head(*representations(out, tensors), tensors["key_ids"])

    assert [item.key_id for item in field_logits] == [6, 7]

    for item in field_logits:

        assert item.n_candidates == table.size_of(item.key_id)

        keys = np.full(item.n_targets, item.key_id, dtype=np.int64)

        chosen = table.to_global(keys, item.logits.argmax(dim=-1).numpy())

        entry = table.vocab.key_entry_by_id(item.key_id)

        assert (chosen >= entry.value_start).all()
        assert (chosen < entry.value_end).all()


def test_positions_are_grouped_by_field_without_loss():
    backbone, head, inputs, targets, _, _ = head_and_inputs()

    backbone.eval()
    head.eval()

    tensors = targets.tensors()

    with torch.no_grad():
        out = backbone(inputs, gather=targets.gather())
        field_logits = head(*representations(out, tensors), tensors["key_ids"])

    covered = torch.cat([item.index for item in field_logits]).sort().values

    assert covered.tolist() == list(range(targets.n))

    for item in field_logits:
        assert bool((tensors["key_ids"][item.index] == item.key_id).all())


def test_head_creates_no_parameters_inside_forward():
    backbone, head, inputs, targets, _, _ = head_and_inputs()

    before = head.n_parameters()

    tensors = targets.tensors()

    with torch.no_grad():
        out = backbone(inputs, gather=targets.gather())
        head(*representations(out, tensors), tensors["key_ids"])

    assert head.n_parameters() == before


def test_unknown_field_has_no_head():
    _, head, _, _, _, config = head_and_inputs()

    zeros = torch.zeros(1, config.d_model)

    with pytest.raises(TargetError, match="нет головы"):
        head(zeros, zeros, zeros, torch.tensor([99], dtype=torch.long))


def test_empty_batch_gives_no_logits():
    _, head, _, _, _, config = head_and_inputs()

    empty = torch.zeros(0, config.d_model)

    assert head(empty, empty, empty, torch.zeros(0, dtype=torch.long)) == []


def mask_one_field(history, key_id: int):
    """
    Маскирует только одно поле: соседнее остаётся видимым.
    """

    from dataclasses import replace

    from src.tokenizer.config import MASK_ID
    from src.tokenizer.masking import IGNORE_INDEX

    values = np.asarray(history.tokens.value_ids).copy()

    chosen = np.flatnonzero(np.asarray(history.tokens.key_ids) == key_id)

    targets = np.full(values.size, IGNORE_INDEX, dtype=np.int64)
    mask = np.zeros(values.size, dtype=bool)

    targets[chosen] = values[chosen]
    values[chosen] = MASK_ID
    mask[chosen] = True

    return replace(
        history,
        tokens=replace(history.tokens, value_ids=values),
        targets=targets,
        mask=mask,
    )


def test_visible_neighbour_changes_the_logits():
    """
    Голова читает содержимое события, а не только имя поля.

    Маскируется только размер; цвет остаётся видимым, и его
    замена обязана менять предсказание размера.
    """

    config = mlm_config()

    vocab = toy_vocab()

    table = FieldTable(vocab)

    backbone = build_backbone(config, seed=12).eval()
    head = MLMHead(config, table).eval()

    def logits_for(colors):

        examples = [mlm_example(vocab, 0, [0, 1, 2], 24, colors=colors)]

        history = mask_one_field(
            prepare_history_batch(collate(examples), metadata_from_examples(examples), 10),
            key_id=7,
        )

        targets = build_targets(history, table)

        tensors = targets.tensors()

        with torch.no_grad():
            out = backbone(to_model_inputs(history, config), gather=targets.gather())
            return head(*representations(out, tensors), tensors["key_ids"])

    left = logits_for(["red", "red", "red"])
    right = logits_for(["blue", "blue", "blue"])

    assert [item.key_id for item in left] == [7]

    assert not torch.allclose(left[0].logits, right[0].logits)


def test_fully_masked_event_hides_the_neighbour():
    """
    Обратная сторона того же: когда замаскировано всё, разное
    содержимое неотличимо. Это свойство маскирования, и тест
    держит его явным, чтобы предыдущая проверка не выглядела
    случайной.
    """

    config = mlm_config()

    vocab = toy_vocab()

    table = FieldTable(vocab)

    backbone = build_backbone(config, seed=12).eval()
    head = MLMHead(config, table).eval()

    def logits_for(colors):

        examples = [mlm_example(vocab, 0, [0, 1, 2], 24, colors=colors)]

        masker = Masker(vocab, MaskingConfig(mode="token", seed=7, token_rate=1.0))

        history = prepare_history_batch(
            collate(examples), metadata_from_examples(examples), 10, masker=masker, step=0
        )

        targets = build_targets(history, table)

        tensors = targets.tensors()

        with torch.no_grad():
            out = backbone(to_model_inputs(history, config), gather=targets.gather())
            return head(*representations(out, tensors), tensors["key_ids"])

    left = logits_for(["red", "red", "red"])
    right = logits_for(["blue", "blue", "blue"])

    torch.testing.assert_close(left[0].logits, right[0].logits)


# ============================================================
# ГРАДИЕНТЫ
# ============================================================


def test_gradients_reach_the_head_and_the_whole_backbone():
    backbone, head, inputs, targets, _, _ = head_and_inputs(seed=13)

    backbone.train()
    head.train()

    tensors = targets.tensors()

    out = backbone(inputs, gather=targets.gather())

    field_logits = head(*representations(out, tensors), tensors["key_ids"])

    result = mlm_loss(field_logits, tensors["local_targets"])

    result.field_balanced.backward()

    checked = {
        "fuse": head.fuse[0].weight,
        "head_color": head.heads["6"].weight,
        "head_size": head.heads["7"].weight,
        "embeddings": backbone.pair.embeddings.token.weight,
        "event": backbone.pair.event.layers[0].linear1.weight,
        "profile": backbone.pair.profile.layers[0].linear1.weight,
        "history": backbone.history.layers[0].linear1.weight,
        "time": backbone.time.linear.weight,
    }

    for name, parameter in checked.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name

    from src.tokenizer.config import PAD_ID

    assert (backbone.pair.embeddings.token.weight.grad[PAD_ID] == 0).all()


# ============================================================
# РЕАЛЬНЫЕ ДАННЫЕ
# ============================================================


def test_real_batch_produces_valid_targets(tok_run):
    tokenizer = Tokenizer.load(tok_run["vocab"], tok_run["artifacts"])

    table = FieldTable.load(tokenizer.vocab, tok_run["vocab"])

    config = config_from_tokenizer(tokenizer)

    data = TokenizedDataset(tok_run["tokenized"], "train", vocab_dir=tok_run["vocab"])

    examples = [data.load(index) for index in range(4)]

    masker = Masker(tokenizer.vocab, MaskingConfig(mode="field_balanced", seed=3))

    history = prepare_history_batch(
        collate(examples), metadata_from_examples(examples), 64, masker=masker, step=0
    )

    targets = build_targets(history, table)

    assert targets.n > 0
    assert targets.n_fields > 1

    # Каждая цель это настоящее значение своего поля.
    table.check_targets(targets.key_ids, targets.global_targets)

    assert (targets.local_targets >= 0).all()
    assert (targets.local_targets < table.n_candidates[targets.key_ids]).all()

    backbone = build_backbone(config, seed=4).eval()
    head = MLMHead(config, table).eval()

    tensors = targets.tensors()

    with torch.no_grad():
        out = backbone(to_model_inputs(history, config), gather=targets.gather())
        field_logits = head(*representations(out, tensors), tensors["key_ids"])

    assert sum(item.n_targets for item in field_logits) == targets.n

    for item in field_logits:
        assert item.n_candidates == table.size_of(item.key_id)
        assert torch.isfinite(item.logits).all()
