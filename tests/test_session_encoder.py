"""
Session Encoder: группировка, forward, MLM и градиенты.

Четыре проверки, каждая покрывает свою часть целиком. Всё, что
касается причинности, проверяется на результате ДО наблюдательного
шума: шум обнуляет часть статусов и городов и скрыл бы нарушение.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta

import numpy as np
import pytest
import torch

from src.model.backbone import build_backbone
from src.model.batching import BatchError
from src.model.checkpoint import compare_artifacts, compare_model_config
from src.model.config import (
    EVENT_ARCHITECTURE,
    SESSION_ARCHITECTURE,
    STRUCTURE_EVENT,
    STRUCTURE_SESSION,
    ModelConfig,
)
from src.model.data import ClientStore, SessionExample, session_keys_from_examples
from src.model.history_batching import (
    metadata_from_examples,
    prepare_history_batch,
    to_model_inputs,
)
from src.model.losses import mlm_loss
from src.model.mlm_batching import build_targets, representations, within_session
from src.model.mlm_head import FieldTable, MLMHead, TargetError
from src.model.session_batching import group_events
from src.model.sessions import SessionSidecarError, build_sidecar, load_session_keys
from src.tokenizer.config import EVT_ID, MASK_ID, USR_ID
from src.tokenizer.dataset import Events, collate
from src.tokenizer.encode import encode_pairs
from src.tokenizer.masking import Masker, MaskingConfig

from tests.test_tok_encode import toy_vocab


BASE = datetime(2025, 1, 1)

COLORS = ("red", "blue")
SIZES = (1, 2)


# ============================================================
# СИНТЕТИЧЕСКАЯ ЛЕНТА
# ============================================================
#
# Один клиент, из ленты которого нарезаются примеры с разными
# cutoff. Сессия k1 растянута во времени, и между её экранами
# лежат чужие события: так проверяется и порядок элементов,
# и то, что префикс не тянет за собой будущее.
# ============================================================

LEDGER: tuple[tuple[int, str, int], ...] = (
    (-3600, "communication", -1),
    (0, "app_operation", -1),
    (10, "app_operation", -1),
    (14, "app_screen", 1),
    (30, "app_screen", 1),
    (31, "banner", -1),
    (50, "transaction", -1),
    (70, "app_screen", 1),
    (95, "app_operation", -1),
    (7200, "product_event", -1),
    (20000, "app_screen", 2),
    (30000, "app_screen", 3),
    (30060, "app_screen", 3),
)


def ledger(vocab, client_id: int, spec, cutoff_s: float) -> SessionExample:
    """
    Пример с событиями в заданные секунды и ключами сессий.
    """

    records = [
        encode_pairs(
            vocab,
            [("toy__color", COLORS[index % 2]), ("toy__size", SIZES[index % 2])],
            EVT_ID,
        )
        for index in range(len(spec))
    ]

    widths = np.array([len(record.key_ids) for record in records], dtype=np.int64)

    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    np.cumsum(widths, out=offsets[1:])

    events = Events(
        key_ids=np.concatenate([record.key_ids for record in records]),
        value_ids=np.concatenate([record.value_ids for record in records]),
        positions=np.concatenate([record.positions for record in records]),
        offsets=offsets,
        event_type=np.array([kind for _, kind, _ in spec], dtype=object),
        ts=np.array(
            [np.datetime64(BASE + timedelta(seconds=t), "us") for t, _, _ in spec]
        ),
        seq=np.arange(len(spec), dtype=np.int64),
    )

    return SessionExample(
        client_id=client_id,
        cutoff=BASE + timedelta(seconds=cutoff_s),
        dataset="toy",
        client_group="train",
        seq_end=len(spec),
        snapshot_ts=BASE - timedelta(hours=2),
        profile=encode_pairs(vocab, [("toy__flag", True)], USR_ID),
        events=events,
        session_keys=np.array([key for _, _, key in spec], dtype=np.int64),
    )


def prefix(spec, cutoff_s: float):
    return [row for row in spec if row[0] < cutoff_s]


def examples(vocab=None):
    """
    Пять примеров: полный, два обрезанных cutoff, без сессий
    и с единственным экраном.
    """

    vocab = vocab or toy_vocab()

    return {
        "ex0": ledger(vocab, 1, LEDGER, 40000),
        "ex1": ledger(vocab, 1, prefix(LEDGER, 32), 32),
        "ex2": ledger(vocab, 1, prefix(LEDGER, 12), 12),
        "ex3": ledger(vocab, 2, ((0, "transaction", -1), (100, "transaction", -1)), 40000),
        "ex4": ledger(vocab, 3, ((0, "app_screen", 5),), 40000),
    }


def model_config(structure: str, vocab=None) -> ModelConfig:

    vocab = vocab or toy_vocab()

    settings = SESSION_ARCHITECTURE if structure == STRUCTURE_SESSION else EVENT_ARCHITECTURE

    return ModelConfig(vocab_size=vocab.size, max_position_embeddings=16, **{**settings, "dropout": 0.0})


def prepared(items, structure: str, max_events=None, masker=None, step: int = 0, vocab=None):
    """
    История, вход модели и цели одного batch.
    """

    vocab = vocab or toy_vocab()

    keys = session_keys_from_examples(items)

    history = prepare_history_batch(
        collate(items),
        metadata_from_examples(items),
        max_events,
        masker=masker,
        step=step,
        session_keys=keys if structure == STRUCTURE_SESSION else None,
        structure=structure,
    )

    config = model_config(structure, vocab)

    targets = None

    if masker is not None:
        targets = build_targets(history, FieldTable(vocab))

    return history, to_model_inputs(history, config), targets, config


def seconds(value) -> int:
    return int((np.datetime64(value) - np.datetime64(BASE, "us")) / np.timedelta64(1, "s"))


# ============================================================
# 1. ГРУППИРОВКА, ПОРЯДОК И CUTOFF
# ============================================================


def test_grouping_respects_keys_order_and_cutoff(tok_run, tmp_path):
    """
    Сессия это экраны с одним ключом внутри примера; всё
    остальное отдельно. Ничего не потеряно, ничего не учтено
    дважды, будущее в прошлое не заглядывает.
    """

    items = examples()

    history, inputs, _, _ = prepared(
        [items["ex0"], items["ex1"], items["ex2"]], STRUCTURE_SESSION
    )

    layout = history.layout

    types = history.tokens.event_type
    ts = history.tokens.ts

    # --- состав сессии -----------------------------------
    first = layout.session_of_event[: len(LEDGER)]

    members = np.flatnonzero(first >= 0)

    assert [seconds(ts[row]) for row in members if first[row] == first[3]] == [14, 30, 70]
    assert [int(layout.position_in_session[row]) for row in members if first[row] == first[3]] == [0, 1, 2]

    # Операции, баннер и транзакция остаются отдельными.
    for row in range(len(LEDGER)):
        if types[row] != "app_screen":
            assert layout.session_of_event[row] < 0

    # --- ничего не потеряно ------------------------------
    grouped = np.flatnonzero(layout.session_of_event >= 0)

    assert layout.standalone_rows.size + grouped.size == layout.n_events

    valid = layout.member_rows[layout.member_rows >= 0]

    assert np.array_equal(np.sort(valid), grouped)
    assert np.unique(valid).size == valid.size

    # --- порядок и слоты ---------------------------------
    # Сессия стоит там, где она закончилась: транзакция в 50 с
    # раньше сессии, закончившейся в 70 с.
    session = int(first[3])

    assert int(layout.slot_of_event[6]) < int(layout.session_slot[session])
    assert seconds(layout.session_end[session]) == 70

    for row in np.flatnonzero(layout.session_of_event >= 0):
        assert layout.slot_of_event[row] == layout.session_slot[layout.session_of_event[row]]

    per_example = np.bincount(
        history.tokens.example_of_event[layout.standalone_rows], minlength=3
    ) + np.bincount(layout.session_example, minlength=3)

    assert np.array_equal(per_example, layout.used_history_length)
    assert int(layout.slot_of_event.min()) >= 1

    # --- cutoff ------------------------------------------
    # ex1 обрезан 32 секундами: в сессии ровно доступный префикс,
    # а её начало то же, что в полном примере.
    second = layout.session_of_event[len(LEDGER) : len(LEDGER) + len(prefix(LEDGER, 32))]

    partial = int(second[second >= 0][0])

    assert int(layout.session_length[partial]) == 2
    assert seconds(layout.session_end[partial]) == 30
    assert seconds(layout.session_start[partial]) == seconds(layout.session_start[session])

    # ex2 обрезан 12 секундами: экранов ещё нет, сессий тоже.
    assert int((layout.session_example == 2).sum()) == 0

    # --- время -------------------------------------------
    assert np.allclose(
        layout.member_gap_minutes[session][:3], [0.0, 16 / 60, 40 / 60], atol=1e-9
    )

    # gap сессии считается от предыдущего ЭЛЕМЕНТА истории.
    assert np.isclose(layout.session_hours[session, 0], 20 / 3600, atol=1e-9)
    assert np.isclose(layout.session_hours[session, 1], (40000 - 70) / 3600, atol=1e-6)

    # --- обрезка -----------------------------------------
    short, short_inputs, _, _ = prepared([items["ex0"]], STRUCTURE_SESSION, max_events=4)

    assert short.tokens.n_events == 4
    assert short.layout.n_events == 4
    assert int(short_inputs.slot_of_event.min()) >= 1

    # Первый уцелевший элемент помнит, что до него что-то было.
    assert float(short.layout.standalone_hours[0, 0]) > 0.0

    # --- sidecar на реальных данных ----------------------
    root = tmp_path / "tokenized"
    shutil.copytree(tok_run["tokenized"], root)

    build_sidecar(tok_run["processed"], root)

    store = ClientStore(root, "train", tok_run["vocab"], max_clients=3, sessions=True)

    import pyarrow.parquet as pq

    from src.model.sessions import SESSION_ID_COLUMNS, available_columns, is_named

    source = tok_run["processed"] / "clients" / "train_clients" / "events.parquet"

    columns = available_columns(source)

    table = pq.read_table(
        source,
        columns=["client_id", "seq", "event_type", *columns.values()],
    ).to_pandas()

    for client_id in store.client_ids:

        keys = store.session_keys[client_id]
        block = table[table.client_id == client_id]

        # Ключ есть ровно там, где у события есть свой session_id.
        # Правило не про тип события, а про метаданную: у RAW
        # ревизии 1 её несут только экраны, у ревизии 2 ещё
        # операции и баннеры своей сессии.
        def own_id(index: int, kind: str):
            if kind not in columns:
                return None
            return block[SESSION_ID_COLUMNS[kind]].to_numpy()[index]

        kinds = block.event_type.to_numpy()

        own = np.array(
            [is_named(own_id(index, kind)) for index, kind in enumerate(kinds)],
            dtype=bool,
        )

        assert np.array_equal(keys >= 0, own)

        # Ключа не бывает у события не из приложения.
        assert not (keys >= 0)[~np.isin(kinds, tuple(columns))].any()

        # Разные session_id не слились в один ключ.
        distinct = {
            own_id(index, kind)
            for index, kind in enumerate(kinds)
            if is_named(own_id(index, kind))
        }

        assert len(distinct) == np.unique(keys[keys >= 0]).size

        # Поэлементная сверка seq, а не совпадение длин.
        assert np.array_equal(
            store.events[client_id].seq, np.arange(keys.size, dtype=np.int64)
        )

    # Подделанный отпечаток и чужой набор отвергаются.
    manifest = root / "sessions" / "sessions_manifest.json"

    saved = json.loads(manifest.read_text(encoding="utf-8"))

    spoiled = {**saved}
    spoiled["groups"] = {**saved["groups"], "train": {**saved["groups"]["train"], "fingerprint": "0" * 64}}
    manifest.write_text(json.dumps(spoiled, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(SessionSidecarError):
        load_session_keys(root, "train")

    alien = {**saved, "tokenized_manifest_sha256": "0" * 64}
    manifest.write_text(json.dumps(alien, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(SessionSidecarError):
        load_session_keys(root, "train")


# ============================================================
# 2. FORWARD ОБЕИХ СТРУКТУР
# ============================================================


def test_forward_shapes_pooling_and_padding_invariance_in_both_modes():
    """
    Формы, pooling по [SES] и [USR], безразличие к padding
    и то, что режим event остался прежним путём.
    """

    items = examples()

    batch = [items["ex0"], items["ex4"], items["ex3"]]

    history, inputs, _, config = prepared(batch, STRUCTURE_SESSION)

    backbone = build_backbone(config, seed=1).eval()

    with torch.no_grad():
        out = backbone(inputs)

    sessions = inputs.sessions

    d = config.d_model

    assert tuple(out.client_embedding.shape) == (3, d)
    assert tuple(out.contextualized.shape) == (3, inputs.max_length, d)
    assert tuple(out.session_hidden.shape) == (sessions.n_sessions, sessions.max_session_length + 1, d)

    # Padding маска истории и нулевые padded-строки.
    expected = torch.arange(inputs.max_length).unsqueeze(0) >= (
        inputs.used_history_length + 1
    ).unsqueeze(1)

    assert torch.equal(out.padding_mask, expected)
    assert float(out.contextualized[out.padding_mask].abs().sum()) == 0.0

    # Адреса: вектор несущего элемента и вектор сессии.
    assert torch.allclose(
        out.event_embeddings,
        out.contextualized[inputs.example_of_event, inputs.slot_of_event],
    )
    assert torch.allclose(
        out.session_embeddings,
        out.contextualized[sessions.session_example, sessions.session_slot],
    )
    assert torch.allclose(out.session_pooled, out.session_hidden[:, 0])

    # Padding внутри сессии нулевой.
    rows, cols = torch.nonzero(~sessions.member_valid, as_tuple=True)

    if rows.numel():
        assert float(out.session_hidden[rows, cols + 1].abs().sum()) == 0.0

    # [SES] не получает временного слагаемого.
    with torch.no_grad():
        vectors = backbone.pair.encode_events(inputs.events, None)
        x, mask = backbone.session.assemble(vectors, sessions)

    from src.model.time_encoding import sinusoidal_positions

    lead = sinusoidal_positions(x.shape[1], d, dtype=x.dtype)[0]

    assert torch.allclose(x[:, 0], backbone.session.ses + lead, atol=1e-6)
    assert not bool(mask[:, 0].any())

    # Безразличие к padding: сессия из одного экрана одна и в batch.
    _, alone, _, _ = prepared([items["ex4"]], STRUCTURE_SESSION)

    with torch.no_grad():
        single = backbone(alone)

    index = int(torch.nonzero(sessions.session_example == 1)[0])

    assert alone.sessions.max_session_length < sessions.max_session_length
    torch.testing.assert_close(
        out.session_pooled[index], single.session_pooled[0], rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        out.client_embedding[1], single.client_embedding[0], rtol=1e-5, atol=1e-5
    )

    # Позиций у History Encoder меньше, чем событий.
    assert inputs.max_length < 1 + inputs.n_events

    # --- прежняя структура --------------------------------
    _, event_inputs, _, event_config = prepared(batch, STRUCTURE_EVENT)

    event_backbone = build_backbone(event_config, seed=1).eval()

    assert event_backbone.session is None
    assert event_inputs.sessions is None
    assert event_inputs.standalone_rows is None
    assert event_inputs.max_length == 1 + int(event_inputs.used_history_length.max())

    # На batch без сессий обе структуры дают одно и то же:
    # общие веса строятся до Session Encoder.
    _, plain_event, _, _ = prepared([items["ex3"]], STRUCTURE_EVENT)
    _, plain_session, _, _ = prepared([items["ex3"]], STRUCTURE_SESSION)

    assert plain_session.sessions.n_sessions == 0

    with torch.no_grad():
        left = event_backbone(plain_event)
        right = backbone(plain_session)

    torch.testing.assert_close(left.client_embedding, right.client_embedding)

    # Структура модели и структура входа обязаны совпадать.
    with pytest.raises(BatchError):
        with torch.no_grad():
            event_backbone(plain_session)

    # Fast path отключён только там, где длина велика.
    assert all(not layer._forward_hooks for layer in backbone.session.layers)
    assert all(layer._forward_hooks for layer in backbone.history.layers)


# ============================================================
# 3. MLM ЧЕРЕЗ СЕССИИ
# ============================================================


def test_mlm_masks_precede_encoders_and_targets_map_through_sessions():
    """
    Маски ставятся до энкодеров, адреса целей переживают
    группировку, кандидаты ограничены полем.
    """

    vocab = toy_vocab()

    items = examples(vocab)

    batch = [items["ex0"], items["ex4"]]

    masker = Masker(vocab, MaskingConfig(mode="token", seed=7, token_rate=1.0))

    history, inputs, targets, config = prepared(
        batch, STRUCTURE_SESSION, masker=masker, vocab=vocab
    )

    assert targets.n > 0

    # Маска стоит в самом входе Event Encoder.
    rows = torch.from_numpy(targets.event_row)
    cols = torch.from_numpy(targets.col)

    assert bool((inputs.events.value_ids[rows, cols] == MASK_ID).all())

    # Целей во входе модели нет.
    assert not hasattr(inputs, "targets")

    # Раскладка не зависит от того, что замаскировано.
    clean, _, _, _ = prepared(batch, STRUCTURE_SESSION, vocab=vocab)

    assert np.array_equal(clean.layout.session_of_event, history.layout.session_of_event)
    assert np.array_equal(clean.layout.slot_of_event, history.layout.slot_of_event)
    assert np.array_equal(clean.layout.member_rows, history.layout.member_rows)

    backbone = build_backbone(config, seed=3).eval()
    table = FieldTable(vocab)
    head = MLMHead(config, table).eval()

    tensors = targets.tensors("cpu")

    calls = {"n": 0}
    original = backbone.pair.encode_events

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    backbone.pair.encode_events = counted

    with torch.no_grad():
        out = backbone(inputs, gather=targets.gather("cpu"))

    assert calls["n"] == 1

    h_local, h_event, h_usr = representations(out, tensors)

    within = within_session(out, tensors)

    assert within is not None

    index, h_within = within

    owner = out.session_of_event[rows]

    assert torch.equal(index, torch.nonzero(owner >= 0, as_tuple=True)[0])

    position = out.position_in_session[rows[index]]

    assert torch.allclose(h_within, out.session_hidden[owner[index], position + 1])

    # Вектор несущего элемента для обоих видов целей.
    assert torch.allclose(
        h_event, out.contextualized[tensors["example"], inputs.slot_of_event[rows]]
    )

    # Локальное состояние из того же прохода Event Encoder.
    with torch.no_grad():
        direct = backbone.pair.event.forward(
            inputs.events.key_ids,
            inputs.events.value_ids,
            inputs.events.positions,
            inputs.events.padding_mask,
            return_hidden=True,
        )

    torch.testing.assert_close(h_local, direct.hidden[rows, cols])

    with torch.no_grad():
        logits = head(h_local, h_event, h_usr, tensors["key_ids"], within)

    for item in logits:
        assert item.n_candidates == table.size_of(item.key_id)
        assert int(tensors["local_targets"][item.index].max()) < item.n_candidates

    loss = mlm_loss(logits, tensors["local_targets"])

    assert torch.isfinite(loss.field_balanced)

    # Вторая проекция влияет только на сгруппированные цели.
    grouped_before = logits[0].logits.clone()

    with torch.no_grad():
        for parameter in head.fuse_session.parameters():
            parameter.zero_()

        after = head(h_local, h_event, h_usr, tensors["key_ids"], within)

    changed = ~torch.isclose(after[0].logits, grouped_before).all(dim=-1)

    inside = torch.isin(logits[0].index, index)

    assert torch.equal(changed, inside)

    # Адреса переживают обрезку.
    trimmed, trimmed_inputs, trimmed_targets, _ = prepared(
        batch, STRUCTURE_SESSION, max_events=6, masker=masker, vocab=vocab
    )

    full = collate(batch)

    assert np.array_equal(
        np.asarray(full.value_ids)[trimmed_targets.original_tokens],
        trimmed_targets.global_targets,
    )

    # Маскирование по ключу достаёт цели и в сессии, и вне её.
    key_masker = Masker(vocab, MaskingConfig(mode="key", seed=5, keys_per_example=1))

    _, key_inputs, key_targets, _ = prepared(
        batch, STRUCTURE_SESSION, masker=key_masker, vocab=vocab
    )

    owners = history.layout.session_of_event[key_targets.event_row]

    assert bool((owners >= 0).any())
    assert bool((owners < 0).any())

    # Прежняя структура не принимает состояния сессий.
    event_config = model_config(STRUCTURE_EVENT, vocab)
    event_head = MLMHead(event_config, table)

    assert event_head.fuse_session is None

    with pytest.raises(TargetError):
        event_head(h_local, h_event, h_usr, tensors["key_ids"], within)

    # Совместимость конфигураций: старый checkpoint без поля.
    legacy = model_config(STRUCTURE_EVENT, vocab).as_dict()
    del legacy["structure"]

    compare_model_config(legacy, model_config(STRUCTURE_EVENT, vocab).as_dict())

    from src.tokenizer.config import IncompatibleArtifactsError

    with pytest.raises(IncompatibleArtifactsError):
        compare_model_config(legacy, config.as_dict())

    # То же правило для отпечатков артефактов: ключа sessions в
    # старом checkpoint нет, и это значит "sidecar не собирали",
    # а не "любой sidecar подойдёт".
    compare_artifacts({"vocab": "a"}, {"vocab": "a", "sessions": None})

    with pytest.raises(IncompatibleArtifactsError):
        compare_artifacts({"vocab": "a"}, {"vocab": "a", "sessions": "beef"})

    with pytest.raises(IncompatibleArtifactsError):
        compare_artifacts({"vocab": "a", "sessions": "beef"}, {"vocab": "a", "sessions": "cafe"})

    with pytest.raises(IncompatibleArtifactsError):
        compare_artifacts({"vocab": "a", "unigram": "b"}, {"vocab": "a"})


# ============================================================
# 4. BACKWARD ЧЕРЕЗ ВСЕ ЭНКОДЕРЫ
# ============================================================


def gradient_names(backbone, head) -> dict:

    return {
        "embeddings.token": backbone.pair.embeddings.token.weight,
        "embeddings.position": backbone.pair.embeddings.position.weight,
        "event": backbone.pair.event.layers[0].linear1.weight,
        "profile": backbone.pair.profile.layers[0].linear1.weight,
        "session.ses": backbone.session.ses,
        "session.layer": backbone.session.layers[0].linear1.weight,
        "session.time": backbone.session.time.weight,
        "history": backbone.history.layers[0].linear1.weight,
        "history.time": backbone.time.linear.weight,
        "fuse": head.fuse[0].weight,
        "fuse_session": head.fuse_session[0].weight,
    }


def test_backward_gives_finite_gradients_through_all_encoders_and_both_projections():
    """
    Loss и градиенты конечны, и они доходят до каждого энкодера
    и до обеих входных проекций головы.
    """

    vocab = toy_vocab()

    items = examples(vocab)

    batch = [items["ex0"], items["ex4"]]

    masker = Masker(vocab, MaskingConfig(mode="token", seed=11, token_rate=1.0))

    _, inputs, targets, config = prepared(
        batch, STRUCTURE_SESSION, masker=masker, vocab=vocab
    )

    backbone = build_backbone(config, seed=5)
    table = FieldTable(vocab)
    head = MLMHead(config, table)

    backbone.train()
    head.train()

    tensors = targets.tensors("cpu")

    out = backbone(inputs, gather=targets.gather("cpu"))

    logits = head(*representations(out, tensors), tensors["key_ids"], within_session(out, tensors))

    loss = mlm_loss(logits, tensors["local_targets"])

    assert torch.isfinite(loss.field_balanced)

    loss.field_balanced.backward()

    for name, parameter in gradient_names(backbone, head).items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert float(parameter.grad.abs().sum()) > 0.0, name

    for key_id in head.key_ids:
        head_weight = head.heads[str(key_id)].weight
        assert head_weight.grad is not None
        assert torch.isfinite(head_weight.grad).all()

    # Строка [PAD] не учится.
    assert float(backbone.pair.embeddings.token.weight.grad[config.pad_id].abs().sum()) == 0.0

    # Batch без сессий не трогает вторую проекцию.
    plain_backbone = build_backbone(config, seed=5)
    plain_head = MLMHead(config, table)

    plain_backbone.train()
    plain_head.train()

    _, plain_inputs, plain_targets, _ = prepared(
        [items["ex3"]], STRUCTURE_SESSION, masker=masker, vocab=vocab
    )

    plain_tensors = plain_targets.tensors("cpu")

    plain_out = plain_backbone(plain_inputs, gather=plain_targets.gather("cpu"))

    assert within_session(plain_out, plain_tensors) is None

    plain_logits = plain_head(
        *representations(plain_out, plain_tensors), plain_tensors["key_ids"]
    )

    mlm_loss(plain_logits, plain_tensors["local_targets"]).field_balanced.backward()

    assert plain_head.fuse_session[0].weight.grad is None
    assert plain_backbone.session.ses.grad is None

    # Прежняя структура: backward идёт, второй проекции нет.
    event_config = model_config(STRUCTURE_EVENT, vocab)

    _, event_inputs, event_targets, _ = prepared(
        batch, STRUCTURE_EVENT, masker=masker, vocab=vocab
    )

    event_backbone = build_backbone(event_config, seed=5)
    event_head = MLMHead(event_config, table)

    event_backbone.train()
    event_head.train()

    event_tensors = event_targets.tensors("cpu")

    event_out = event_backbone(event_inputs, gather=event_targets.gather("cpu"))

    event_logits = event_head(
        *representations(event_out, event_tensors), event_tensors["key_ids"]
    )

    event_loss = mlm_loss(event_logits, event_tensors["local_targets"])

    assert torch.isfinite(event_loss.field_balanced)

    event_loss.field_balanced.backward()

    assert event_head.fuse_session is None
    assert torch.isfinite(event_backbone.history.layers[0].linear1.weight.grad).all()
