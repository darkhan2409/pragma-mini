"""
Downstream: product_open_90d.

Проверяется не качество (на фикстуре его нет и быть не может),
а устройство: один пример на клиента на правильном срезе, метка
рядом с примером и вне признаков, признаки строго из прошлого,
преобразования обучены только на train.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model.checkpoint import save_checkpoint
from src.model.downstream import (
    CATEGORICAL_SLICE,
    FEATURE_NAMES,
    GROUPS,
    LABEL,
    DownstreamError,
    build_embeddings,
    build_examples,
    build_features,
    encode,
    fit_encoder,
    render_downstream,
    run_downstream,
)
from src.model.trainer import Trainer

from tests.test_trainer import env, small_config  # noqa: F401


@pytest.fixture(scope="module")
def checkpoint(env, tmp_path_factory):
    """
    Крошечная необученная модель: downstream читает веса, а не
    учит их.
    """

    path = tmp_path_factory.mktemp("downstream") / "best.pt"

    trainer = Trainer(small_config(), env.tokenizer, env.table, env.unigram, "cpu")

    save_checkpoint(
        path,
        backbone=trainer.backbone,
        head=trainer.head,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        counters=trainer.counters(),
        train_config=trainer.config.as_dict(),
        model_config=trainer.model_config.as_dict(),
        masking_config=trainer.config.masking().as_dict(),
        sampler_state=None,
        splits={},
        artifacts=env.hashes,
        metrics=None,
    )

    return path


def test_downstream_examples_features_and_models_are_leak_free_and_fit_on_train_only(
    env, prep_run, prep_raw_dir, checkpoint, tmp_path
):

    processed = prep_run["processed"]

    examples = build_examples(processed, prep_raw_dir)

    # --- срез один и он же начало окна метки ---------------
    import pyarrow.parquet as pq

    from src.preprocessing.raw import read_manifest

    manifest = read_manifest(prep_raw_dir)

    assert examples.cutoff == np.datetime64(manifest.feature_end, "us")

    labels_table = pq.read_table(prep_raw_dir / "labels.parquet")

    assert set(labels_table.schema.names) >= {"client_id", "label_start", LABEL}

    # --- по одному примеру на клиента, три группы ----------
    assert set(examples.rows) == set(GROUPS)

    seen: set[int] = set()

    index = pq.read_table(processed / "cutoff_index.parquet")

    stamps = index.column("cutoff").to_numpy().astype("datetime64[us]")

    declared = {
        int(row["client_id"]): row
        for row in index.filter(stamps == examples.cutoff).to_pylist()
        if row["valid"]
    }

    for group, rows in examples.rows.items():

        assert rows

        for row in rows:

            client_id = int(row["client_id"])

            assert client_id not in seen
            seen.add(client_id)

            assert row["client_group"] == group
            assert client_id in examples.labels
            assert int(row["seq_end"]) == int(declared[client_id]["seq_end"])

    assert seen == set(declared)

    # --- признаки строго из прошлого -----------------------
    features = {
        group: build_features(processed, group, rows, examples.cutoff)
        for group, rows in examples.rows.items()
    }

    for group, matrix in features.items():
        assert matrix.shape == (len(examples.rows[group]), len(FEATURE_NAMES))

    # Метка в признаки не попадает ни под каким именем.
    assert not any(LABEL in name for name in FEATURE_NAMES)

    # Сдвинутый за срез префикс это ошибка, а не тихая утечка.
    spoiled = [dict(row) for row in examples.rows["val"]]

    spoiled[0]["seq_end"] = int(spoiled[0]["seq_end"]) + 10 ** 6

    with pytest.raises(DownstreamError):
        build_features(processed, "val", spoiled, examples.cutoff)

    # Снимок профиля позже среза тоже.
    shifted = [dict(row) for row in examples.rows["val"]]

    shifted[0]["snapshot_ts"] = examples.cutoff.astype("datetime64[us]").item()

    with pytest.raises(DownstreamError):
        build_features(processed, "val", shifted, examples.cutoff)

    # --- кодировщик обучен только на train -----------------
    encoder = fit_encoder(features["train"])

    known = set(encoder.categories_[0])

    invented = np.array(features["val"], dtype=object)
    invented[0, CATEGORICAL_SLICE.start] = "нет такой категории"

    encoded = encode(invented, encoder)

    assert encoded.shape[1] == features["val"].shape[1]

    # Неизвестная категория становится -1, а не ошибкой и не
    # новым уровнем.
    assert encoded[0, -len(encoder.categories_)] == -1
    assert "нет такой категории" not in known

    # --- эмбеддинги ----------------------------------------
    import torch

    from src.model.trainer import TrainConfig

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    trainer = Trainer(
        TrainConfig.from_dict(payload["train_config"]),
        env.tokenizer,
        env.table,
        env.unigram,
        "cpu",
    )

    vectors = build_embeddings(
        env, trainer, env.root, "val", examples.rows["val"], batch_size=4
    )

    assert vectors.shape == (len(examples.rows["val"]), trainer.model_config.d_model)
    assert np.isfinite(vectors).all()

    # Разные клиенты получают разные векторы.
    assert np.unique(vectors, axis=0).shape[0] > 1

    # --- прогон целиком ------------------------------------
    report = run_downstream(
        env=env,
        processed=processed,
        raw=prep_raw_dir,
        checkpoint=checkpoint,
        out_dir=tmp_path / "downstream",
        device="cpu",
        batch_size=4,
        seed=7,
        draws=20,
    )

    assert (tmp_path / "downstream" / "downstream.json").exists()
    assert (tmp_path / "downstream" / "downstream.md").exists()

    assert set(report["models"]) == {
        "features_boosting",
        "embedding_logistic",
        "both_boosting",
    }

    for name, item in report["models"].items():
        for split in ("val", "test", "train"):
            metrics = item[split]
            for key in ("roc_auc", "pr_auc"):
                value = metrics[key]
                # На фикстуре сплит может остаться без единого
                # положительного примера: тогда метрики нет, и
                # это должно быть None, а не выдуманное число.
                assert value is None or 0.0 <= value <= 1.0, (name, split, key)

    # train всегда с обоими классами: там метрика обязана быть.
    assert report["models"]["features_boosting"]["train"]["roc_auc"] is not None

    assert render_downstream(report)

    # Категории это последние колонки блока признаков и остаются
    # на своих местах, когда справа приписаны эмбеддинги. На
    # маленькой фикстуре подмена не видна: у эмбеддинга там мало
    # различных значений, и boosting принимает их молча.
    from src.model.downstream import N_CATEGORICAL, categorical_mask

    width = report["features"]["n"]

    alone = categorical_mask(width)
    with_vectors = categorical_mask(width + 128, width)

    assert alone.sum() == N_CATEGORICAL
    assert with_vectors.sum() == N_CATEGORICAL
    assert list(np.flatnonzero(alone)) == list(np.flatnonzero(with_vectors))
    assert not with_vectors[width:].any()
