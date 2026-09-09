"""
Диагностика внимания History Encoder.

Тесты делятся на две части. Первая проверяет, что маска
действительно запрещает ровно то, что обещает: это утверждение
о причинности, и оно проверяется возмущением входа. Вторая
проверяет, что сама машинерия ничего не меняет там, где не
должна.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from src.tokenizer.dataset import collate
from src.tokenizer.encode import encode_pairs
from src.tokenizer.config import USR_ID
from src.model.ablation import run_ablation
from src.model.backbone import build_backbone
from src.model.history_batching import (
    metadata_from_examples,
    prepare_history_batch,
    to_model_inputs,
)
from src.model.history_encoder import (
    ATTENTION_RULES,
    RULE_EVENTS_ISOLATED,
    RULE_EVENTS_VIA_PROFILE,
    RULE_FULL,
    RULE_PROFILE_ISOLATED,
    RULE_SELF_ONLY,
    HistoryEncoder,
    structural_block,
)
from src.model.trainer import Trainer, run_training

from tests.test_mlm_head import mlm_config, mlm_example, prepared
from tests.test_tok_encode import toy_vocab
from tests.test_trainer import env, small_config, store, trainer_for  # noqa: F401


# ============================================================
# СТРУКТУРА МАСКИ
# ============================================================


def test_full_blocks_nothing():
    assert not structural_block(RULE_FULL, 4).any()


def test_self_only_leaves_the_diagonal():
    blocked = structural_block(RULE_SELF_ONLY, 4)

    assert blocked.tolist() == (~torch.eye(4, dtype=torch.bool)).tolist()


def test_events_isolated_opens_only_the_profile_column():
    blocked = structural_block(RULE_EVENTS_ISOLATED, 4)

    assert blocked.tolist() == [
        [False, True, True, True],
        [False, False, True, True],
        [False, True, False, True],
        [False, True, True, False],
    ]


def test_events_via_profile_also_opens_the_profile_row():
    blocked = structural_block(RULE_EVENTS_VIA_PROFILE, 4)

    assert blocked.tolist() == [
        [False, False, False, False],
        [False, False, True, True],
        [False, True, False, True],
        [False, True, True, False],
    ]


def test_profile_isolated_cuts_the_profile_both_ways():
    blocked = structural_block(RULE_PROFILE_ISOLATED, 4)

    assert blocked.tolist() == [
        [False, True, True, True],
        [True, False, False, False],
        [True, False, False, False],
        [True, False, False, False],
    ]


def test_the_diagonal_is_never_blocked():
    for rule in ATTENTION_RULES:
        blocked = structural_block(rule, 5)
        assert not blocked.diagonal().any(), rule


def test_unknown_rule_is_refused():
    with pytest.raises(ValueError, match="неизвестное правило"):
        structural_block("everything", 4)


# ============================================================
# МАСКА С PADDING
# ============================================================


@pytest.fixture()
def encoder():
    return HistoryEncoder(mlm_config()).eval()


def padding_for(lengths: list[int], width: int) -> torch.Tensor:
    mask = torch.zeros(len(lengths), width, dtype=torch.bool)
    for index, length in enumerate(lengths):
        mask[index, length:] = True
    return mask


def test_mask_shape_follows_the_heads(encoder):
    padding = padding_for([5, 3], 5)

    mask = encoder.attention_mask(RULE_FULL, padding)

    assert mask.shape == (2 * encoder.config.n_heads, 5, 5)
    assert mask.dtype == torch.bool


def test_padded_keys_are_blocked_for_every_query(encoder):
    padding = padding_for([5, 3], 5)

    mask = encoder.attention_mask(RULE_FULL, padding)

    heads = encoder.config.n_heads

    # Пример 1 короче: ключи 3 и 4 закрыты для всех его запросов.
    assert bool(mask[heads:, :, 3:].all())

    # У полного примера не закрыто ничего.
    assert not bool(mask[:heads].any())


def test_padded_query_keeps_the_profile_open(encoder):
    padding = padding_for([5, 3], 5)

    mask = encoder.attention_mask(RULE_SELF_ONLY, padding)

    heads = encoder.config.n_heads

    # Ни одна строка не закрыта целиком: иначе softmax по пустому
    # множеству дал бы NaN.
    assert not bool(mask.all(dim=-1).any())

    # Запасной ключ это именно профиль.
    assert not bool(mask[heads:, 3:, 0].any())


def test_rows_repeat_per_head_and_differ_per_example(encoder):
    padding = padding_for([5, 3], 5)

    mask = encoder.attention_mask(RULE_FULL, padding)

    heads = encoder.config.n_heads

    for head in range(1, heads):
        assert torch.equal(mask[0], mask[head])
        assert torch.equal(mask[heads], mask[heads + head])

    assert not torch.equal(mask[0], mask[heads])


# ============================================================
# МАШИНЕРИЯ НЕЙТРАЛЬНА
# ============================================================


def test_full_rule_reproduces_the_default_path(encoder):
    torch.manual_seed(4)

    x = torch.randn(2, 6, encoder.config.d_model)

    padding = padding_for([6, 4], 6)

    with torch.no_grad():
        default = encoder(x, padding)
        through_mask = encoder(x, padding, RULE_FULL)

    torch.testing.assert_close(default, through_mask)


def test_full_rule_reproduces_the_default_backbone():
    config = mlm_config()

    history, targets, _ = prepared([[0, 1, 2, 3], [0, 5]])

    inputs = to_model_inputs(history, config)

    backbone = build_backbone(config, seed=17).eval()

    with torch.no_grad():
        default = backbone(inputs)
        through_mask = backbone(inputs, attention_rule=RULE_FULL)

    torch.testing.assert_close(default.client_embedding, through_mask.client_embedding)
    torch.testing.assert_close(default.event_embeddings, through_mask.event_embeddings)


def test_every_rule_gives_finite_outputs_with_padding(encoder):
    torch.manual_seed(5)

    x = torch.randn(3, 7, encoder.config.d_model)

    padding = padding_for([7, 4, 1], 7)

    for rule in ATTENTION_RULES:

        with torch.no_grad():
            out = encoder(x, padding, rule)

        assert torch.isfinite(out).all(), rule
        assert bool((out[padding] == 0).all()), rule


# ============================================================
# ЗАПРЕТЫ ДЕЙСТВУЮТ
# ============================================================


def run_with(backbone, config, colors, rule, flag: bool = True):
    """
    Один пример из трёх событий, заданные цвета и профиль.
    """

    vocab = toy_vocab()

    example = mlm_example(vocab, 0, [0, 1, 2], 24, colors=colors)

    example = replace(example, profile=encode_pairs(vocab, [("toy__flag", flag)], USR_ID))

    history = prepare_history_batch(
        collate([example]), metadata_from_examples([example]), 10
    )

    with torch.no_grad():
        return backbone(to_model_inputs(history, config), attention_rule=rule)


@pytest.fixture(scope="module")
def toy_backbone():
    return build_backbone(mlm_config(), seed=21).eval()


def test_events_isolated_hides_neighbours_and_the_client_vector(toy_backbone):
    """
    Изменение первого события не должно доходить ни до других
    событий, ни до профиля: пути наружу у события нет.
    """

    config = mlm_config()

    left = run_with(toy_backbone, config, ["red", "red", "red"], RULE_EVENTS_ISOLATED)
    right = run_with(toy_backbone, config, ["blue", "red", "red"], RULE_EVENTS_ISOLATED)

    assert not torch.allclose(left.event_embeddings[0], right.event_embeddings[0])

    torch.testing.assert_close(left.event_embeddings[1:], right.event_embeddings[1:])
    torch.testing.assert_close(left.client_embedding, right.client_embedding)


def test_events_via_profile_lets_the_client_vector_move(toy_backbone):
    config = mlm_config()

    left = run_with(toy_backbone, config, ["red", "red", "red"], RULE_EVENTS_VIA_PROFILE)
    right = run_with(toy_backbone, config, ["blue", "red", "red"], RULE_EVENTS_VIA_PROFILE)

    assert not torch.allclose(left.client_embedding, right.client_embedding)

    # И это уже другой режим, чем полная изоляция.
    isolated = run_with(toy_backbone, config, ["red", "red", "red"], RULE_EVENTS_ISOLATED)

    assert not torch.allclose(left.client_embedding, isolated.client_embedding)


def test_events_reach_each_other_through_the_profile_on_the_second_layer(toy_backbone):
    """
    У events_via_profile второй слой доносит соседа: профиль
    успевает собрать историю на первом.
    """

    config = mlm_config()

    left = run_with(toy_backbone, config, ["red", "red", "red"], RULE_EVENTS_VIA_PROFILE)
    right = run_with(toy_backbone, config, ["blue", "red", "red"], RULE_EVENTS_VIA_PROFILE)

    assert not torch.allclose(left.event_embeddings[1], right.event_embeddings[1])


def test_profile_isolated_cuts_the_profile_out(toy_backbone):
    config = mlm_config()

    left = run_with(toy_backbone, config, ["red", "red", "red"], RULE_PROFILE_ISOLATED, flag=True)
    right = run_with(toy_backbone, config, ["red", "red", "red"], RULE_PROFILE_ISOLATED, flag=False)

    # Профиль не доходит до событий.
    torch.testing.assert_close(left.event_embeddings, right.event_embeddings)

    # Но сам профиль другой, поэтому вектор клиента меняется.
    assert not torch.allclose(left.client_embedding, right.client_embedding)


def test_profile_isolated_keeps_events_out_of_the_client_vector(toy_backbone):
    config = mlm_config()

    left = run_with(toy_backbone, config, ["red", "red", "red"], RULE_PROFILE_ISOLATED)
    right = run_with(toy_backbone, config, ["blue", "blue", "blue"], RULE_PROFILE_ISOLATED)

    torch.testing.assert_close(left.client_embedding, right.client_embedding)

    assert not torch.allclose(left.event_embeddings, right.event_embeddings)


def test_full_lets_everything_through(toy_backbone):
    config = mlm_config()

    left = run_with(toy_backbone, config, ["red", "red", "red"], RULE_FULL, flag=True)
    right = run_with(toy_backbone, config, ["red", "red", "red"], RULE_FULL, flag=False)

    assert not torch.allclose(left.event_embeddings, right.event_embeddings)
    assert not torch.allclose(left.client_embedding, right.client_embedding)


def test_self_only_isolates_every_position(toy_backbone):
    config = mlm_config()

    left = run_with(toy_backbone, config, ["red", "red", "red"], RULE_SELF_ONLY, flag=True)
    right = run_with(toy_backbone, config, ["blue", "red", "red"], RULE_SELF_ONLY, flag=False)

    torch.testing.assert_close(left.event_embeddings[1:], right.event_embeddings[1:])
    assert not torch.allclose(left.event_embeddings[0], right.event_embeddings[0])
    assert not torch.allclose(left.client_embedding, right.client_embedding)


# ============================================================
# ОЦЕНКА С ПРАВИЛОМ
# ============================================================


@pytest.fixture(scope="module")
def splits(env):
    from src.model.trainer import build_validation

    made, _ = build_validation(env, small_config(), trainer_for(env).model_config)

    return made


def test_rule_changes_the_metrics_but_not_the_model(env, splits):
    trainer = trainer_for(env)

    trainer.train_mode()

    before = [parameter.detach().clone() for parameter in trainer.parameters()]

    torch.manual_seed(77)

    state = torch.get_rng_state().clone()

    baseline = trainer.evaluate(splits)

    restricted = trainer.evaluate(splits, attention_rule=RULE_SELF_ONLY)

    assert torch.equal(torch.get_rng_state(), state)

    for old, new in zip(before, trainer.parameters()):
        assert torch.equal(old, new)

    assert trainer.backbone.training

    for name in splits:

        assert np.isfinite(restricted[name]["field_balanced_ce"])
        assert restricted[name]["n_targets"] == baseline[name]["n_targets"]

    assert any(
        restricted[name]["field_balanced_ce"] != baseline[name]["field_balanced_ce"]
        for name in splits
    )


def test_full_rule_matches_the_default_evaluation(env, splits):
    trainer = trainer_for(env)

    baseline = trainer.evaluate(splits)
    through_mask = trainer.evaluate(splits, attention_rule=RULE_FULL)

    for name in splits:
        assert through_mask[name]["field_balanced_ce"] == pytest.approx(
            baseline[name]["field_balanced_ce"]
        )


# ============================================================
# ЦЕЛИКОМ
# ============================================================


@pytest.fixture(scope="module")
def ablation(env, tmp_path_factory):
    trained = tmp_path_factory.mktemp("trained")

    run = run_training(env, small_config(), trained, device="cpu", quiet=True)

    out = tmp_path_factory.mktemp("ablation")

    report = run_ablation(env, trained / "best.pt", out, device="cpu", quiet=True)

    return {"run": run, "report": report, "out": out}


def test_every_rule_is_evaluated(ablation):
    report = ablation["report"]

    assert report["rules"] == list(ATTENTION_RULES)

    for name in report["aggregates"]:
        assert [row["rule"] for row in report["aggregates"][name]] == list(ATTENTION_RULES)


def test_baseline_matches_the_run_report(ablation):
    report = ablation["report"]

    for name, item in report["baseline_check"]["splits"].items():
        assert item["saved"] == pytest.approx(item["recomputed"])
        assert item["difference"] == pytest.approx(0.0, abs=1e-9)


def test_full_has_zero_deltas(ablation):
    for rows in ablation["report"]["aggregates"].values():

        full = next(row for row in rows if row["rule"] == "full")

        assert full["delta_field_balanced_ce"] == 0.0
        assert full["delta_accuracy"] == 0.0


def test_targets_are_the_same_for_every_rule(ablation):
    for rows in ablation["report"]["aggregates"].values():
        assert len({row["n_targets"] for row in rows}) == 1


def test_field_rows_carry_every_rule(ablation):
    report = ablation["report"]

    for rows in report["fields"].values():

        assert rows

        for row in rows:
            assert set(row["ce_model"]) == set(ATTENTION_RULES)
            assert row["delta_ce_model"]["full"] == 0.0
            assert row["n_targets"] > 0


def test_fields_are_sorted_by_the_first_restriction(ablation):
    for rows in ablation["report"]["fields"].values():

        losses = [row["delta_ce_model"][RULE_EVENTS_ISOLATED] for row in rows]

        assert losses == sorted(losses, reverse=True)


def test_artefacts_are_written(ablation):
    out = ablation["out"]

    assert (out / "ablation.json").exists()

    text = (out / "ablation.md").read_text(encoding="utf-8")

    assert "Диагностика внимания" in text
    assert "events_isolated" in text
    assert "Сверка baseline" in text

    saved = json.loads((out / "ablation.json").read_text(encoding="utf-8"))

    assert saved["rules"] == list(ATTENTION_RULES)


def test_a_foreign_checkpoint_is_refused(env, ablation, tmp_path):
    """
    Диагностика на чужих масках бессмысленна, поэтому загрузка
    обязана отказать.
    """

    from src.tokenizer.config import IncompatibleArtifactsError

    path = tmp_path / "spoiled.pt"

    payload = torch.load(
        ablation["run"]["checkpoints"]["best"], map_location="cpu", weights_only=False
    )

    payload["splits"]["val_time"]["targets_sha256"] = "0" * 64

    torch.save(payload, path)

    with pytest.raises(IncompatibleArtifactsError, match="маски и цели"):
        run_ablation(env, path, tmp_path / "out", device="cpu", quiet=True)


def test_rules_must_include_full(env, ablation, tmp_path):
    with pytest.raises(ValueError, match="без full"):
        run_ablation(
            env,
            ablation["run"]["checkpoints"]["best"],
            tmp_path / "nofull",
            device="cpu",
            rules=(RULE_SELF_ONLY,),
            quiet=True,
        )


def test_unknown_rule_is_refused_early(env, ablation, tmp_path):
    with pytest.raises(ValueError, match="неизвестные правила"):
        run_ablation(
            env,
            ablation["run"]["checkpoints"]["best"],
            tmp_path / "bogus",
            device="cpu",
            rules=(RULE_FULL, "everything"),
            quiet=True,
        )
