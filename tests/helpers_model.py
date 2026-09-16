"""
Общие конфигурации модели и сборщики batch для тестов.

Опирается на tests/helpers_data.py: сначала данные, потом модель
поверх них.
"""

from __future__ import annotations

from dataclasses import replace

from src.tokenizer.dataset import collate
from src.tokenizer.masking import MODE_COMBINED, Masker, MaskingConfig
from src.model.config import ARCHITECTURE, ModelConfig
from src.model.history_batching import metadata_from_examples, prepare_history_batch
from src.model.mlm_batching import build_targets
from src.model.mlm_head import FieldTable
from src.model.trainer import TrainConfig, Trainer

from tests.helpers_data import mlm_example, toy_vocab


def toy_config(vocab=None) -> ModelConfig:
    vocab = vocab or toy_vocab()
    return ModelConfig(vocab_size=vocab.size, max_position_embeddings=16)


def mlm_config(vocab=None) -> ModelConfig:
    vocab = vocab or toy_vocab()
    return ModelConfig(vocab_size=vocab.size, max_position_embeddings=16, dropout=0.0)


def full_config(vocab=None) -> ModelConfig:

    vocab = vocab or toy_vocab()

    return ModelConfig(
        vocab_size=vocab.size,
        max_position_embeddings=16,
        **{**ARCHITECTURE, "dropout": 0.0},
    )


def small_config(**overrides) -> TrainConfig:
    """
    Конфигурация, на которой тест идёт секунды, а не минуты.
    """

    base = TrainConfig(
        max_train_clients=4,
        max_val_clients=2,
        batch_size=2,
        eval_batch_size=2,
        max_events_per_history=24,
        max_steps=3,
        warmup_steps=1,
        eval_every=2,
        log_every=1,
        precision="float32",
    )

    return replace(base, **overrides) if overrides else base


def combined_config(**overrides) -> TrainConfig:
    return replace(
        small_config(),
        masking_mode=MODE_COMBINED,
        token_rate=0.15,
        event_rate=0.10,
        key_rate=0.10,
        **overrides,
    )


def trainer_for(env, config=None) -> Trainer:
    return Trainer(config or small_config(), env.tokenizer, env.table, env.unigram, "cpu")


def random_names(parameters: dict) -> list[str]:
    """
    Имена тензоров со случайной инициализацией.

    Биасы внимания и веса LayerNorm задаются константами, они
    совпадают у любых двух блоков по построению, а не потому,
    что блок скопирован. Сравнивать имеет смысл только то, что
    действительно разыгрывается.
    """

    return [name for name, tensor in parameters.items() if tensor.detach().unique().numel() > 2]


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
