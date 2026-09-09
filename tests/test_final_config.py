"""
Финальный конфиг: он должен выражаться командой, а не правкой
кода, и всё заявленное в нём должно попадать в отчёт.

Отдельно проверяется, что none это значение, а не отсутствие
значения: у обрезки истории и у числа клиентов None это
собственная ветка, и большим числом её не заменить.
"""

from __future__ import annotations

import json

import pytest

from src.model.train import build_parser, config_from_args
from src.model.trainer import (
    BEST_SCOPE_RECENT,
    TrainConfig,
    run_training,
)
from src.model.targets import HISTORY_EXCLUDES, POLICY_HISTORY
from src.tokenizer.masking import SCHEME_EXAMPLE

from tests.test_trainer import env, small_config  # noqa: F401


FINAL_FLAGS = [
    "run",
    "--name",
    "v21_10k",
    "--structure",
    "session",
    "--d-model",
    "128",
    "--n-heads",
    "4",
    "--dim-feedforward",
    "512",
    "--profile-layers",
    "1",
    "--event-layers",
    "3",
    "--session-layers",
    "1",
    "--history-layers",
    "2",
    "--max-events",
    "none",
    "--train-clients",
    "none",
    "--val-clients",
    "none",
    "--masking-mode",
    "combined",
    "--token-rate",
    "0.15",
    "--event-rate",
    "0.10",
    "--key-rate",
    "0.10",
    "--mask-scheme",
    "example",
    "--target-policy",
    "history",
    "--epochs",
    "1",
    "--stream-validation",
    "--best-metric",
    "recent",
    "--final-splits",
    "test_client,test_time",
    "--batch-size",
    "2",
    "--eval-batch-size",
    "2",
]


def test_cli_expresses_the_final_config_and_a_tiny_run_records_preflight_recent_and_excluded(
    env, tmp_path
):

    # --- команда даёт ровно заявленный конфиг --------------
    config = config_from_args(build_parser().parse_args(FINAL_FLAGS))

    assert config.structure == "session"
    assert (config.d_model, config.n_heads, config.dim_feedforward) == (128, 4, 512)
    assert (
        config.n_profile_layers,
        config.n_event_layers,
        config.n_session_layers,
        config.n_history_layers,
    ) == (1, 3, 1, 2)

    # none это None, а не большое число.
    assert config.max_events_per_history is None
    assert config.max_train_clients is None
    assert config.max_val_clients is None

    assert config.masking_mode == "combined"
    assert (config.token_rate, config.event_rate, config.key_rate) == (0.15, 0.10, 0.10)
    assert config.mask_scheme == SCHEME_EXAMPLE
    assert config.target_policy == POLICY_HISTORY
    assert config.epochs == 1
    assert config.stream_validation is True
    assert config.best_metric == BEST_SCOPE_RECENT
    assert config.final_splits == ("test_client", "test_time")

    assert config.masking().exclude_fields == HISTORY_EXCLUDES

    # --- ничего не передали значит прежние умолчания -------
    bare = config_from_args(build_parser().parse_args(["run", "--name", "dev"]))

    assert bare == TrainConfig(precision="auto")

    # --- конфиг из старого checkpoint грузится -------------
    legacy = {
        name: value
        for name, value in TrainConfig().as_dict().items()
        if name
        not in (
            "target_policy",
            "mask_scheme",
            "best_metric",
            "final_splits",
            "stream_validation",
        )
    }

    assert TrainConfig.from_dict(legacy) == TrainConfig()

    # --- маленький прогон в тех же режимах -----------------
    small = small_config(
        target_policy=POLICY_HISTORY,
        mask_scheme=SCHEME_EXAMPLE,
        stream_validation=True,
        best_metric=BEST_SCOPE_RECENT,
        final_splits=("test_client",),
        masking_mode="combined",
        max_events_per_history=None,
        max_steps=2,
        eval_every=2,
    )

    out = tmp_path / "final"

    report = run_training(env, small, out, device="cpu", quiet=True, preflight=True)

    # Полные истории подтверждены проходом, а не настройкой.
    checks = report["truncation_check"]

    assert checks["passed"]
    assert checks["n_truncated"] == 0
    assert "histories_over_manifest_limit" in checks

    # Политика целей названа и исполнена.
    assert report["targets"]["policy"] == POLICY_HISTORY
    assert report["targets"]["excluded"]

    declared = set(report["targets"]["excluded"])

    for name in report["splits"]:

        item = report["after"][name]

        named = set(item["excluded_fields"])

        assert named
        assert named <= declared

        # Разница только вырожденные поля: у них целей и не
        # могло быть, и «нечего предсказывать» точнее, чем
        # «выведено политикой».
        statuses = {field["field"]: field["status"] for field in item["fields"]}

        for missing in declared - named:
            assert statuses[missing] == "degenerate", missing

        for excluded in named:
            assert statuses[excluded] == "excluded"

        assert item["recent"]["n_targets"] <= item["n_targets"]

    # Validation собрана потоком.
    for description in report["splits"].values():
        assert description["settings"]["stream"] is True
        assert description["settings"]["masking"]["scheme"] == SCHEME_EXAMPLE

    # best выбран по срезу нового месяца.
    assert report["config"]["best_metric"] == BEST_SCOPE_RECENT

    # Финальная оценка сделана один раз и на заявленном наборе.
    final = report["final"]

    assert set(final["metrics"]) == {"test_client"}
    assert final["weights"] in ("best.pt", "last.pt")
    assert final["metrics"]["test_client"]["recent"]["scope"]

    # Отчёт читается и говорит про оба среза.
    text = (out / "report.md").read_text(encoding="utf-8")

    assert "Только месяц наблюдения" in text
    assert "Финальная оценка" in text
    assert "Политика history" in text

    # И всё это записано в JSON, а не только напечатано.
    stored = json.loads((out / "report.json").read_text(encoding="utf-8"))

    assert stored["config"]["mask_scheme"] == SCHEME_EXAMPLE
    assert stored["config"]["stream_validation"] is True
    assert stored["config"]["final_splits"] == ["test_client"]
