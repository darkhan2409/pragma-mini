from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

from src.dataset.inputs import DatasetInputs
from src.dataset.settings import ContextPolicy, DatasetConfig
from src.preprocessing.run import EXIT_CONTRACT_MISMATCH, EXIT_OK
from src.tokenization.layout import FrozenArtifacts

from tests.prep_fixtures import MiniRaw, purchase_payload
from tests.tok_fixtures import FULL_HORIZON, PRODUCTS, SEEDS, WORLD_SEED, build_vocab, cli, prepared


# ============================================================
# ИДЕЯ
# ============================================================
#
# Датасет проверяется на том же наборе, что и токенизатор: три
# группы, прошедшие пять этапов препроцессинга, и замороженный
# словарь поверх них.
#
# Отдельно живёт набор с ПОЗДНИМ исправлением: версия 2 события
# датирована позже версии 1, и между ними есть срез. Без него
# нельзя отличить «исправление видно с момента своего события»
# от «исправление видно всегда»: в обычной фикстуре обе версии
# лежат на одном времени и проверяют только края.
# ============================================================


# Срезы вокруг позднего исправления.
BEFORE_EVENT = datetime(2025, 3, 5)
BETWEEN_VERSIONS = datetime(2025, 3, 15)
AFTER_CORRECTION = datetime(2025, 3, 25)

ORIGINAL_AMOUNT = 33_000
CORRECTED_AMOUNT = 44_000


def ready(tmp_path: Path, **kwargs) -> tuple[Path, Path, Path]:
    """
    Готовые данные и замороженный словарь над ними.
    """

    root, out = prepared(tmp_path, **kwargs)

    target = tmp_path / "vocab"

    build_vocab(root, out, target)

    return root, out, target


def artifacts_of(target: Path) -> FrozenArtifacts:
    return FrozenArtifacts.load(target)


def config_all(**kwargs) -> DatasetConfig:
    return replace(DatasetConfig(), **kwargs)


def config_tight(max_events: int = 4, max_tokens: int = 4096,
                 milestone_share: float = 0.5, **kwargs) -> DatasetConfig:
    """
    Тесный бюджет: отбор обязан сработать даже на короткой
    истории фикстуры.
    """

    policy = ContextPolicy(
        policy="recent_plus_milestones",
        max_events=max_events,
        max_tokens=max_tokens,
        milestone_share=milestone_share,
    )

    return replace(DatasetConfig(), context=policy, **kwargs)


def inputs_of(root: Path, out: Path, target: Path, config: DatasetConfig | None = None) -> DatasetInputs:
    return DatasetInputs.open(out, root, target, config or config_all())


# ------------------------------------------------------------
# НАБОР С ПОЗДНИМ ИСПРАВЛЕНИЕМ
# ------------------------------------------------------------


def _late_correction(mini: MiniRaw, client_id: str) -> None:
    """
    Покупка и её исправление, датированное на десять дней позже.

    Место события при этом задаёт первая версия, поэтому в
    истории исправленная запись остаётся ПЕРЕД покупкой от
    десятого числа, хотя её собственное время позже.
    """

    mini.cover_all(client_id, first_seen="2023-01-01")
    mini.profile_version(client_id, 1, "2023-01-01", declared_income=200_000, age=30, city="Almaty")

    corrected = mini.event(
        client_id, "purchase", "2025-03-10 12:00:00",
        payload=purchase_payload(amount=ORIGINAL_AMOUNT, merchant_name="Magnum Astana"),
    )

    mini.event(
        client_id, "purchase", "2025-03-20 09:00:00", event_id=corrected, version=2,
        payload=purchase_payload(amount=CORRECTED_AMOUNT, merchant_name="Magnum Astana"),
    )

    # Соседняя покупка между версиями: по ней видно, что
    # исправление не переезжает в конец истории.
    mini.event(
        client_id, "purchase", "2025-03-18 12:00:00",
        payload=purchase_payload(amount=7_000, merchant_name="Small Shop"),
    )


def build_late_correction(root: Path) -> Path:
    """
    Три группы, где у train-клиента есть позднее исправление.
    """

    root = Path(root)

    for name in ("train", "val", "test"):

        mini = MiniRaw(root / name, history_start=FULL_HORIZON, seed=SEEDS[name],
                       world_seed=WORLD_SEED)

        for item in PRODUCTS:
            mini.product(valid_from=FULL_HORIZON, **item)

        _late_correction(mini, f"{name}_c1")

        mini.write()

    return root


def prepared_late_correction(tmp_path: Path) -> tuple[Path, Path, Path]:
    """
    Набор с поздним исправлением, прошедший препроцессинг, и
    словарь над ним.
    """

    root = build_late_correction(tmp_path / "raw")
    out = tmp_path / "processed"

    assert cli("passport", "--raw-root", str(root), "--out", str(out), "--name", "ds") in (
        EXIT_OK,
        EXIT_CONTRACT_MISMATCH,
    )
    assert cli("canonical", "--raw-root", str(root), "--out", str(out), "--name", "ds") == EXIT_OK
    assert cli("split", "--raw-root", str(root), "--out", str(out), "--name", "ds") == EXIT_OK

    for group in ("train", "val", "test"):
        assert cli("semantic", "--name", "ds", "--group", group,
                   "--raw", str(root / group), "--out", str(out)) == EXIT_OK

    target = tmp_path / "vocab"

    build_vocab(root, out, target)

    return root, out, target


__all__ = [
    "AFTER_CORRECTION",
    "BEFORE_EVENT",
    "BETWEEN_VERSIONS",
    "CORRECTED_AMOUNT",
    "ORIGINAL_AMOUNT",
    "artifacts_of",
    "build_late_correction",
    "config_all",
    "config_tight",
    "inputs_of",
    "prepared_late_correction",
    "ready",
]
