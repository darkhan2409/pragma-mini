from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.generator import emit


# Датасет для проверок RAW. 24 клиента хватает, чтобы каждый
# поток был непустым: приложением пользуются не все.
EMIT_CLIENTS = 24
EMIT_CHUNK = 8


@pytest.fixture(scope="session")
def emit_clients() -> int:
    return EMIT_CLIENTS


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Один раз за сессию собирает RAW во временный каталог.
    data/raw при этом не затрагивается.
    """

    out = tmp_path_factory.mktemp("raw")

    emit.generate_dataset(
        total_clients=EMIT_CLIENTS,
        chunk_clients=EMIT_CHUNK,
        out_dir=out,
        workers=1,
    )

    return out


@pytest.fixture(scope="session")
def raw_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(raw_dir / f"{name}.parquet")
        for name in emit.SCHEMAS
    }


# ============================================================
# PREPROCESSING
# ============================================================

# Спецификация требует end-to-end минимум на 100 клиентах.
PREP_CLIENTS = 100
PREP_CHUNK = 25


@pytest.fixture(scope="session")
def prep_clients() -> int:
    return PREP_CLIENTS


@pytest.fixture(scope="session")
def prep_raw_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Отдельный RAW на 100 клиентов: фикстура генератора на 24
    клиента остаётся нетронутой.
    """

    out = tmp_path_factory.mktemp("prep_raw")

    emit.generate_dataset(
        total_clients=PREP_CLIENTS,
        chunk_clients=PREP_CHUNK,
        out_dir=out,
        workers=1,
    )

    return out


@pytest.fixture(scope="session")
def prep_run(prep_raw_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """
    Один прогон конвейера на сессию.
    """

    from src.preprocessing.run import run

    root = tmp_path_factory.mktemp("prep_out")

    processed = root / "processed"
    artifacts = root / "artifacts"

    result = run(prep_raw_dir, "test", processed, artifacts, quiet=True)

    return {"processed": processed, "artifacts": artifacts, **result}


# ============================================================
# TOKENIZER
# ============================================================


@pytest.fixture(scope="session")
def tok_run(prep_run, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """
    Один прогон tokenizer поверх готового preprocessing.
    """

    from src.tokenizer.run import run

    root = tmp_path_factory.mktemp("tok_out")

    tokenized = root / "tokenized"
    vocab = root / "vocab"

    result = run(
        processed_in=prep_run["processed"],
        artifacts_in=prep_run["artifacts"],
        out_dir=tokenized,
        vocab_out=vocab,
        quiet=True,
    )

    return {
        "tokenized": tokenized,
        "vocab": vocab,
        "processed": prep_run["processed"],
        "artifacts": prep_run["artifacts"],
        **result,
    }

# ============================================================
# ОКРУЖЕНИЕ МОДЕЛИ
# ============================================================


@pytest.fixture(scope="module")
def env(tok_run):
    # Импорт внутри фикстуры: слой модели ждёт препроцессинга под
    # новый контракт RAW и при сборе тестов не должен требоваться.
    from src.model.trainer import load_environment

    return load_environment(tok_run["tokenized"], tok_run["vocab"], tok_run["artifacts"])


@pytest.fixture(scope="module")
def store(env):
    from src.model.data import ClientStore

    return ClientStore(env.root, "train", env.vocab_dir, max_clients=4)
