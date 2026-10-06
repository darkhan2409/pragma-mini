from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.generator import config, emit


# ============================================================
# ИДЕЯ
# ============================================================
#
# Мир и задачи разделены:
#
#   - в ленте нет ни одного имени скрытого состояния: ни ключа,
#     ни значения, которые пересказывали бы недовольство, долю
#     банка, уход, отлучку или их причины;
#   - типы событий и ключи payload — только из каталога;
#   - генератор не импортирует ничего за своими пределами, кроме
#     стандартной библиотеки, numpy и pyarrow: о downstream-задачах
#     он не знает;
#   - правду симуляции (truth/) не читает никто вне генератора, и
#     выход препроцессинга от неё не зависит;
#   - выгрузка несёт версию семантики генератора.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

START = datetime(2024, 1, 1)
END = datetime(2024, 6, 1)

# Значения скрытого состояния (behaviour/engagement, truth/).
HIDDEN = frozenset({
    "friction", "affinity", "regime", "lapsed", "engagement", "migrating", "away",
    "no_app", "no_cards", "offline", "never_started", "exogenous", "before_window",
    "baseline", "rollback", "salary_here", "frailty", "burn_in",
})


@pytest.fixture(scope="module")
def raw(tmp_path_factory) -> Path:

    out = tmp_path_factory.mktemp("isolation") / "raw"

    emit.generate_dataset(
        total_clients=16, out_dir=out, seed=55, world_seed=42, history_start=START,
        history_end=END, workers=1, community_size=4, quiet=True,
    )

    return out


def test_the_tape_names_no_hidden_state(raw):

    catalogue = config.key_catalogue()

    table = pq.read_table(raw / "events.parquet", columns=["payload"])

    seen_types = set()

    for text in table.column("payload").to_pylist():

        payload = json.loads(text)
        kind = payload["type"]
        seen_types.add(kind)

        allowed = {item["name"] for item in catalogue[kind]["fields"]}

        assert set(payload) <= allowed, (kind, sorted(set(payload) - allowed))

        for key, value in payload.items():
            assert key not in HIDDEN, (kind, key)
            if isinstance(value, str):
                assert value not in HIDDEN, (kind, key, value)

    assert seen_types <= set(config.EVENT_TYPES)


def test_the_generator_imports_nothing_downstream():

    allowed = {"numpy", "pyarrow"}

    offenders = []

    for path in sorted((ROOT / "src" / "generator").rglob("*.py")):

        tree = ast.parse(path.read_text(encoding="utf-8"))

        for node in ast.walk(tree):

            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue

            for name in names:
                top = name.split(".")[0]
                if name.startswith("src.generator") or top in allowed or top in _stdlib():
                    continue
                offenders.append((str(path.relative_to(ROOT)), name))

    assert offenders == []


def _stdlib() -> frozenset:
    import sys
    return frozenset(sys.stdlib_module_names) | {"__future__"}


def test_nothing_outside_the_generator_reads_truth():
    """
    Вход PRAGMA, признаки и метки churn и fraud правду симуляции
    не читают: всё, что знает банк, лежит в events и profile.
    """

    pattern = re.compile(r"""["'/]truth\b""")

    places = [
        path
        for folder in (ROOT / "src", ROOT / "churn_baseline" / "churn", ROOT / "fraud_baseline" / "fraud")
        for path in sorted(folder.rglob("*.py"))
        if not path.is_relative_to(ROOT / "src" / "generator")
    ]

    assert places

    offenders = [str(path.relative_to(ROOT)) for path in places if pattern.search(path.read_text(encoding="utf-8"))]

    assert offenders == []


def test_preprocessing_does_not_depend_on_truth(raw, tmp_path):

    from src.preprocessing.canonical.build import build_group
    from src.preprocessing.settings import PreprocessingConfig

    bare = tmp_path / "bare"
    shutil.copytree(raw, bare)
    shutil.rmtree(bare / "truth")

    settings = PreprocessingConfig.load(None)

    build_group(raw, tmp_path / "with", settings, "train")
    build_group(bare, tmp_path / "without", settings, "train")

    produced = sorted(path.relative_to(tmp_path / "with") for path in (tmp_path / "with").rglob("*") if path.is_file())

    assert produced

    for name in produced:
        left = hashlib.sha256((tmp_path / "with" / name).read_bytes()).hexdigest()
        right = hashlib.sha256((tmp_path / "without" / name).read_bytes()).hexdigest()
        assert left == right, name


def test_the_export_carries_the_generator_version_and_its_truth(raw):

    manifest = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["generator_version"] == config.GENERATOR_VERSION == "1.0.0"

    for name in ("transitions", "states"):
        assert manifest[f"{name}_rows"] == pq.ParquetFile(raw / "truth" / f"{name}.parquet").metadata.num_rows
        assert manifest[f"{name}_sha256"] == hashlib.sha256((raw / "truth" / f"{name}.parquet").read_bytes()).hexdigest()
