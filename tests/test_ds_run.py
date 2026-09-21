from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.dataset.run import main
from src.dataset.storage import MANIFEST_FILE
from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK

from tests.ds_fixtures import ready


# ============================================================
# ИДЕЯ
# ============================================================
#
# Здесь проверяется не формат примера, а поведение команды: что
# она берёт за правила, куда кладёт результат и что делает с уже
# готовым набором рядом.
# ============================================================


@pytest.fixture(scope="module")
def prepared_dataset(tmp_path_factory) -> tuple[Path, Path, Path, Path]:

    base = tmp_path_factory.mktemp("ds_run")

    root, out, target = ready(base)

    return root, out, target, base / "home"


def _cli(*args: str) -> int:
    with pytest.raises(SystemExit) as result:
        main(list(args))
    return int(result.value.code)


def _common(prepared_dataset) -> tuple[str, ...]:

    root, out, target, home = prepared_dataset

    return (
        "--raw-root", str(root), "--name", "ds",
        "--processed", str(out), "--vocab", str(target), "--out", str(home),
    )


# ============================================================
# КОМАНДЫ
# ============================================================


def test_contract_names_the_future_dataset(prepared_dataset, capsys):

    *_rest, home = prepared_dataset

    assert _cli("contract", *_common(prepared_dataset)) == EXIT_OK

    printed = capsys.readouterr().out

    assert "набор получит имя" in printed

    report = json.loads((home / "contract" / "dataset_contract.json").read_text(encoding="utf-8"))

    assert report["dataset_id"]
    assert report["channels"]["model"]
    assert report["readiness"]["status"] in ("ready", "diagnostic")

    # Тождество набора не содержит ни путей, ни времени запуска.
    text = json.dumps(report["identity"], ensure_ascii=False)

    assert str(home) not in text
    assert "\\\\" not in text


def test_measure_reports_lengths_without_writing_samples(prepared_dataset, capsys):

    *_rest, home = prepared_dataset

    assert _cli("measure", *_common(prepared_dataset), "--group", "train") == EXIT_OK

    printed = capsys.readouterr().out

    assert "events_per_client" in printed

    files = list((home / "measurements").glob("*.json"))

    assert len(files) == 1

    report = json.loads(files[0].read_text(encoding="utf-8"))

    assert report["lengths"]["events_per_client"]["max"] > 0
    assert not (home / "shards").exists(), "измерение примеров не сохраняет"


def test_build_then_check(prepared_dataset, capsys):

    *_rest, home = prepared_dataset

    assert _cli("build", *_common(prepared_dataset)) == EXIT_OK

    printed = capsys.readouterr().out

    assert "набор" in printed

    dataset_id = next(
        path.name for path in home.iterdir()
        if path.is_dir() and (path / MANIFEST_FILE).exists()
    )

    assert _cli("check", *_common(prepared_dataset), "--dataset-id", dataset_id,
                "--recompute", "2") == EXIT_OK

    printed = capsys.readouterr().out

    assert "[ок]" in printed

    report = json.loads(
        (home / dataset_id / "check_report.json").read_text(encoding="utf-8")
    )

    assert report["ok"]
    assert report["violations"] == 0
    assert all(item["identical"] for item in report["recomputed"])


def test_existing_dataset_is_not_rebuilt_silently(prepared_dataset, capsys):

    assert _cli("build", *_common(prepared_dataset)) == EXIT_BLOCKED

    assert "уже собран" in capsys.readouterr().out

    assert _cli("build", *_common(prepared_dataset), "--force") == EXIT_OK


def test_check_requires_an_explicit_dataset(prepared_dataset):
    """
    «Последний собранный» набор не существует: потребитель
    называет его явно.
    """

    with pytest.raises(SystemExit) as result:
        main(["check", *_common(prepared_dataset)])

    assert "укажите --dataset-id" in str(result.value)


def test_policy_flags_change_the_dataset(prepared_dataset, capsys):

    *_rest, home = prepared_dataset

    before = {path.name for path in home.iterdir() if path.is_dir()}

    assert _cli(
        "build", *_common(prepared_dataset),
        "--policy", "recent_plus_milestones", "--max-events", "6", "--max-tokens", "100000",
    ) == EXIT_OK

    capsys.readouterr()

    after = {path.name for path in home.iterdir() if path.is_dir()}

    assert len(after - before) == 1, "другая политика это другой набор"

    name = next(iter(after - before))

    manifest = json.loads((home / name / MANIFEST_FILE).read_text(encoding="utf-8"))

    assert manifest["config"]["context"]["policy"] == "recent_plus_milestones"
    assert manifest["config"]["context"]["max_events"] == 6


# ============================================================
# ВЕРСИИ
# ============================================================


def test_version_tracks_sources():
    """
    Правка любого модуля пакета без поднятия версии реализации
    не проходит: прежние наборы считались бы собранными этим
    кодом.
    """

    from src.dataset import (
        build,
        check,
        collate,
        context,
        dependencies,
        encoding,
        inputs,
        measure,
        reader,
        run,
        sample,
        settings,
        storage,
        targets,
        version,
    )
    from src.dataset import report as report_module

    modules = {
        "build": build,
        "check": check,
        "collate": collate,
        "context": context,
        "dependencies": dependencies,
        "encoding": encoding,
        "inputs": inputs,
        "measure": measure,
        "reader": reader,
        "report": report_module,
        "run": run,
        "sample": sample,
        "settings": settings,
        "storage": storage,
        "targets": targets,
        "version": version,
    }

    stored = json.loads(Path("tests/ds_sources.json").read_text(encoding="utf-8"))

    assert stored["version"] == version.IMPLEMENTATION_VERSION

    actual = {
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in modules.items()
    }

    assert stored["modules"] == actual, (
        "модули датасета изменены: поднимите IMPLEMENTATION_VERSION "
        "и обновите tests/ds_sources.json"
    )
