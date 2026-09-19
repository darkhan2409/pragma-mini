from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.preprocessing.artifacts import write_json
from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK
from src.tokenization.contract import CONFIG_FILE
from src.tokenization.run import group_cutoff, main, prepare_directory, run_dirname
from src.tokenization.settings import TokenizerConfig
from src.tokenization.transform import EVENTS_FILE, MANIFEST_FILE, TransformError

from tests.tok_fixtures import FIT_END, build_vocab, prepared


# ============================================================
# ИДЕЯ
# ============================================================
#
# Здесь проверяется не формат токенов, а поведение команды: что
# она берёт за правила, куда кладёт результат и что делает с уже
# лежащим рядом чужим прогоном.
#
# Это ровно те места, где ошибка тихо уничтожает работу: диагно-
# стический разбор одного клиента, затирающий полный прогон,
# и cutoff, взятый не из того разделения.
# ============================================================


@pytest.fixture(scope="module")
def ready(tmp_path_factory) -> tuple[Path, Path, Path]:

    base = tmp_path_factory.mktemp("tok_run")

    root, out = prepared(base)

    target = base / "vocab"

    build_vocab(root, out, target)

    return root, out, target


def _cli(*args: str) -> int:
    with pytest.raises(SystemExit) as result:
        main(list(args))
    return int(result.value.code)


# ============================================================
# КАТАЛОГ РЕЗУЛЬТАТА
# ============================================================


def test_partial_runs_get_their_own_directory():
    """
    Полный прогон группы, несколько срезов и разбор одного
    клиента это разные результаты, и лежать они обязаны врозь.
    """

    one = [datetime(2026, 1, 1)]
    many = [datetime(2025, 6, 1), datetime(2026, 1, 1)]

    full = run_dirname("train", one, None)
    several = run_dirname("train", many, None)
    single = run_dirname("train", one, "c1")

    assert full == "train__2026-01-01"
    assert len({full, several, single}) == 3
    assert "c1" in single
    assert several.endswith("__2")


def test_existing_result_is_not_erased_silently(tmp_path):
    """
    Непустой каталог результата не перезаписывается без явного
    разрешения.
    """

    directory = tmp_path / "out"
    directory.mkdir()
    (directory / "events.parquet").write_bytes(b"prior")

    with pytest.raises(TransformError, match="уже лежит результат"):
        prepare_directory(directory, force=False)

    assert (directory / "events.parquet").exists()

    prepare_directory(directory, force=True)

    assert not (directory / "events.parquet").exists()


def test_single_client_run_does_not_overwrite_the_group(ready, tmp_path):
    """
    Диагностический разбор одного клиента не трогает полный
    результат группы.
    """

    root, out, target = ready

    tokenized = tmp_path / "tokenized"

    common = (
        "encode", "--raw-root", str(root), "--name", "tok",
        "--processed", str(out), "--out", str(target), "--group", "train",
    )

    assert _cli(*common, "--tokenized", str(tokenized / "full")) == EXIT_OK

    full = json.loads((tokenized / "full" / MANIFEST_FILE).read_text(encoding="utf-8"))

    assert full["rows"]["clients"] == 2

    # Тот же прогон без --tokenized уходит в свой каталог по
    # имени группы и среза.
    assert _cli(*common, "--client", "train_c1", "--tokenized", str(tokenized / "one")) == EXIT_OK

    one = json.loads((tokenized / "one" / MANIFEST_FILE).read_text(encoding="utf-8"))

    assert one["rows"]["clients"] == 1

    # Полный результат на месте и не изменился.
    assert json.loads((tokenized / "full" / MANIFEST_FILE).read_text(encoding="utf-8")) == full


def test_repeated_run_into_the_same_directory_needs_force(ready, tmp_path, capsys):

    root, out, target = ready

    common = (
        "encode", "--raw-root", str(root), "--name", "tok",
        "--processed", str(out), "--out", str(target), "--group", "train",
        "--tokenized", str(tmp_path / "out"),
    )

    assert _cli(*common) == EXIT_OK

    capsys.readouterr()

    assert _cli(*common) == EXIT_BLOCKED
    assert "уже лежит результат" in capsys.readouterr().out

    assert _cli(*common, "--force") == EXIT_OK


# ============================================================
# ПРАВИЛА И МОМЕНТ СРЕЗА
# ============================================================


def test_encode_uses_the_frozen_configuration(ready, tmp_path, capsys):
    """
    Команда читает конфигурацию из замороженного комплекта, а
    чужую отвергает.
    """

    root, out, target = ready

    other = tmp_path / "other.json"

    payload = TokenizerConfig().as_dict()
    payload["max_pieces_per_value"] = 4

    write_json(other, payload)

    capsys.readouterr()

    code = _cli(
        "encode", "--raw-root", str(root), "--name", "tok",
        "--processed", str(out), "--out", str(target), "--group", "train",
        "--config", str(other), "--tokenized", str(tmp_path / "out"),
    )

    assert code == EXIT_BLOCKED
    assert "отличается от той, которой заморожен словарь" in capsys.readouterr().out


def test_edited_frozen_configuration_is_refused(ready, tmp_path, capsys):
    """
    Правленый tokenizer_config.json в каталоге артефактов не
    становится правилами кодирования.

    Ловит его проверка отпечатков: конфигурация входит в
    замороженный комплект наравне со словарями, и это правильное
    место — подмена замечается раньше, чем ею успеют что-то
    закодировать.
    """

    root, out, target = ready

    path = target / CONFIG_FILE

    original = path.read_bytes()

    payload = json.loads(original.decode("utf-8"))
    payload["max_pieces_per_value"] = 4
    write_json(path, payload)

    capsys.readouterr()

    try:
        code = _cli(
            "encode", "--raw-root", str(root), "--name", "tok",
            "--processed", str(out), "--out", str(target), "--group", "train",
            "--tokenized", str(tmp_path / "out"),
        )
    finally:
        path.write_bytes(original)

    assert code == EXIT_BLOCKED

    assert "tokenizer_config.json изменился после заморозки" in capsys.readouterr().out


def test_default_cutoff_comes_from_the_actual_split(ready, tmp_path):
    """
    Конечный cutoff берётся из фактического разделения, а не из
    значений по умолчанию: на нестандартном окне умолчание
    указало бы не на тот момент.
    """

    from src.tokenization.corpus import GroupCorpus

    root, out, target = ready

    corpus = GroupCorpus.open(out, root / "train", "train")

    assert corpus.final_cutoff == FIT_END
    assert group_cutoff(corpus, "train") == FIT_END

    # Разделения рядом нет — момент берётся из конфигурации, и
    # это видно в коде, а не угадывается.
    corpus.final_cutoff = None

    assert group_cutoff(corpus, "val") == datetime(2026, 6, 1)


def test_encode_writes_the_expected_files(ready, tmp_path):

    root, out, target = ready

    directory = tmp_path / "out"

    assert _cli(
        "encode", "--raw-root", str(root), "--name", "tok",
        "--processed", str(out), "--out", str(target), "--group", "train",
        "--tokenized", str(directory),
    ) == EXIT_OK

    assert (directory / EVENTS_FILE).exists()

    manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))

    assert manifest["cutoffs"] == [FIT_END.isoformat()]
