from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pytest

from src.preprocessing.artifacts import sha256_file, write_json, write_table
from src.tokenization.contract import TEXT_SCHEMA, STATISTICS_DIR, TEXT_FILE
from src.tokenization.layout import (
    EMPTY,
    EVT,
    INVALID,
    KEY_VOCAB_FILE,
    MANIFEST_FILE,
    MASK,
    MISSING,
    PAD,
    SPECIAL_TOKENS,
    VOCABULARY_FILES,
    UNK,
    USR,
    VALUE_VOCAB_FILE,
    FrozenArtifacts,
    LayoutError,
)
from src.tokenization.settings import BpeConfig, TokenizerConfig
from src.tokenization.text import PROBES, check_roundtrip, corpus_digest, corpus_rows, train_bpe

from tests.tok_fixtures import build_vocab, prepared


# ============================================================
# ОБЩЕЕ
# ============================================================


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> tuple[Path, Path]:
    return prepared(tmp_path_factory.mktemp("tok_vocab"))


@pytest.fixture(scope="module")
def frozen(dataset, tmp_path_factory) -> tuple[Path, dict]:

    root, out = dataset

    target = tmp_path_factory.mktemp("vocab_frozen")

    manifest = build_vocab(root, out, target).report

    return target, manifest


# ============================================================
# ПРОСТРАНСТВО ID
# ============================================================


def test_ranges_are_disjoint_and_cover_everything(frozen):
    """
    Специальные, ключи, значения и куски текста живут в одном
    пространстве непересекающимися отрезками.
    """

    _target, manifest = frozen

    layout = manifest["layout"]
    ranges = layout["ranges"]

    assert ranges["special"][0] == 0
    assert ranges["special"][1] == ranges["keys"][0]
    assert ranges["keys"][1] == ranges["values"][0]
    assert ranges["values"][1] == ranges["bpe"][0] == layout["bpe_offset"]
    assert ranges["bpe"][1] == layout["size"]

    sizes = layout["sizes"]

    assert sizes["special"] + sizes["keys"] + sizes["values"] + sizes["bpe"] == sizes["total"]
    assert sizes["categorical"] + sizes["buckets"] == sizes["values"]


def test_specials_keep_their_places(frozen):
    """
    Первые шесть номеров те же, что у прежнего словаря: менять их
    было бы дорого и бессмысленно.
    """

    target, _manifest = frozen

    artifacts = FrozenArtifacts.load(target)

    assert [artifacts.special(name) for name in (PAD, UNK, MASK, EVT, USR, MISSING)] == [0, 1, 2, 3, 4, 5]
    assert artifacts.special(INVALID) == 6
    assert artifacts.special(EMPTY) == 7

    assert SPECIAL_TOKENS[:6] == (PAD, UNK, MASK, EVT, USR, MISSING)


def test_keys_and_values_do_not_share_numbers(frozen):
    """
    Ключ и значение различаются по диапазону, поэтому служебная
    позиция узнаётся по равенству кода ключа и кода значения.
    """

    target, manifest = frozen

    artifacts = FrozenArtifacts.load(target)

    keys_range = manifest["layout"]["ranges"]["keys"]

    for key, key_id in artifacts.key_ids.items():
        assert keys_range[0] <= key_id < keys_range[1], key

    for row in artifacts.value_rows:
        assert row["id"] >= artifacts.first_value_id


def test_candidates_of_a_key_are_its_own_values(frozen):
    """
    Кандидаты ключа это значения, которые видели у него, а не
    весь домен: иначе поле предсказывалось бы значениями соседа.
    """

    target, _manifest = frozen

    artifacts = FrozenArtifacts.load(target)

    shared = artifacts.candidates["related_event_type"]["value_ids"]
    whole = artifacts.candidates["event_type"]["value_ids"]

    # У набора есть возврат по покупке, поэтому тип события-
    # причины наблюдался и его кандидаты не пусты.
    assert shared
    assert set(shared) < set(whole)

    # Диапазоны числового ключа идут подряд и по порядку.
    ids = artifacts.candidates["transaction_amount"]["value_ids"]

    assert ids == list(range(ids[0], ids[0] + len(ids)))


# ============================================================
# BPE
# ============================================================


def test_unicode_survives_encoding(frozen):
    """
    Байтовый алфавит обязан кодировать любой текст без потерь:
    казахские буквы, эмодзи, табуляцию и похожий на служебный
    токен текст.
    """

    target, _manifest = frozen

    artifacts = FrozenArtifacts.load(target)

    assert artifacts.bpe.enabled

    assert check_roundtrip(artifacts.bpe, list(PROBES)) == []

    # «[MASK]» из данных это обычный текст, а не управляющий токен.
    pieces = artifacts.bpe.pieces("[MASK]")

    assert len(pieces) > 1
    assert artifacts.bpe.decode(pieces) == "[MASK]"


def test_corpus_is_sorted_and_digested(frozen):
    """
    Порядок корпуса задан данными, а не файловой системой.
    """

    target, manifest = frozen

    rows = corpus_rows(target, tuple(manifest["bpe"]["keys"]))

    assert rows == sorted(rows)
    assert corpus_digest(rows) == manifest["bpe"]["corpus"]["sha256"]


def test_corpus_order_does_not_change_the_model(tmp_path):
    """
    Тот же корпус в другом порядке даёт то же разбиение.
    """

    from src.tokenization.text import _iterator

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    rows = [("k", f"магазин номер {index}", 2) for index in range(40)]

    def build(items):
        model = Tokenizer(models.BPE(unk_token=None))
        model.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        model.decoder = decoders.ByteLevel()
        model.train_from_iterator(
            _iterator(items),
            trainer=trainers.BpeTrainer(
                vocab_size=400,
                min_frequency=2,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                special_tokens=[],
                show_progress=False,
            ),
        )
        return model.to_str()

    assert build(rows) == build(list(reversed(rows)))


def test_without_texts_bpe_is_a_byte_alphabet(tmp_path):
    """
    Текстовые поля есть, а текстов на train нет: получается
    честный байтовый запасной вариант без единого слияния.
    """

    target = tmp_path / "vocab"

    write_table(
        target / STATISTICS_DIR / TEXT_FILE,
        pa.Table.from_pylist([], schema=TEXT_SCHEMA),
        TEXT_SCHEMA,
    )

    model = train_bpe(target, BpeConfig(), ("merchant_name",))

    assert model.enabled
    assert model.info["trained_on_empty"] is True
    assert model.size == 256
    assert check_roundtrip(model, ["Қазпошта", "a b", ""]) == []


def test_without_text_keys_bpe_is_off(tmp_path):
    """
    Текста ради BPE не придумывается.
    """

    model = train_bpe(tmp_path, BpeConfig(), ())

    assert model.enabled is False
    assert "не придумывается" in model.info["reason"]


# ============================================================
# ЗАМОРОЗКА
# ============================================================


def test_manifest_records_everything_the_vocabulary_depends_on(frozen):

    _target, manifest = frozen

    assert manifest["artifact_id"]
    assert manifest["fit_content_sha256"]
    assert manifest["config_sha256"]

    # Версии этапов препроцессинга, библиотек и самих модулей.
    assert manifest["versions"]["stages"]["semantic"]
    assert manifest["libraries"]["tokenizers"]
    assert "layout" in manifest["sources"]

    for name in (KEY_VOCAB_FILE, VALUE_VOCAB_FILE):
        assert name in manifest["artifacts"]


def test_substituted_artifact_id_is_refused_at_load(frozen):
    """
    Номер комплекта считается по всему манифесту и проверяется
    при загрузке.

    Подменённый номер при целых файлах прежде проходил молча, а
    это и есть подмена тождества словаря: будущий датасет и
    checkpoint ссылаются именно на него.
    """

    target, _manifest = frozen

    path = target / MANIFEST_FILE

    original = path.read_bytes()

    payload = json.loads(original.decode("utf-8"))
    payload["artifact_id"] = "000000000000"
    write_json(path, payload)

    try:
        with pytest.raises(LayoutError, match="манифест словаря изменён"):
            FrozenArtifacts.load(target)
    finally:
        path.write_bytes(original)

    FrozenArtifacts.load(target)


def test_edited_manifest_field_is_refused_at_load(frozen):
    """
    Правка любого поля манифеста, а не только списка файлов,
    ломает тождество комплекта.
    """

    target, _manifest = frozen

    path = target / MANIFEST_FILE

    original = path.read_bytes()

    payload = json.loads(original.decode("utf-8"))
    payload["format_version"] = "9.9.9"
    write_json(path, payload)

    try:
        with pytest.raises(LayoutError, match="манифест словаря изменён"):
            FrozenArtifacts.load(target)
    finally:
        path.write_bytes(original)


def test_edited_artifact_is_refused_at_load(frozen, tmp_path):
    """
    Словарь после заморозки неизменен, и проверяется это по
    файлам, а не по обещанию.
    """

    target, _manifest = frozen

    path = target / VALUE_VOCAB_FILE

    original = path.read_bytes()

    payload = json.loads(original.decode("utf-8"))
    payload["values"][0]["train_count"] += 1
    write_json(path, payload)

    try:
        with pytest.raises(LayoutError, match="изменился после заморозки"):
            FrozenArtifacts.load(target)
    finally:
        path.write_bytes(original)

    FrozenArtifacts.load(target)


def test_freeze_is_reproducible(dataset, tmp_path):

    root, out = dataset

    first = build_vocab(root, out, tmp_path / "one").report
    second = build_vocab(root, out, tmp_path / "two").report

    assert first["artifact_id"] == second["artifact_id"]
    assert first["artifacts"] == second["artifacts"]


def test_vocabulary_is_byte_identical_in_another_process(dataset, tmp_path):
    """
    Другой процесс с другим PYTHONHASHSEED собирает тот же
    словарь до байта.
    """

    root, out = dataset

    here = build_vocab(root, out, tmp_path / "here").report

    script = (
        "from pathlib import Path;"
        "from tests.tok_fixtures import build_vocab;"
        f"build_vocab(Path({str(root)!r}), Path({str(out)!r}), Path({str(tmp_path / 'there')!r}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONHASHSEED": "24680", "PYTHONPATH": str(Path.cwd())},
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")

    there = json.loads((tmp_path / "there" / MANIFEST_FILE).read_text(encoding="utf-8"))

    assert there["artifact_id"] == here["artifact_id"]

    for name, digest in here["artifacts"].items():
        assert sha256_file(tmp_path / "there" / name) == digest, name


def test_other_groups_do_not_reach_the_vocabulary(tmp_path):
    """
    Замена validation и test другими клиентами и запись после
    fit_end словарь не меняют.

    Сравнивается отпечаток самого словаря. artifact_id при этом
    законно отличается: он включает происхождение входов, а
    манифест разделения описывает все три группы и вместе с ними
    изменился. Смешивать эти два вопроса нельзя.
    """

    base_root, base_out = prepared(tmp_path / "base")
    other_root, other_out = prepared(tmp_path / "other", future=True, variant=1)

    base = build_vocab(base_root, base_out, tmp_path / "base_vocab").report
    other = build_vocab(other_root, other_out, tmp_path / "other_vocab").report

    assert other["vocab_sha256"] == base["vocab_sha256"]
    assert other["fit_content_sha256"] == base["fit_content_sha256"]

    for name in sorted(base["artifacts"]):
        if name in VOCABULARY_FILES:
            assert sha256_file(tmp_path / "other_vocab" / name) == base["artifacts"][name], name
