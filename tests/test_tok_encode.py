from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.preprocessing.artifacts import sha256_file
from src.tokenization.corpus import GroupCorpus
from src.tokenization.encode import EncodeError, encode_event, encode_profile
from src.tokenization.layout import EMPTY, EVT, MISSING, UNK, USR, FrozenArtifacts
from src.tokenization.settings import TokenizerConfig
from src.tokenization.transform import (
    CLIENTS_FILE,
    EVENTS_FILE,
    GOLDEN_JSON_FILE,
    MANIFEST_FILE,
    PROFILES_FILE,
    REPORT_JSON_FILE,
    TransformError,
    transform_group,
)

from tests.tok_fixtures import FIT_END, build_vocab, prepared


# ============================================================
# ОБЩЕЕ
# ============================================================


CONFIG = TokenizerConfig()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> tuple[Path, Path]:
    return prepared(tmp_path_factory.mktemp("tok_encode"))


@pytest.fixture(scope="module")
def frozen(dataset, tmp_path_factory) -> tuple[Path, FrozenArtifacts]:

    root, out = dataset

    target = tmp_path_factory.mktemp("vocab_encode")

    build_vocab(root, out, target)

    return target, FrozenArtifacts.load(target)


@pytest.fixture(scope="module")
def train_history(dataset, frozen):

    root, out = dataset
    _target, _artifacts = frozen

    corpus = GroupCorpus.open(out, root / "train", "train")

    return corpus.history("train_c1", FIT_END)


def _first_id(record, key: str) -> int:
    """
    Первый код значения этого ключа.

    Обращаться к value_ids по номеру ключа нельзя: номер это
    место в списке значений, а коды лежат по позициям токенов, и
    у текста их несколько. Адрес даёт value_starts.
    """

    keys = record["value_keys"] if isinstance(record, dict) else record.value_keys
    starts = record["value_starts"] if isinstance(record, dict) else record.value_starts
    ids = record["value_ids"] if isinstance(record, dict) else record.value_ids

    return ids[starts[keys.index(key)]]


def _encoded(frozen, train_history, event_type: str):

    _target, artifacts = frozen

    event = next(
        item for item in train_history.events if item.values.get("event_type") == event_type
    )

    return artifacts, event, encode_event(artifacts, event, CONFIG.max_pieces_per_value)


# ============================================================
# ФОРМАТ ЗАПИСИ
# ============================================================


def test_arrays_agree_and_spans_cover_every_content_position(frozen, train_history):

    _target, artifacts = frozen

    for event in train_history.events:

        record = encode_event(artifacts, event, CONFIG.max_pieces_per_value)

        assert len(record.key_ids) == len(record.value_ids) == len(record.positions)
        assert len(record.value_starts) == len(record.value_lengths) == len(record.value_keys)

        # Содержимое начинается после маркера и покрыто целиком.
        covered = 1

        for start, length in zip(record.value_starts, record.value_lengths):
            assert start == covered
            covered += length

        assert covered == record.n_tokens

        # Номер это место куска ВНУТРИ значения.
        for start, length in zip(record.value_starts, record.value_lengths):
            assert record.positions[start : start + length] == list(range(length))


def test_marker_is_not_a_business_value(frozen, train_history):
    """
    Маркер занимает позицию в массивах модели, но значением не
    считается: своего span у него нет, в n_values он не входит и
    среди ключей значений не появляется.

    Иначе потребитель, который ходит по spans, однажды
    замаскировал бы служебную позицию как обычное поле.
    """

    _target, artifacts = frozen

    event = train_history.events[0]

    record = encode_event(artifacts, event, CONFIG.max_pieces_per_value)

    assert record.lead == EVT
    assert record.key_ids[0] == record.value_ids[0] == artifacts.special(EVT)
    assert record.positions[0] == 0

    assert EVT not in record.value_keys
    assert record.value_starts[0] == 1
    assert record.n_values == len(record.value_keys)

    # Токенов ровно столько, сколько заняли значения, плюс один
    # на маркер.
    assert record.n_tokens == sum(record.value_lengths) + 1

    profile = encode_profile(artifacts, train_history, CONFIG.max_pieces_per_value)

    assert profile.lead == USR
    assert USR not in profile.value_keys


def test_every_id_belongs_to_its_range_and_there_is_one_marker(frozen, train_history):

    target, artifacts = frozen

    size = artifacts.size

    for event in train_history.events:

        record = encode_event(artifacts, event, CONFIG.max_pieces_per_value)

        assert all(0 <= value < size for value in record.key_ids)
        assert all(0 <= value < size for value in record.value_ids)

        # Ровно один ведущий маркер, и он занимает оба слота.
        assert record.key_ids[0] == record.value_ids[0] == artifacts.special(EVT)
        assert record.key_ids.count(artifacts.special(EVT)) == 1
        assert artifacts.special(USR) not in record.value_ids

    profile = encode_profile(artifacts, train_history, CONFIG.max_pieces_per_value)

    assert profile.key_ids[0] == profile.value_ids[0] == artifacts.special(USR)
    assert profile.key_ids.count(artifacts.special(USR)) == 1


def test_declared_key_without_value_becomes_a_missing_pair(frozen, train_history):
    """
    У покупки причина отказа объявлена, но её нет: это пара
    ключ/[MISSING] с настоящим кодом ключа, а не пропуск позиции.
    """

    from src.tokenization.encode import absent_reasons

    artifacts, event, record = _encoded(frozen, train_history, "purchase")

    start = record.value_starts[record.value_keys.index("decline_reason")]

    assert record.value_ids[start] == artifacts.special(MISSING)
    assert record.key_ids[start] == artifacts.key_id("decline_reason")
    assert record.key_ids[start] != record.value_ids[start]

    # Расчётному ключу пара не выдаётся вовсе: его отсутствие
    # объясняет причина, а не специальный токен.
    reasons = absent_reasons(event, None)

    assert reasons

    for key in reasons:
        assert key not in record.value_keys


def test_reference_keys_never_reach_the_arrays(frozen, train_history):
    """
    Ссылки остаются связями и в embedding не входят.
    """

    artifacts, event, record = _encoded(frozen, train_history, "purchase")

    assert "account_ref" in event.values

    for key in artifacts.link_keys:
        assert key not in record.value_keys
        assert artifacts.key_id(key) is None


def test_text_is_not_truncated_and_pieces_share_one_key(frozen, train_history):
    """
    Текст занимает столько позиций, сколько кусков дал BPE, и все
    они принадлежат одному ключу.
    """

    artifacts, event, record = _encoded(frozen, train_history, "purchase")

    index = record.value_keys.index("merchant_name")

    start = record.value_starts[index]
    length = record.value_lengths[index]

    assert length > 1

    key_id = artifacts.key_id("merchant_name")

    assert record.key_ids[start : start + length] == [key_id] * length
    assert record.positions[start : start + length] == list(range(length))

    # Куски лежат в диапазоне BPE и собираются обратно в текст.
    assert all(value >= artifacts.bpe_offset for value in record.value_ids[start : start + length])

    decoded = artifacts.bpe.decode(
        [value - artifacts.bpe_offset for value in record.value_ids[start : start + length]]
    )

    assert decoded == event.values["merchant_name"].casefold()


def test_value_over_the_limit_is_an_error_not_a_cut(frozen, train_history):
    """
    Длинное значение не обрезается молча: это явная ошибка.
    """

    _target, artifacts = frozen

    event = next(
        item for item in train_history.events
        if item.values.get("event_type") == "purchase" and "merchant_name" in item.values
    )

    with pytest.raises(EncodeError, match="не обрезается"):
        encode_event(artifacts, event, limit=1)


def test_empty_text_is_not_a_missing_value(frozen, train_history):
    """
    Пустая строка это пришедшее поле без символов, а не
    отсутствие значения.
    """

    _target, artifacts = frozen

    values = {"event_type": "purchase", "merchant_name": "   "}

    from src.tokenization.encode import encode_values

    record = encode_values(artifacts, values, (), EVT, CONFIG.max_pieces_per_value)

    assert _first_id(record, "merchant_name") == artifacts.special(EMPTY)
    assert _first_id(record, "merchant_name") != artifacts.special(MISSING)


# ============================================================
# ВРЕМЯ И ТРАССИРОВКА
# ============================================================


def test_day_precision_event_keeps_its_uncertain_hour(dataset, frozen, tmp_path):
    """
    У записи дневной точности час суток не наблюдался, и признак
    об этом обязан доехать до потребителя.
    """

    root, out = dataset
    _target, artifacts = frozen

    corpus = GroupCorpus.open(out, root / "train", "train")

    transform_group(artifacts, corpus, [FIT_END], tmp_path / "out", CONFIG)

    rows = pq.read_table(tmp_path / "out" / EVENTS_FILE).to_pylist()

    daily = [row for row in rows if row["event_type"] == "installment_due"]

    assert daily
    assert all(row["hour_known"] is False for row in daily)
    assert all(row["time_precision"] == "day" for row in daily)

    # Календарь едет отдельным каналом и токеном не становится.
    assert all(len(row["calendar"]) == 6 for row in rows)


def test_provenance_names_the_record_not_the_position(dataset, frozen, tmp_path):
    """
    Происхождение ссылается на устойчивую пару «запись, версия»,
    а не на внутренний номер, который зависит от состава
    видимой истории.
    """

    root, out = dataset
    _target, artifacts = frozen

    corpus = GroupCorpus.open(out, root / "train", "train")

    transform_group(artifacts, corpus, [FIT_END], tmp_path / "out", CONFIG)

    rows = pq.read_table(tmp_path / "out" / EVENTS_FILE).to_pylist()

    found = False

    for row in rows:

        sources = json.loads(row["provenance"])

        for items in sources.values():
            for item in items:
                if item.get("kind") == "event":
                    assert "event_id" in item and "event_version" in item
                    assert "stable_event_index" not in item
                    found = True

    assert found


def test_correction_is_encoded_as_the_acting_version(dataset, frozen, tmp_path):
    """
    На срезе раньше события его нет вовсе, на срезе позже
    действует исправленная версия.
    """

    root, out = dataset
    _target, artifacts = frozen

    corpus = GroupCorpus.open(out, root / "train", "train")

    early = datetime(2025, 3, 1)
    late = datetime(2025, 4, 1)

    transform_group(artifacts, corpus, [early, late], tmp_path / "out", CONFIG)

    rows = pq.read_table(tmp_path / "out" / EVENTS_FILE).to_pylist()

    by_cutoff = {}

    for row in rows:
        by_cutoff.setdefault(row["cutoff"], []).append(row)

    corrected = [
        row for row in by_cutoff[late] if row["event_time"] == datetime(2025, 3, 10, 12, 0, 0)
    ]

    assert len(corrected) == 1
    assert corrected[0]["event_version"] == 2

    assert not [
        row for row in by_cutoff[early] if row["event_time"] == datetime(2025, 3, 10, 12, 0, 0)
    ]


def test_future_record_does_not_change_an_earlier_slice(tmp_path, frozen):
    """
    Запись после среза прежний срез не меняет.
    """

    _target, artifacts = frozen

    base_root, base_out = prepared(tmp_path / "base")
    later_root, later_out = prepared(tmp_path / "later", future=True)

    def encode(root: Path, out: Path, name: str) -> list[dict]:

        corpus = GroupCorpus.open(out, root / "train", "train")

        transform_group(artifacts, corpus, [FIT_END], tmp_path / name, CONFIG)

        return pq.read_table(tmp_path / name / EVENTS_FILE).to_pylist()

    assert encode(base_root, base_out, "base_out") == encode(later_root, later_out, "later_out")


# ============================================================
# ЗАМОРОЖЕННЫЙ TRANSFORM
# ============================================================


def test_new_value_becomes_unknown_without_touching_the_vocabulary(dataset, frozen, tmp_path):
    """
    Продукт, которого train не видел, кодируется как [UNK], а
    словарь при этом не расширяется.
    """

    root, out = dataset
    target, artifacts = frozen

    before = {name: sha256_file(target / name) for name in artifacts.manifest["artifacts"]}

    corpus = GroupCorpus.open(out, root / "val", "val")

    result = transform_group(
        artifacts, corpus, [datetime(2026, 6, 1)], tmp_path / "val_out", CONFIG
    )

    rows = pq.read_table(tmp_path / "val_out" / EVENTS_FILE).to_pylist()

    opened = next(row for row in rows if row["event_type"] == "product_opened")

    assert _first_id(opened, "product_code") == artifacts.special(UNK)

    # Семейство продукта известно и кодируется своим ключом.
    assert _first_id(opened, "product_family") != artifacts.special(UNK)

    assert result.report["specials"]["unknown"] > 0

    after = {name: sha256_file(target / name) for name in artifacts.manifest["artifacts"]}

    assert after == before


def test_silent_client_and_client_without_profile_are_kept(dataset, frozen, tmp_path):
    """
    Молчание это факт, а не причина потерять человека.
    """

    root, out = dataset
    _target, artifacts = frozen

    corpus = GroupCorpus.open(out, root / "train", "train")

    transform_group(artifacts, corpus, [FIT_END], tmp_path / "out", CONFIG)

    clients = pq.read_table(tmp_path / "out" / CLIENTS_FILE).to_pylist()

    assert {row["client_id"] for row in clients} == {"train_c1", "train_c2"}

    quiet = next(row for row in clients if row["client_id"] == "train_c2")

    assert quiet["n_events"] == 0
    assert quiet["has_profile"] is False

    profiles = pq.read_table(tmp_path / "out" / PROFILES_FILE).to_pylist()

    empty = next(row for row in profiles if row["client_id"] == "train_c2")

    # Представление профиля есть, но содержательных значений в
    # нём нет: только ведущий маркер, который значением не
    # считается.
    assert empty["profile_state"] == "no_version_known_yet"
    assert empty["n_values"] == 0
    assert empty["n_tokens"] == 1
    assert empty["value_keys"] == []


def test_totals_agree_with_the_per_client_rows(dataset, frozen, tmp_path):
    """
    Итог группы обязан сходиться с суммой по клиентам.

    Профиль это тоже закодированная запись: пока его значения
    попадали в строку клиента, но не в итог, две величины,
    описывающие одно и то же, расходились молча.
    """

    root, out = dataset
    _target, artifacts = frozen

    corpus = GroupCorpus.open(out, root / "train", "train")

    report = transform_group(artifacts, corpus, [FIT_END], tmp_path / "out", CONFIG).report

    clients = pq.read_table(tmp_path / "out" / CLIENTS_FILE).to_pylist()

    counts = report["counts"]

    assert counts["values"] == sum(row["n_values"] for row in clients)
    assert counts["tokens"] == sum(row["n_tokens"] for row in clients)

    # Итог складывается из событий и профилей, и обе доли видны.
    assert counts["values"] == counts["event_values"] + counts["profile_values"]
    assert counts["tokens"] == counts["event_tokens"] + counts["profile_tokens"]
    assert counts["profile_values"] > 0


def test_event_count_matches_the_semantic_history(dataset, frozen, tmp_path, train_history):

    root, out = dataset
    _target, artifacts = frozen

    corpus = GroupCorpus.open(out, root / "train", "train")

    report = transform_group(artifacts, corpus, [FIT_END], tmp_path / "out", CONFIG).report

    rows = pq.read_table(tmp_path / "out" / EVENTS_FILE).to_pylist()

    mine = [row for row in rows if row["client_id"] == "train_c1"]

    assert len(mine) == train_history.n_events
    assert report["counts"]["events"] == len(rows)


def test_cutoff_beyond_the_extract_is_reported_not_encoded(dataset, frozen, tmp_path):
    """
    Среза позже границы выгрузки не существует, и придумывать его
    нельзя.
    """

    root, out = dataset
    _target, artifacts = frozen

    corpus = GroupCorpus.open(out, root / "train", "train")

    report = transform_group(
        artifacts, corpus, [FIT_END, datetime(2030, 1, 1)], tmp_path / "out", CONFIG
    ).report

    assert [item["cutoff"][:10] for item in report["skipped_cutoffs"]] == ["2030-01-01"]
    assert report["cutoffs"] == [FIT_END.isoformat()]


def test_unknown_client_is_refused(dataset, frozen, tmp_path):

    root, out = dataset
    _target, artifacts = frozen

    corpus = GroupCorpus.open(out, root / "train", "train")

    with pytest.raises(TransformError, match="нет в группе"):
        transform_group(artifacts, corpus, [FIT_END], tmp_path / "out", CONFIG, clients=["val_c1"])


def test_another_configuration_cannot_encode_with_this_vocabulary(dataset, frozen, tmp_path):
    """
    Конфигурация это часть замороженного комплекта. Предел кусков
    в значении влияет на результат, поэтому кодировать одним
    словарём по правилам другого нельзя.
    """

    root, out = dataset
    _target, artifacts = frozen

    corpus = GroupCorpus.open(out, root / "train", "train")

    other = replace(CONFIG, max_pieces_per_value=4)

    with pytest.raises(TransformError, match="не та, которой заморожен словарь"):
        transform_group(artifacts, corpus, [FIT_END], tmp_path / "out", other)


def test_known_but_empty_profile_is_not_a_missing_profile(dataset, frozen, tmp_path):
    """
    Известная версия анкеты, у которой все поля пусты, это
    анкета: все её ключи получают [MISSING], а клиент не
    считается клиентом без профиля.
    """

    from src.tokenization.encode import profile_known

    root, out = dataset
    _target, artifacts = frozen

    corpus = GroupCorpus.open(out, root / "train", "train")

    history = corpus.history("train_c1", FIT_END)

    assert profile_known(history)

    # Анкета известна, но значений в ней не осталось.
    history.profile = {}

    record = encode_profile(artifacts, history, CONFIG.max_pieces_per_value)

    assert record.n_values == len(artifacts.profile_keys)
    assert set(record.value_ids[1:]) == {artifacts.special(MISSING)}

    # А у клиента без единой версии анкеты значений нет вовсе.
    silent = corpus.history("train_c2", FIT_END)

    assert not profile_known(silent)
    assert encode_profile(artifacts, silent, CONFIG.max_pieces_per_value).n_values == 0


def test_transform_writes_everything_the_plan_asks_for(dataset, frozen, tmp_path):

    root, out = dataset
    _target, artifacts = frozen

    corpus = GroupCorpus.open(out, root / "train", "train")

    result = transform_group(artifacts, corpus, [FIT_END], tmp_path / "out", CONFIG)

    for name in (EVENTS_FILE, PROFILES_FILE, CLIENTS_FILE, MANIFEST_FILE, REPORT_JSON_FILE,
                 GOLDEN_JSON_FILE):
        assert (tmp_path / "out" / name).exists()

    manifest = json.loads((tmp_path / "out" / MANIFEST_FILE).read_text(encoding="utf-8"))

    assert manifest["artifact_id"] == artifacts.manifest["artifact_id"]
    assert manifest["contract"]["marker_owner"] == "tokenizer"
    assert "12" not in manifest["contract"]["truncation"]

    golden = json.loads((tmp_path / "out" / GOLDEN_JSON_FILE).read_text(encoding="utf-8"))

    assert golden

    # Пример читается обратно: у каждой пары есть исходное
    # значение и расшифровка.
    for example in golden:
        for pair in example["pairs"]:
            assert "decoded" in pair
            assert "value_ids" in pair

    assert result.report["counts"]["clients"] == 2
    assert result.report["counts"]["client_slices"] == 2
