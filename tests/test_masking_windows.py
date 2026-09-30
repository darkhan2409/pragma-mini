from __future__ import annotations

import json
from datetime import datetime, timedelta

import pyarrow.parquet as pq
import pytest

from tests.test_profile_state import (
    BANK,
    QUIET_SNAPSHOT,
    RAW_CLIENT,
    prepare,
    raw_event,
    when,
    write_profile_vocab,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# У группы два окна, оба [начало, конец), границы — полночь банка:
#
#   группа  контекст                   маскирование
#   train   [2024-01-01, 2026-01-01)   [2024-01-01, 2026-01-01)
#   val     [2024-01-01, 2026-05-01)   [2026-01-01, 2026-05-01)
#   test    [2024-01-01, 2026-09-01)   [2026-05-01, 2026-09-01)
#
# Проверяется:
#
#   границы     начало окна маскирования — уже цель, миг до него —
#               ещё нет; конец — уже нет, миг до него — ещё да;
#   контекст    событие вне окна маскирования остаётся в примере,
#               но целью не становится; источник вехи и изменение
#               анкеты не цели и внутри окна;
#   маски       маска val фиксирована и повторяется, метки
#               только внутри окна;
#   старое      набор и клеймо без окон или с другими окнами
#               отвергаются.
#
# Ожидаемые окна написаны руками.
# ============================================================


MICROSECOND = timedelta(microseconds=1)


def local(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=BANK)


EXPECTED = {
    "train": ((local(2024, 1, 1), local(2026, 1, 1)), (local(2024, 1, 1), local(2026, 1, 1))),
    "val": ((local(2024, 1, 1), local(2026, 5, 1)), (local(2026, 1, 1), local(2026, 5, 1))),
    "test": ((local(2024, 1, 1), local(2026, 9, 1)), (local(2026, 5, 1), local(2026, 9, 1))),
}

GROUPS = tuple(EXPECTED)


def window(group: str):

    from src.preprocessing.settings import PreprocessingConfig

    return PreprocessingConfig.load(None).windows[group]


# ============================================================
# ОКНА
# ============================================================


@pytest.mark.parametrize("group", GROUPS)
def test_windows_are_the_declared_ones(group: str):

    found = window(group)

    (context, masking) = EXPECTED[group]

    assert (found.history_start, found.final_cutoff) == context
    assert (found.target_start, found.target_end) == masking


@pytest.mark.parametrize("group", GROUPS)
def test_context_is_the_whole_export_of_the_group(group: str):
    """
    Контекст — вся выгрузка группы: от её начала до её конца в
    DATASETS генератора.
    """

    from src.generator.config import DATASETS

    found = window(group)
    export = DATASETS[group]

    assert found.history_start == export.history_start.replace(tzinfo=BANK)
    assert found.final_cutoff == export.history_end.replace(tzinfo=BANK)


@pytest.mark.parametrize("group", GROUPS)
def test_masking_window_includes_its_start(group: str):

    from src.dataset.targets import eligible

    found = window(group)

    assert eligible(found.target_start, found)
    assert not eligible(found.target_start - MICROSECOND, found)


@pytest.mark.parametrize("group", GROUPS)
def test_masking_window_excludes_its_end(group: str):

    from src.dataset.targets import eligible

    found = window(group)

    assert eligible(found.target_end - MICROSECOND, found)
    assert not eligible(found.target_end, found)


def test_windows_must_nest():
    """
    Окно маскирования внутри контекста и не пусто: иначе конфиг
    не собирается.
    """

    from src.preprocessing.settings import GroupWindow

    start, cutoff = local(2024, 1, 1), local(2026, 4, 1)

    for target_start, target_end in (
        (local(2023, 12, 1), cutoff),          # раньше контекста
        (local(2026, 1, 1), local(2026, 5, 1)),  # позже cutoff
        (local(2026, 1, 1), local(2026, 1, 1)),  # пустое
    ):
        with pytest.raises(ValueError, match="не вложены"):
            GroupWindow(start, cutoff, target_start, target_end)


# ============================================================
# ПРИМЕР: КОНТЕКСТ ШИРЕ ОКНА МАСКИРОВАНИЯ
# ============================================================


@pytest.mark.parametrize("group", ["val", "test"])
def test_outside_the_window_is_context_not_a_target(stage, group: str):
    """
    Миг до начала окна — контекст. Начало окна и миг до его конца —
    цели. Источник вехи и изменение анкеты в самом начале окна —
    контекст. Все события остаются в примере.
    """

    from src.dataset.sample import build_sample
    from src.dataset.settings import ContextPolicy
    from src.dataset.tokenized import TokenizedClient, TokenizedEvent
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.specials import EVT, USR

    write_profile_vocab(stage)

    artifacts = FrozenArtifacts.load()
    evt, usr = artifacts.special(EVT), artifacts.special(USR)

    found = window(group)
    start, end = found.target_start, found.target_end

    def event(moment: datetime, kind: str, source: str | None = None) -> TokenizedEvent:
        return TokenizedEvent(
            event_time=moment, event_type=kind, key_ids=[evt], value_ids=[evt],
            positions=[0], calendar=[0.0] * 6, lifelong_source=source,
        )

    events = [
        event(local(2024, 3, 1), "purchase"),
        event(start - MICROSECOND, "purchase"),
        event(start, "purchase"),
        event(start, "card_activated", source="first_card_activated"),
        event(start, "profile_change"),
        event(end - MICROSECOND, "purchase"),
    ]

    client = TokenizedClient(
        client_id=RAW_CLIENT, events=events,
        profile_key_ids=[usr], profile_value_ids=[usr], profile_positions=[0], profile_time=[None],
    )

    sample = build_sample(artifacts, client, found, ContextPolicy())

    assert sample.n_events == len(events)
    assert sample.target_event_mask.tolist() == [False, False, True, False, False, True]


# ============================================================
# 02 → 05 → ЧТЕНИЕ: ФИКСИРОВАННАЯ МАСКА VAL
# ============================================================


def test_validation_labels_only_inside_its_window_and_repeat(stage):
    """
    Выгрузка → 02 → 04 → 05 для val, маска при чтении. Маскер
    закрывает каждое допустимое событие: метки есть только у событий
    окна [1 января, 1 апреля), кроме источника вехи. Событие в миг
    cutoff до примера не доходит вовсе. Маска с одним конфигом
    повторяется от чтения к чтению.
    """

    from src.dataset.build import build_group as build_dataset
    from src.dataset.settings import DatasetConfig
    from src.masking.apply import IGNORE
    from src.masking.settings import MaskingConfig
    from src.mlm.inputs import Source
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.settings import TokenizerConfig
    from src.tokenization.transform import encode_group

    write_profile_vocab(stage, event_types=("purchase", "card_activated"))

    def purchase(moment: str) -> dict:
        return raw_event(RAW_CLIENT, moment, {
            "type": "purchase", "amount": 700, "direction": "debit", "status": "approved"})

    # Окно val в UTC: [2025-12-31 19:00, 2026-04-30 19:00).
    tape = [
        purchase("2024-03-01T09:00:00"),
        purchase("2025-12-31T18:59:59"),
        purchase("2025-12-31T19:00:00"),
        raw_event(RAW_CLIENT, "2025-12-31T19:00:00", {
            "type": "card_activated", "product_id": "prd_card", "card_id": "crd_1"}),
        purchase("2026-04-30T18:59:59"),
        purchase("2026-04-30T19:00:00"),
    ]

    milestones = [
        {"type": "first_card_activated", "event_time": when("2025-12-31T19:00:00"), "source_id": "crd_1"},
    ]

    history = prepare(stage, tape, dict(QUIET_SNAPSHOT, lifelong=milestones))

    found = window("val")

    expected = [
        found.target_start <= item.event_time < found.target_end and item.lifelong_source is None
        for item in history.events
    ]

    assert len(history.events) == len(tape) - 1
    assert expected.count(True) == 2

    artifacts = FrozenArtifacts.load()

    encode_group(artifacts, "val", TokenizerConfig.load(None))
    build_dataset(artifacts, "val", DatasetConfig.load(None))

    every_event = MaskingConfig(
        event_probability=1.0, value_probability=0.0, key_probability=0.0, unknown_probability=0.0
    )

    (client,) = list(Source("val", masking=every_event).clients())

    labelled = [
        bool((client.labels[start:start + length] != IGNORE).any())
        for start, length in zip(client.event_starts, client.event_lengths)
    ]

    assert labelled == expected

    # Маска val разыгрывается при чтении, и два чтения с обычным
    # конфигом дают одну и ту же маску.
    def masks() -> list[tuple]:
        return [
            (item.client_id, item.value_ids.tolist(), item.labels.tolist(), item.reason)
            for item in Source("val", masking=MaskingConfig.load(None)).clients()
        ]

    assert masks() == masks()


# ============================================================
# СТАРЫЕ АРТЕФАКТЫ
# ============================================================


def test_samples_without_their_window_are_refused(stage):
    """
    Набор 05 без окна или с окном прежней сборки (val до 1 мая)
    читатель набора не принимает.
    """

    from src.dataset.build import SAMPLES_SCHEMA
    from src.dataset.lineage import lineage
    from src.dataset.settings import dataset_dir
    from src.temporal.samples import SamplesError, SamplesGroup

    directory = dataset_dir("val")
    directory.mkdir(parents=True, exist_ok=True)

    pq.write_table(SAMPLES_SCHEMA.empty_table(), directory / "samples.parquet")

    current = lineage()

    meta = {
        "format": current["dataset_format"],
        "profile_semantics": current["profile_semantics"],
        "profile_lifelong_types": current["profile_lifelong_types"],
        "events_cutoff": window("val").final_cutoff.isoformat(),
    }

    stale = dict(window("val").as_dict(), final_cutoff=local(2026, 4, 1).isoformat(),
                 target_end=local(2026, 4, 1).isoformat())

    for broken in (meta, dict(meta, window=stale), dict(meta, window=window("test").as_dict())):

        (directory / "meta.json").write_text(json.dumps(broken), encoding="utf-8")

        with pytest.raises(SamplesError, match="окна"):
            SamplesGroup("val")

    (directory / "meta.json").write_text(json.dumps(dict(meta, window=window("val").as_dict())),
                                         encoding="utf-8")

    SamplesGroup("val")


def test_stamp_without_windows_is_refused(stage):
    """
    Клеймо этапов 06 и 09 прежнего кода — без окон или с другими
    окнами — отвергается, текущее принимается.
    """

    from src.dataset.lineage import LINEAGE_FILE, lineage, lineage_problem, write_lineage

    directory = stage / "stamped"
    directory.mkdir()

    current = lineage()

    without = {key: value for key, value in current.items() if key != "windows"}

    moved = dict(current, windows=dict(current["windows"], val=dict(
        current["windows"]["val"], target_end=local(2026, 4, 1).isoformat())))

    for stamp in (without, moved):

        (directory / LINEAGE_FILE).write_text(json.dumps(stamp), encoding="utf-8")

        assert lineage_problem(directory, "пересобрать") is not None

    write_lineage(directory)

    assert lineage_problem(directory, "пересобрать") is None
