"""
Принадлежность события к сессии приложения.

Ревизия схемы RAW 2: операция и баннер, рождённые внутри
сессии, несут её настоящий session_id. Ничего не
восстанавливается по времени: у события либо известна
принадлежность, либо оно остаётся отдельным.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.generator.config import HISTORY_START, LABEL_END, SOURCE_AVAILABILITY
from src.generator.emit import SCHEMAS, schemas_for
from src.generator.version import RAW_SCHEMA_REVISION, REVISION_KEY
from src.preprocessing.config import KIND_METADATA, REGISTRY, payload_fields
from src.tokenizer.encode import EVENT_WIDTH


# Сессия это минуты, а не сутки. Границы намеренно щедрые:
# они ловят не дрейф на секунды, а перепутанную сессию.
MAX_SESSION_GAP_SECONDS = 3600.0
MAX_SESSION_SPAN_SECONDS = 7200.0

# Баннер показывается на экране витрины: 0-3 с после него,
# клик ещё через 2-25 с.
BANNER_AFTER_SCREEN_SECONDS = 30.0

LIFECYCLE_CLIENTS = range(12)


# ============================================================
# ФИКСТУРА: V2.1 ЧЕРЕЗ ВЕСЬ КОНВЕЙЕР
# ============================================================


@pytest.fixture(scope="session")
def v2_prep_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """
    RAW ревизии 2 -> preprocessing -> tokenizer во временный
    каталог. Каталог data не трогается.
    """

    from src.generator import emit
    from src.preprocessing.run import run as prep
    from src.tokenizer.run import run as tokenize

    root = tmp_path_factory.mktemp("v2_pipeline")

    raw = root / "raw"

    emit.generate_dataset(
        total_clients=100, chunk_clients=25, out_dir=raw, workers=1, version="v2.1"
    )

    processed = root / "processed"
    artifacts = root / "artifacts"

    prep(raw, "v2test", processed, artifacts, quiet=True)

    tokenized = root / "tokenized"
    vocab = root / "vocab"

    tokenize(
        processed_in=processed,
        artifacts_in=artifacts,
        out_dir=tokenized,
        vocab_out=vocab,
        quiet=True,
    )

    return {
        "raw": raw,
        "processed": processed,
        "artifacts": artifacts,
        "tokenized": tokenized,
        "vocab": vocab,
    }


# ============================================================
# 1. СОБЫТИЕ ЗНАЕТ СВОЮ СЕССИЮ, V1 НЕ ИЗМЕНИЛСЯ
# ============================================================


def test_v21_operations_and_banners_carry_the_session_id_and_v1_stays_frozen(
    raw_dir: Path, v2_raw_dir: Path, raw_tables: dict, v2_raw_tables: dict
):
    """
    Проверяется на результате lifecycle, до шума и до покрытия:
    там ещё видно, из какой сессии событие вышло.
    """

    from src.generator.chains import derive_lifecycle
    from src.generator.v2.lifecycle import derive_lifecycle_v2

    for client_id in LIFECYCLE_CLIENTS:

        result = derive_lifecycle_v2(client_id, HISTORY_START, LABEL_END)

        operations = result.app_operations
        banners = result.banners

        # --- принадлежность известна у каждого -------------
        assert all(event.session_id for event in operations)
        assert all(event.session_id for event in banners)

        # --- и это сессия, которая действительно была ------
        real = {run.session_id for run in result.runs}

        assert {event.session_id for event in operations} <= real
        assert {event.session_id for event in banners} <= real

        # --- события одной сессии это один эпизод ----------
        by_session: dict[str, list] = defaultdict(list)

        for event in (*result.app_screens, *operations, *banners):
            by_session[event.session_id].append(event.ts)

        for session_id, stamps in by_session.items():

            stamps = sorted(stamps)

            span = (stamps[-1] - stamps[0]).total_seconds()

            assert span <= MAX_SESSION_SPAN_SECONDS, (client_id, session_id, span)

            for earlier, later in zip(stamps, stamps[1:]):
                gap = (later - earlier).total_seconds()
                assert gap <= MAX_SESSION_GAP_SECONDS, (client_id, session_id, gap)

        # --- баннер стоит у экрана СВОЕЙ сессии ------------
        #
        # Самая острая проверка привязки: баннер порождается
        # экраном витрины и обязан идти сразу за экраном именно
        # той сессии, чей id он несёт.
        screens_of: dict[str, list] = defaultdict(list)

        for screen in result.app_screens:
            screens_of[screen.session_id].append(screen.ts)

        for banner in banners:
            assert any(
                0.0 <= (banner.ts - screen).total_seconds() <= BANNER_AFTER_SCREEN_SECONDS
                for screen in screens_of[banner.session_id]
            ), (client_id, banner.session_id, banner.ts)

        # --- V1 заморожена ---------------------------------
        v1 = derive_lifecycle(client_id, HISTORY_START, LABEL_END)

        assert all(event.session_id is None for event in v1.app_operations)
        assert all(event.session_id is None for event in v1.banners)

    # --- RAW ревизии 2 ------------------------------------
    for table in ("app_operations", "banners"):

        schema = schemas_for(RAW_SCHEMA_REVISION)[table]

        # Место колонки это и есть порядок ключей payload.
        assert schema.names[:2] == ["client_id", "ts"]
        assert schema.names[2] == "session_id"

        assert "session_id" not in SCHEMAS[table].names

        assert list(v2_raw_tables[table].columns) == schema.names
        assert v2_raw_tables[table].session_id.notna().all()

        assert "session_id" not in raw_tables[table].columns

    # Ключи payload ленты идут в порядке колонок.
    timeline = v2_raw_tables["timeline"]

    for event_type in ("app_operation", "banner"):

        rows = timeline[timeline.event_type == event_type]

        assert len(rows) > 0

        expected = list(payload_fields(event_type, RAW_SCHEMA_REVISION))

        for payload in rows.payload.head(50):
            assert list(json.loads(payload).keys()) == expected

    # --- манифесты ----------------------------------------
    v2_manifest = json.loads((v2_raw_dir / "manifest.json").read_text(encoding="utf-8"))
    v1_manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))

    assert v2_manifest[REVISION_KEY] == RAW_SCHEMA_REVISION
    assert REVISION_KEY not in v1_manifest


# ============================================================
# 2. РЕВИЗИЯ 2 ДОХОДИТ ДО РАСКЛАДКИ ИСТОРИИ
# ============================================================


def test_revision_2_flows_through_preprocessing_and_sidecar(
    v2_prep_run: dict, tok_run: dict, tmp_path: Path
):

    from src.model.session_batching import group_events
    from src.model.sessions import (
        APP_SCREEN,
        APP_TYPES,
        available_columns,
        build_sidecar,
        is_named,
        session_ids_of_block,
        session_key_of,
    )
    from src.tokenizer.build import client_runs

    source = v2_prep_run["processed"] / "clients" / "train_clients" / "events.parquet"

    columns = available_columns(source)

    # --- processed: metadata, а не признак -----------------
    assert set(columns) == set(APP_TYPES)

    names = set(pq.read_schema(source).names)

    for namespace in APP_TYPES:
        spec = REGISTRY[(namespace, "session_id")]
        assert spec.kind == KIND_METADATA
        assert spec.column in names

    # --- в токены session_id не попадает -------------------
    key_vocab = json.loads(
        (v2_prep_run["vocab"] / "key_vocab.json").read_text(encoding="utf-8")
    )

    keys_in_vocab = {entry["key"] for entry in key_vocab["keys"]}

    assert not any(name.endswith("__session_id") for name in keys_in_vocab)

    assert EVENT_WIDTH["app_operation"] == 2 + 3
    assert EVENT_WIDTH["banner"] == 2 + 3

    # --- sidecar -------------------------------------------
    #
    # Пишется в копию: sidecar входит в artifact_hashes, и
    # запись в общую фикстуру меняла бы окружение соседних
    # тестов в зависимости от порядка запуска.
    import shutil

    tokenized = tmp_path / "tokenized_v2"
    shutil.copytree(v2_prep_run["tokenized"], tokenized)

    report = build_sidecar(v2_prep_run["processed"], tokenized)

    summary = report["groups"]["train"]

    assert summary["named_by_type"]["app_operation"] > 0
    assert summary["named_by_type"]["banner"] > 0
    assert summary["named"] == sum(summary["named_by_type"].values())

    table = pq.read_table(
        source,
        columns=[
            "client_id",
            "seq",
            "ts",
            "event_type",
            *columns.values(),
            "app_operation__domain",
        ],
    )

    client_id = table.column("client_id").to_numpy()
    ts = table.column("ts").to_numpy().astype("datetime64[us]")
    event_type = np.asarray(table.column("event_type").to_pylist(), dtype=object)
    domain = np.asarray(table.column("app_operation__domain").to_pylist(), dtype=object)

    session_id = session_ids_of_block(table, columns, event_type)

    screens_from = np.datetime64(SOURCE_AVAILABILITY["app_screens"], "us")

    unexplained = 0
    without_screens = 0

    blocks = list(client_runs(client_id))

    for value, lo, hi in blocks:

        keys = session_key_of(session_id[lo:hi])
        kinds = event_type[lo:hi]

        # Ключ ровно там, где событие несёт свой session_id.
        assert np.array_equal(
            keys >= 0,
            np.array([is_named(value) for value in session_id[lo:hi]], dtype=bool),
        )

        assert not (keys >= 0)[~np.isin(kinds, APP_TYPES)].any()

        # Разные session_id не слились в один ключ.
        assert len({v for v in session_id[lo:hi] if is_named(v)}) == np.unique(
            keys[keys >= 0]
        ).size

        named = keys >= 0

        if not named.any():
            continue

        unique, inverse = np.unique(keys[named], return_inverse=True)
        inverse = np.asarray(inverse).ravel()

        screens = np.bincount(
            inverse,
            weights=(kinds[named] == APP_SCREEN).astype(np.float64),
            minlength=unique.size,
        )

        stamps = ts[lo:hi][named]
        domains = domain[lo:hi][named]

        for position in np.flatnonzero(screens == 0):

            without_screens += 1

            members = inverse == position

            # Сессия без экранов это либо период, когда экраны
            # ещё не сохранялись, либо несостоявшийся вход.
            if stamps[members].max() < screens_from:
                continue

            if all(item == "auth" for item in domains[members]):
                continue

            unexplained += 1

    assert without_screens > 0
    assert unexplained == 0

    # --- раскладка берёт операции и баннеры в сессию -------
    picked = next(
        (value, lo, hi)
        for value, lo, hi in blocks
        if (event_type[lo:hi] == "app_operation").any()
        and (event_type[lo:hi] == APP_SCREEN).any()
    )

    _, lo, hi = picked

    keys = session_key_of(session_id[lo:hi])
    kinds = event_type[lo:hi]
    own = session_id[lo:hi]

    layout = group_events(
        example_of_event=np.zeros(hi - lo, dtype=np.int64),
        ts=ts[lo:hi],
        seq=np.arange(hi - lo, dtype=np.int64),
        session_keys=keys,
        cutoffs=np.array([ts[lo:hi].max() + np.timedelta64(1, "D")]),
        n_examples=1,
    )

    # Событие в сессии тогда и только тогда, когда у него ключ.
    assert np.array_equal(layout.session_of_event >= 0, keys >= 0)

    # И операция стоит в той же сессии, что экраны её session_id.
    mixed = 0

    for position in range(layout.n_sessions):

        members = layout.member_rows[position][: layout.session_length[position]]

        assert len({own[row] for row in members}) == 1

        if {APP_SCREEN, "app_operation"} <= set(kinds[members]):
            mixed += 1

    assert mixed > 0

    # --- набор ревизии 1 читается как раньше ---------------
    legacy = tmp_path / "tokenized_v1"
    shutil.copytree(tok_run["tokenized"], legacy)

    v1 = build_sidecar(tok_run["processed"], legacy)

    assert v1["groups"]["train"]["named_by_type"]["app_operation"] == 0
    assert v1["groups"]["train"]["named_by_type"]["banner"] == 0
    assert v1["groups"]["train"]["named"] == v1["groups"]["train"]["screens"]
