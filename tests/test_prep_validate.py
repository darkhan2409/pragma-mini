"""
Проверка RAW: находит подмену, а не просто проходит.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from src.preprocessing.raw import RawDataset
from src.preprocessing.validate import ValidationError, validate_raw


# ============================================================
# ПОДМЕНА RAW
# ============================================================


def copy_raw(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


def rewrite(raw_dir: Path, name: str, table: pa.Table, patch_manifest: bool = True) -> None:
    """
    Перезаписывает таблицу и, если нужно, счётчик в манифесте.
    """

    pq.write_table(table, raw_dir / f"{name}.parquet", compression="zstd")

    if not patch_manifest:
        return

    path = raw_dir / "manifest.json"

    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["rows"][name] = table.num_rows

    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def failing_checks(raw_dir: Path) -> set[str]:

    with pytest.raises(ValidationError) as error:
        validate_raw(RawDataset(raw_dir))

    return {
        check["name"]
        for check in error.value.report["checks"]
        if check["status"] == "failed"
    }


def set_column(table: pa.Table, name: str, column) -> pa.Table:
    return table.set_column(table.schema.get_field_index(name), name, column)


# ============================================================
# ЧИСТЫЙ RAW
# ============================================================


def test_clean_raw_passes(prep_raw_dir):
    report = validate_raw(RawDataset(prep_raw_dir))

    assert report["status"] == "ok"
    assert all(check["status"] != "failed" for check in report["checks"])


def test_report_lists_expected_checks(prep_raw_dir):
    report = validate_raw(RawDataset(prep_raw_dir))

    names = {check["name"] for check in report["checks"]}

    assert {
        "schemas_match",
        "manifest_rows_match",
        "latent_columns_absent",
        "latent_payload_keys_absent",
        "timestamps_before_feature_end",
        "availability_respected",
        "first_seen_respected",
        "coverage_rows_complete",
        "profile_monthly_grid",
        "timestamp_quality_consistent",
        "timeline_sorted_and_dense",
        "timeline_tie_break_priority",
        "payload_keys_match_contract",
        "payload_parses",
        "timeline_matches_tables",
        "labels_shape",
    } <= names


def test_summary_reports_defects_without_fixing_them(prep_raw_dir):
    summary = validate_raw(RawDataset(prep_raw_dir))["summary"]

    assert summary["equal_ts"]["events_with_shared_ts_share"] > 0
    assert summary["history_length"]["clients_over_limit"] == 0
    assert set(summary["exact_duplicate_rows"]) >= {"transactions", "app_screens"}
    assert summary["source_coverage_share"]["transactions"] == 1.0
    assert summary["source_coverage_share"]["app_screens"] < 1.0
    assert summary["profile_missing"]["group_all_null_share"]["employment"] > 0


# ============================================================
# ПОДСУНУТЫЕ ДЕФЕКТЫ
# ============================================================


def test_latent_column_is_caught(prep_raw_dir, tmp_path):
    raw_dir = copy_raw(prep_raw_dir, tmp_path / "latent")

    table = pq.read_table(raw_dir / "transactions.parquet")

    leaked = table.append_column("credit_need", pa.array(np.zeros(table.num_rows), pa.float64()))

    rewrite(raw_dir, "transactions", leaked)

    assert {"schemas_match", "latent_columns_absent"} <= failing_checks(raw_dir)


def test_latent_payload_key_is_caught(prep_raw_dir, tmp_path):
    raw_dir = copy_raw(prep_raw_dir, tmp_path / "latent_payload")

    table = pq.read_table(raw_dir / "timeline.parquet")

    payload = table.column("payload").to_pylist()
    payload[0] = '{"amount":100,"credit_need":0.5}'

    rewrite(raw_dir, "timeline", set_column(table, "payload", pa.array(payload, pa.string())))

    failures = failing_checks(raw_dir)

    # Скрытый ключ виден и как совпадение регэкспа, и как
    # лишнее поле при разборе с явной схемой.
    assert "latent_payload_keys_absent" in failures
    assert "payload_parses" in failures


def test_duplicated_seq_is_caught(prep_raw_dir, tmp_path):
    raw_dir = copy_raw(prep_raw_dir, tmp_path / "seq")

    table = pq.read_table(raw_dir / "timeline.parquet")

    seq = table.column("seq").to_pylist()
    seq[1] = seq[0]

    rewrite(raw_dir, "timeline", set_column(table, "seq", pa.array(seq, pa.int64())))

    assert "timeline_sorted_and_dense" in failing_checks(raw_dir)


def test_non_monotonic_ts_is_caught(prep_raw_dir, tmp_path):
    raw_dir = copy_raw(prep_raw_dir, tmp_path / "order")

    table = pq.read_table(raw_dir / "timeline.parquet")

    ts = table.column("ts").to_pylist()
    ts[5] = datetime(2024, 1, 1)

    rewrite(raw_dir, "timeline", set_column(table, "ts", pa.array(ts, pa.timestamp("us"))))

    failures = failing_checks(raw_dir)

    assert "timeline_sorted_and_dense" in failures
    assert "timeline_matches_tables" in failures


def test_future_event_is_caught(prep_raw_dir, tmp_path):
    raw_dir = copy_raw(prep_raw_dir, tmp_path / "future")

    table = pq.read_table(raw_dir / "transactions.parquet")

    ts = table.column("ts").to_pylist()
    ts[0] = datetime(2026, 7, 1)

    rewrite(raw_dir, "transactions", set_column(table, "ts", pa.array(ts, pa.timestamp("us"))))

    assert "timestamps_before_feature_end" in failing_checks(raw_dir)


def test_table_diverging_from_timeline_is_caught(prep_raw_dir, tmp_path):
    """
    Регресс на класс ошибок «шум применён только к таблице».
    """

    raw_dir = copy_raw(prep_raw_dir, tmp_path / "diverge")

    table = pq.read_table(raw_dir / "transactions.parquet")

    amount = table.column("amount").to_pylist()
    amount[0] = amount[0] + 1

    rewrite(raw_dir, "transactions", set_column(table, "amount", pa.array(amount, pa.int64())))

    assert "timeline_matches_tables" in failing_checks(raw_dir)


def test_date_only_with_time_is_caught(prep_raw_dir, tmp_path):
    raw_dir = copy_raw(prep_raw_dir, tmp_path / "quality")

    table = pq.read_table(raw_dir / "product_events.parquet")

    quality = table.column("timestamp_quality").to_pylist()
    ts = table.column("ts").to_pylist()

    position = quality.index("date_only")
    ts[position] = ts[position].replace(hour=13)

    rewrite(raw_dir, "product_events", set_column(table, "ts", pa.array(ts, pa.timestamp("us"))))

    assert "timestamp_quality_consistent" in failing_checks(raw_dir)


def test_event_before_first_seen_is_caught(prep_raw_dir, tmp_path):
    raw_dir = copy_raw(prep_raw_dir, tmp_path / "coverage")

    coverage = pq.read_table(raw_dir / "source_coverage.parquet")

    mask = pc.equal(coverage.column("source"), "app_screens").to_pylist()
    first_seen = coverage.column("first_seen").to_pylist()

    moved = 0

    for index, is_screens in enumerate(mask):
        if is_screens and first_seen[index] is not None:
            first_seen[index] = datetime(2026, 5, 1)
            moved += 1
            if moved > 3:
                break

    rewrite(raw_dir, "source_coverage", set_column(coverage, "first_seen", pa.array(first_seen, pa.timestamp("us"))))

    assert "first_seen_respected" in failing_checks(raw_dir)


def test_manifest_row_count_mismatch_is_caught(prep_raw_dir, tmp_path):
    raw_dir = copy_raw(prep_raw_dir, tmp_path / "manifest")

    path = raw_dir / "manifest.json"

    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["rows"]["transactions"] += 1

    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    assert "manifest_rows_match" in failing_checks(raw_dir)


def test_broken_schema_is_caught(prep_raw_dir, tmp_path):
    raw_dir = copy_raw(prep_raw_dir, tmp_path / "schema")

    table = pq.read_table(raw_dir / "banners.parquet")

    rewrite(raw_dir, "banners", table.drop_columns(["offer"]))

    assert "schemas_match" in failing_checks(raw_dir)


def test_label_window_change_is_caught(prep_raw_dir, tmp_path):
    raw_dir = copy_raw(prep_raw_dir, tmp_path / "labels")

    table = pq.read_table(raw_dir / "labels.parquet")

    start = table.column("label_start").to_pylist()
    start[0] = datetime(2025, 1, 1)

    rewrite(raw_dir, "labels", set_column(table, "label_start", pa.array(start, pa.timestamp("us"))))

    assert "labels_shape" in failing_checks(raw_dir)


def test_missing_manifest_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        RawDataset(tmp_path)


# ============================================================
# ПАМЯТЬ НЕ РАСТЁТ С РАЗМЕРОМ ДАТАСЕТА
# ============================================================


def test_scan_keeps_no_tables():
    """
    Регресс: раньше проход копил разобранные payload всех
    событий, и пик памяти рос линейно (86 МБ на миллион).
    Результат прохода обязан быть только счётчиками.
    """

    from dataclasses import fields

    from src.preprocessing.validate import TimelineScan

    for spec in fields(TimelineScan):
        assert "pa." not in str(spec.type), spec.name
        assert "Table" not in str(spec.type), spec.name
        assert "ndarray" not in str(spec.type), spec.name


def test_scan_releases_memory(prep_raw_dir):
    """
    После прохода в пуле arrow не должно остаться удерживаемых
    данных: всё, что читалось, освобождается по ходу.
    """

    import pyarrow as pa

    from src.preprocessing.validate import scan_timeline

    pool = pa.default_memory_pool()

    before = pool.bytes_allocated()

    scan = scan_timeline(RawDataset(prep_raw_dir))

    retained = pool.bytes_allocated() - before

    assert scan.rows > 100_000
    assert retained < 8 << 20, f"проход удерживает {retained / (1 << 20):.1f} МБ"
