from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.generator.emit import COVERAGE_SCHEMA as GEN_COVERAGE_SCHEMA
from src.generator.emit import EVENTS_SCHEMA as GEN_EVENTS_SCHEMA
from src.generator.emit import ContentDigest as GeneratorDigest
from src.preprocessing import manifest as manifest_module
from src.preprocessing import passport as passport_module
from src.preprocessing import rawdata as rawdata_module
from src.preprocessing import run as run_module
from src.preprocessing import settings as settings_module
from src.preprocessing.artifacts import dumps_json
from src.preprocessing.manifest import fingerprint_path, load_fingerprint
from src.preprocessing.passport import (
    STATUS_BLOCKED,
    STATUS_CONTRACT_MISMATCH,
    STATUS_HORIZON_SHORT,
    STATUS_READY,
    build_passport,
    month_range,
    render_passport_md,
)
from src.preprocessing.rawdata import (
    COVERAGE_SCHEMA,
    ENVELOPE_SCHEMA,
    ContentDigest,
    RawDataset,
    parse_payloads,
)
from src.preprocessing.run import EXIT_BLOCKED, EXIT_CONTRACT_MISMATCH, EXIT_OK
from src.preprocessing.run import main as run_main
from src.preprocessing.settings import PreprocessingConfig

from tests.prep_fixtures import MiniRaw, check_raw, purchase_payload, tiny_raw  # noqa: F401


CONFIG = PreprocessingConfig()

FULL_HORIZON = datetime(2023, 1, 1)

# Модули, из которых состоит этап: правка любого меняет его поведение.
STAGE_MODULES = {
    "passport": passport_module,
    "rawdata": rawdata_module,
    "settings": settings_module,
    "manifest": manifest_module,
    "run": run_module,
}


def _cli(*args: str) -> int:
    with pytest.raises(SystemExit) as result:
        run_main(list(args))
    return int(result.value.code)


def _corrupt(path: Path) -> None:
    """
    Портит один байт в середине файла, сохраняя размер.
    """

    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))


# ============================================================
# КОНТРАКТ
# ============================================================


def test_envelope_and_coverage_schemas_match_generator():
    assert ENVELOPE_SCHEMA.equals(GEN_EVENTS_SCHEMA)
    assert COVERAGE_SCHEMA.equals(GEN_COVERAGE_SCHEMA)


def test_content_digest_matches_generator():

    rows = [
        {"a": 1, "b": "x", "t": datetime(2024, 6, 1, 12, 0)},
        {"a": None, "b": "y", "t": None},
    ]

    ours = ContentDigest()
    ours.extend(rows)

    theirs = GeneratorDigest()
    theirs.extend(rows)

    assert ours.value() == theirs.value()


def test_non_contiguous_client_is_blocked(tmp_path):
    """
    Строки клиента обязаны лежать подряд.

    Весь препроцессинг читает ленту пачками целых клиентов.
    Клиент, появившийся второй раз после другого, потерял бы
    часть истории молча: второй блок просто перезаписал бы
    первый в индексе.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")
    mini.cover_all("c2", first_seen="2023-01-01")

    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload())
    mini.event("c2", "purchase", "2023-03-06 10:00:00", payload=purchase_payload())
    mini.event("c1", "purchase", "2023-03-07 10:00:00", payload=purchase_payload())

    report = build_passport(mini.write(), CONFIG, "train")

    assert report["status"] == STATUS_BLOCKED
    assert report["usable"] is False
    assert any("не подряд" in item and "c1" in item for item in report["errors"])


def test_corrupted_manifest_is_blocked(tmp_path):
    """
    Битый manifest.json это нарушение контракта RAW, а не
    поломка препроцессинга: паспорт обязан ответить blocked.
    """

    mini = MiniRaw(tmp_path / "raw")
    mini.cover_all("c1", first_seen="2023-01-01")
    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload())

    raw_dir = mini.write()

    (raw_dir / "manifest.json").write_text('{"schema_version": 5,', encoding="utf-8")

    report = build_passport(raw_dir, CONFIG, "train")

    assert report["status"] == STATUS_BLOCKED
    assert report["usable"] is False
    assert any("manifest.json" in item for item in report["errors"])


def test_stage_version_tracks_sources():
    """
    Правка любого модуля этапа без поднятия STAGE_VERSION не
    проходит: иначе прежние результаты считались бы актуальными.
    """

    stored = json.loads(Path("tests/prep_stage_sources.json").read_text(encoding="utf-8"))["passport"]

    assert stored["version"] == passport_module.STAGE_VERSION

    actual = {
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in STAGE_MODULES.items()
    }

    assert stored["modules"] == actual, (
        "модули этапа изменены: поднимите STAGE_VERSION и обновите tests/prep_stage_sources.json"
    )


# ============================================================
# РАЗБОР PAYLOAD
# ============================================================


def _catalogue_info(raw_dir: Path, event_type: str):
    return RawDataset(raw_dir).manifest.catalogue[event_type]


def test_missing_optional_key_is_allowed_and_extra_key_is_violation(tmp_path):

    mini = MiniRaw(tmp_path / "raw")
    mini.cover_all("c1")

    # Полный payload: эталон.
    mini.event("c1", "purchase", "2024-07-01 10:00:00", payload=purchase_payload())

    # Без необязательного ключа merchant_city: допустимый пропуск.
    short = purchase_payload()
    del short["merchant_city"]
    mini.event("c1", "purchase", "2024-07-02 10:00:00", payload=short)

    # С лишним ключом: нарушение контракта, строка сохраняется.
    mini.event("c1", "purchase", "2024-07-03 10:00:00", payload={**purchase_payload(), "zzz": 1})

    raw_dir = mini.write()

    info = _catalogue_info(raw_dir, "purchase")

    table = pq.read_table(raw_dir / "events.parquet")
    parsed = parse_payloads(info, table.column("payload"))

    assert parsed.table.num_rows == 3
    assert parsed.table.column("merchant_city").to_pylist() == ["Almaty", None, "Almaty"]
    assert parsed.counts == {"unexpected_key": 1}
    assert parsed.by_field["unexpected_key:zzz"] == 1
    assert parsed.samples[0]["field"] == "zzz" and parsed.samples[0]["row"] == 2


def test_missing_required_key_and_type_mismatch_are_violations(tmp_path):

    mini = MiniRaw(tmp_path / "raw")
    mini.cover_all("c1")

    no_amount = purchase_payload()
    del no_amount["amount"]
    mini.event("c1", "purchase", "2024-07-01 10:00:00", payload=no_amount)

    mini.event("c1", "purchase", "2024-07-02 10:00:00", payload=purchase_payload(amount="12500"))

    raw_dir = mini.write()

    parsed = parse_payloads(
        _catalogue_info(raw_dir, "purchase"),
        pq.read_table(raw_dir / "events.parquet").column("payload"),
    )

    assert parsed.counts["missing_required"] == 1
    assert parsed.counts["type_mismatch"] == 1
    assert parsed.by_field["missing_required:amount"] == 1
    assert parsed.by_field["type_mismatch:amount"] == 1
    assert parsed.table.column("amount").to_pylist() == [None, None]
    assert parsed.table.column("merchant_name").to_pylist() == ["Europharma 24", "Europharma 24"]


# ============================================================
# ВЕРДИКТ
# ============================================================


def test_clean_mini_raw_is_ready(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")
    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload())

    report = build_passport(mini.write(), CONFIG, "train")

    assert report["status"] == STATUS_READY
    assert report["usable"] is True
    assert report["errors"] == [] and report["contract_violations"] == []


def test_contract_violation_prevents_ready(tmp_path):
    """
    Горизонт полный, файлы целы, но payload расходится с каталогом
    ключей: набор не готов, и это не «ограничение».
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload())

    broken = purchase_payload()
    del broken["amount"]
    mini.event("c1", "purchase", "2023-03-06 10:00:00", payload=broken)

    report = build_passport(mini.write(), CONFIG, "train")

    assert report["status"] == STATUS_CONTRACT_MISMATCH
    assert report["usable"] is False
    assert report["errors"] == []
    assert report["contract_violations"] == ["purchase.amount: missing_required в 1 строках"]
    assert not any("missing_required" in item for item in report["limitations"])

    text = render_passport_md(report)
    assert "Нарушения входного контракта" in text and "purchase.amount" in text


def test_unknown_delivery_result_is_not_a_contract_violation(tmp_path):
    """
    У communication_sent.delivered три состояния: доставлено,
    не доставлено и результат неизвестен. Неизвестный результат
    это null, а не нарушение контракта и не false.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    def message(at: str, delivered):
        mini.event(
            "c1", "communication_sent", at,
            payload={
                "channel": "sms",
                "template": "T1",
                "campaign_code": "C1",
                "offer_id": None,
                "product_id": None,
                "purpose": "offer",
                "delivered": delivered,
            },
            initiator="bank",
        )

    message("2023-03-05 10:00:00", True)
    message("2023-03-06 10:00:00", False)
    message("2023-03-07 10:00:00", None)

    report = build_passport(mini.write(), CONFIG, "train")

    assert report["contract_violations"] == []
    assert report["status"] == STATUS_READY

    delivered = report["payload"]["by_event_type"]["communication_sent"]["null_share"]["delivered"]
    assert delivered == pytest.approx(1 / 3, abs=1e-6)


def test_unknown_event_type_is_contract_violation(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")
    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload())

    raw_dir = mini.write()

    # Тип, которого нет в каталоге ключей, подменяется прямо в файле,
    # после чего манифест переписывается под новое содержимое.
    table = pq.read_table(raw_dir / "events.parquet")
    patched = table.set_column(
        table.column_names.index("event_type"),
        "event_type",
        pa.array(["mystery_event"] * table.num_rows, pa.string()),
    )
    pq.write_table(patched, raw_dir / "events.parquet", compression="zstd")

    data = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    digest = ContentDigest()
    digest.extend(patched.to_pylist())
    data["content_sha256"]["events"] = digest.value()
    data["file_sha256"]["events.parquet"] = hashlib.sha256((raw_dir / "events.parquet").read_bytes()).hexdigest()
    (raw_dir / "manifest.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    report = build_passport(raw_dir, CONFIG, "train")

    assert report["status"] == STATUS_CONTRACT_MISMATCH
    assert report["errors"] == []
    assert any("mystery_event" in item for item in report["contract_violations"])


def test_blocked_beats_contract_mismatch(tiny_raw, tmp_path):

    copy = tmp_path / "raw"
    shutil.copytree(tiny_raw, copy)
    _corrupt(copy / "profile.parquet")

    report = build_passport(copy, CONFIG, "train")

    assert report["status"] == STATUS_BLOCKED
    assert report["usable"] is False
    assert any("profile.parquet" in item for item in report["errors"])
    assert "events" not in report


def test_null_in_required_envelope_field_is_blocked(tmp_path):
    """
    Пустое обязательное поле конверта это поломка, а не
    расхождение с каталогом ключей: по такой строке нельзя
    построить ни порядок, ни версию, ни связь. Диагностический
    флаг её не пропускает.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload())
    # Время события потеряно.
    mini.event("c1", "purchase", None, payload=purchase_payload(amount=999))

    raw_dir = mini.write()

    report = build_passport(raw_dir, CONFIG, "train")

    assert report["status"] == STATUS_BLOCKED
    assert report["usable"] is False
    assert report["events"]["required_nulls"] == {"event_time": 1}
    assert any("поле event_time: null в 1 строках" in item for item in report["errors"])
    assert report["contract_violations"] == []

    out = tmp_path / "processed"

    # Режим диагностики поломку не снимает.
    assert (
        _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out),
             "--allow-contract-mismatch")
        == EXIT_BLOCKED
    )


def test_missing_required_file_is_blocked(tiny_raw, tmp_path):

    copy = tmp_path / "raw"
    shutil.copytree(tiny_raw, copy)
    (copy / "catalog" / "merchants.parquet").unlink()

    report = build_passport(copy, CONFIG, "train")

    assert report["status"] == STATUS_BLOCKED
    assert any("catalog/merchants.parquet" in item for item in report["errors"])


# ============================================================
# СОДЕРЖИМОЕ
# ============================================================


def test_tiny_raw_passport_is_checked_whole(tiny_raw):

    report = build_passport(tiny_raw, CONFIG, "train")

    assert report["errors"] == []

    checks = report["checks"]
    assert all(item["status"] == "ok" for item in checks["rows"].values())
    assert all(item["status"] == "ok" for item in checks["content"].values())
    assert all(item["status"] == "ok" for item in checks["schemas"].values())
    assert all(
        item["status"] == ("not_read" if name.startswith("truth/") else "ok")
        for name, item in checks["files"].items()
    )

    events = report["events"]
    assert events["rows"] == RawDataset(tiny_raw).manifest.rows["events"]
    assert report["payload"]["rows_checked"] == events["rows"]

    # Горизонт короче согласованного, а у communication_sent.delivered
    # генератор пишет null при nullable=false. Это два разных вердикта,
    # и побеждает более тяжёлый.
    assert report["horizon"]["history_start_ok"] is False
    assert any("2023-01-01" in item for item in report["limitations"])
    assert report["status"] == (
        STATUS_CONTRACT_MISMATCH if report["contract_violations"] else STATUS_HORIZON_SHORT
    )

    months = report["months"]
    assert months["months"][0] == "2023-01"
    transactions = months["table"]["transactions"]
    assert transactions["2023-01"] == "—"
    assert isinstance(transactions["2024-07"], int)

    text = render_passport_md(report)
    assert "Источник × месяц" in text


def test_empty_month_is_not_an_error(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON, history_end=datetime(2026, 9, 1))
    mini.cover_all("c1", first_seen="2023-01-01")

    # Покупки только в двух месяцах трёх лет; заявок нет вовсе.
    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload())
    mini.event("c1", "purchase", "2025-11-05 10:00:00", payload=purchase_payload())

    report = build_passport(mini.write(), CONFIG, "train")

    assert report["status"] == STATUS_READY
    assert report["errors"] == []

    table = report["months"]["table"]
    assert table["transactions"]["2023-03"] == 1
    assert table["transactions"]["2023-04"] == 0
    assert table["applications"]["2024-01"] == 0
    assert "applications" in report["months"]["empty_months_of_available_sources"]
    assert table["support"]["2023-01"] == "n/a"


def test_short_horizon_is_reported_not_hidden(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=datetime(2024, 6, 1), history_end=datetime(2026, 9, 1))
    mini.cover_all("c1")
    mini.event("c1", "purchase", "2024-07-01 10:00:00", payload=purchase_payload())

    report = build_passport(mini.write(), CONFIG, "train")

    assert report["status"] == STATUS_HORIZON_SHORT
    assert report["usable"] is True
    assert report["horizon"]["history_start_ok"] is False
    assert report["horizon"]["groups"]["train"]["extract_time_ok"] is True


def test_history_end_before_group_cutoff_is_horizon_short(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON, history_end=datetime(2025, 6, 1))
    mini.cover_all("c1", first_seen="2023-01-01")
    mini.event("c1", "purchase", "2024-07-01 10:00:00", payload=purchase_payload())

    report = build_passport(mini.write(), CONFIG, "test")

    assert report["status"] == STATUS_HORIZON_SHORT
    assert report["horizon"]["history_start_ok"] is True
    assert report["horizon"]["groups"]["test"]["history_end_ok"] is False


def test_versions_duplicates_and_boundary_rows_are_counted(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    original = mini.event("c1", "purchase", "2024-07-01 10:00:00", payload=purchase_payload())
    mini.event(
        "c1", "purchase", "2024-07-01 10:00:00",
        payload=purchase_payload(), event_id=original, version=1,
    )
    mini.event(
        "c1", "purchase", "2024-07-01 10:00:00",
        payload=purchase_payload(amount=13000), event_id=original, version=2,
    )
    # Ровно на границе выгрузки: такой строки быть не должно, и она считается.
    mini.event("c1", "purchase", "2026-09-01 00:00:00", payload=purchase_payload())
    # До начала окна: реестр договоров, контекст.
    mini.event("c1", "purchase", "2018-05-01 10:00:00", payload=purchase_payload())

    report = build_passport(mini.write(), CONFIG, "train")

    events = report["events"]
    assert events["corrections"] == 1
    assert events["duplicates"] == 1
    assert events["repeated_id_version_pairs"] == {"pairs": 1, "extra_rows": 1}
    assert report["months"]["before_history_start"]["rows"] == 1
    assert report["months"]["event_time_at_or_after_extract"]["rows"] == 1
    assert any("повторные пары" in item for item in report["limitations"])
    assert any("границе extract_time" in item for item in report["limitations"])


def test_month_range_is_end_exclusive():
    assert month_range(datetime(2023, 1, 1), datetime(2023, 4, 1)) == ["2023-01", "2023-02", "2023-03"]
    assert month_range(datetime(2023, 1, 15), datetime(2023, 2, 10)) == ["2023-01", "2023-02"]


def test_passport_is_deterministic(tmp_path):

    mini = MiniRaw(tmp_path / "raw")
    mini.cover_all("c1")
    mini.event("c1", "purchase", "2024-07-01 10:00:00", payload=purchase_payload())
    raw_dir = mini.write()

    first = dumps_json(build_passport(raw_dir, CONFIG, "train"))
    second = dumps_json(build_passport(raw_dir, CONFIG, "train"))

    assert first == second
    assert "tmp" not in first  # ни одного пути в артефакте


def test_passport_never_reads_truth(tiny_raw, tmp_path):
    """
    truth/* подменяется мусором: если бы паспорт его читал или
    хэшировал, он бы упал или заблокировал группу.
    """

    copy = tmp_path / "raw"
    shutil.copytree(tiny_raw, copy)

    for path in (copy / "truth").glob("*.parquet"):
        path.write_bytes(b"garbage")

    report = build_passport(copy, CONFIG, "train")

    assert report["status"] in (STATUS_HORIZON_SHORT, STATUS_CONTRACT_MISMATCH)
    assert report["errors"] == []
    assert all(
        item["status"] == "not_read"
        for name, item in report["checks"]["files"].items()
        if name.startswith("truth/")
    )
    assert any(item["file"].startswith("truth/") for item in report["files"])


# ============================================================
# CLI, ОТПЕЧАТОК И КЭШ
# ============================================================


def _ready_raw(tmp_path: Path) -> Path:

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")
    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload())

    return mini.write()


def test_cli_writes_artifacts_and_skips_when_nothing_changed(tmp_path, capsys):

    raw_dir = _ready_raw(tmp_path)
    out = tmp_path / "processed"

    assert _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK

    assert (out / "passport" / "train.json").exists()
    assert (out / "passport" / "train.md").exists()
    assert (out / "preprocessing_manifest.json").exists()

    marker = fingerprint_path(out, "passport", "train")
    stored = load_fingerprint(marker)
    assert stored["stage"] == "passport" and stored["version"] == passport_module.STAGE_VERSION
    assert stored["status"] == STATUS_READY
    assert set(stored["outputs"]) == {"passport/train.json", "passport/train.md"}

    capsys.readouterr()

    assert _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK
    assert "пропущен" in capsys.readouterr().out

    # Смена конфига меняет отпечаток и пересчитывает этап.
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"required_history_start": "2022-01-01T00:00:00"}), encoding="utf-8")

    assert (
        _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out), "--config", str(config))
        == EXIT_OK
    )
    assert "пропущен" not in capsys.readouterr().out
    assert load_fingerprint(marker)["fingerprint"] != stored["fingerprint"]


def test_cli_recomputes_when_raw_changed_after_success(tmp_path, capsys):
    """
    Отпечаток считается по фактическим файлам, а не по числам из
    манифеста: иначе подменённый events.parquet протащил бы
    повреждение мимо проверки.
    """

    raw_dir = _ready_raw(tmp_path)
    out = tmp_path / "processed"

    assert _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK

    before = json.loads((out / "passport" / "train.json").read_text(encoding="utf-8"))
    assert before["status"] == STATUS_READY

    _corrupt(raw_dir / "events.parquet")

    capsys.readouterr()

    assert _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_BLOCKED

    printed = capsys.readouterr().out
    assert "пропущен" not in printed
    assert "events.parquet" in printed

    after = json.loads((out / "passport" / "train.json").read_text(encoding="utf-8"))
    assert after["status"] == STATUS_BLOCKED


def test_cli_restores_deleted_or_edited_report(tmp_path, capsys):

    raw_dir = _ready_raw(tmp_path)
    out = tmp_path / "processed"

    assert _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK

    md_path = out / "passport" / "train.md"
    json_path = out / "passport" / "train.json"
    original = md_path.read_text(encoding="utf-8")

    md_path.unlink()
    capsys.readouterr()

    assert _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK
    assert "пропущен" not in capsys.readouterr().out
    assert md_path.read_text(encoding="utf-8") == original

    # Правленый отчёт тоже не считается готовым результатом.
    json_path.write_text("{}", encoding="utf-8")
    capsys.readouterr()

    assert _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK
    assert "пропущен" not in capsys.readouterr().out
    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == STATUS_READY


def test_cli_contract_mismatch_needs_explicit_diagnostic_flag(tmp_path, capsys):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")
    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload={**purchase_payload(), "zzz": 1})
    raw_dir = mini.write()

    out = tmp_path / "processed"

    assert _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_CONTRACT_MISMATCH

    printed = capsys.readouterr().out
    assert "нарушение контракта" in printed
    assert "--allow-contract-mismatch" in printed

    report = json.loads((out / "passport" / "train.json").read_text(encoding="utf-8"))
    assert report["status"] == STATUS_CONTRACT_MISMATCH and report["usable"] is False

    # Режим диагностики меняет только код возврата, но не статус,
    # в том числе когда этап пропущен по отпечатку.
    assert (
        _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out), "--allow-contract-mismatch")
        == EXIT_OK
    )
    assert "пропущен" in capsys.readouterr().out

    assert _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_CONTRACT_MISMATCH


def test_cli_blocked_beats_contract_mismatch_across_groups(tmp_path):
    """
    У нескольких групп итог это самый тяжёлый случай, а не самый
    большой код: blocked (2) численно меньше contract_mismatch (3).
    """

    root = tmp_path / "raw"

    train = MiniRaw(root / "train", history_start=FULL_HORIZON)
    train.cover_all("c1", first_seen="2023-01-01")
    train.event("c1", "purchase", "2023-03-05 10:00:00", payload={**purchase_payload(), "zzz": 1})
    train.write()

    val = MiniRaw(root / "val", history_start=FULL_HORIZON)
    val.cover_all("c2", first_seen="2023-01-01")
    val.event("c2", "purchase", "2023-03-05 10:00:00", payload=purchase_payload())
    val_dir = val.write()
    _corrupt(val_dir / "events.parquet")

    out = tmp_path / "processed"

    assert _cli("passport", "--raw-root", str(root), "--out", str(out)) == EXIT_BLOCKED

    statuses = {
        group: json.loads((out / "passport" / f"{group}.json").read_text(encoding="utf-8"))["status"]
        for group in ("train", "val")
    }

    assert statuses == {"train": STATUS_CONTRACT_MISMATCH, "val": STATUS_BLOCKED}

    # Режим диагностики снимает только contract_mismatch; поломка
    # структуры остаётся поломкой.
    assert _cli("passport", "--raw-root", str(root), "--out", str(out), "--allow-contract-mismatch") == EXIT_BLOCKED


def test_cli_restores_missing_manifest_entry(tmp_path, capsys):
    """
    preprocessing_manifest.json это тоже результат этапа: удалённый
    или повреждённый, он восстанавливается из маркера, а не остаётся
    пустым потому, что отчёты на месте.
    """

    raw_dir = _ready_raw(tmp_path)
    out = tmp_path / "processed"

    assert _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK

    manifest_path = out / "preprocessing_manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = original["stages"]["passport"]["train"]

    manifest_path.unlink()
    capsys.readouterr()

    assert _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK

    printed = capsys.readouterr().out
    assert "пропущен" in printed and "восстановлена" in printed
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == original

    # Повреждённый файл тоже восстанавливается.
    manifest_path.write_text("{ не json", encoding="utf-8")
    capsys.readouterr()

    assert _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK
    assert "восстановлена" in capsys.readouterr().out
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["stages"]["passport"]["train"] == entry

    # Когда запись на месте, ничего не переписывается.
    capsys.readouterr()

    assert _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK
    assert "восстановлена" not in capsys.readouterr().out


def test_cli_blocked_group_exits_nonzero(tiny_raw, tmp_path):

    copy = tmp_path / "raw"
    shutil.copytree(tiny_raw, copy)
    (copy / "events.parquet").write_bytes(b"not a parquet file")

    assert _cli("passport", "--raw", str(copy), "--group", "val", "--out", str(tmp_path / "out")) == EXIT_BLOCKED


# ============================================================
# ЛОКАЛЬНЫЙ НАБОР
# ============================================================


def test_check_raw_passport(check_raw):

    report = build_passport(check_raw, CONFIG, "train")

    assert report["errors"] == []
    assert report["events"]["rows"] == RawDataset(check_raw).manifest.rows["events"]

    # Прежний дефект контракта закрыт: delivered объявлен
    # nullable, и неизвестный результат доставки это допустимый
    # пропуск, а не расхождение с каталогом ключей.
    assert report["contract_violations"] == []
    assert report["payload"]["violations_total"] == {}
    assert report["payload"]["by_event_type"]["communication_sent"]["null_share"]["delivered"] > 0

    # Остаётся честное ограничение горизонта: выгрузка начинается
    # позже согласованного начала.
    assert report["status"] == STATUS_HORIZON_SHORT
    assert report["usable"] is True

    # Мир объявлен: у выгрузки есть world_seed.
    assert not any("world_seed" in item for item in report["limitations"])
