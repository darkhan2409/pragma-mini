from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tests.test_downstream import chain
from tests.test_profile_state import EARLY, QUIET_SNAPSHOT, RAW_CLIENT


# ============================================================
# ИДЕЯ
# ============================================================
#
# Визуализация (viz/) берёт все факты у экспортера
# viz/scripts/export_demo.py. Проверяется, что экспортер честен
# на синтетическом мире (data/ не читается):
#
#   - виды токенов покрывают словарь целиком, неизвестный вид —
#     ошибка, а не пропуск;
#   - клиент в экспорте — ровно строка набора 05: события, токены,
#     анкета, календарь и давность до T;
#   - сырые значения событий из 02 сверены с токенами 05 тем же
#     кодированием, и расхождение останавливает экспорт;
#   - кандидаты шага MLM — шкала суммы сквозного события при том
#     же direction; суммы нет в токенах — экспорт отказывает.
#   - диагностика [USR] — только из отчёта текущей пробы прогона
#     с контролем init и без test.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]


def exporter():
    spec = importlib.util.spec_from_file_location("export_demo", ROOT / "viz" / "scripts" / "export_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def val_client(stage):
    """
    Синтетический клиент, доведённый до набора 05 группы val.
    """

    from src.preprocessing.read import Group
    from src.preprocessing.settings import PreprocessingConfig
    from src.temporal.samples import TemporalGroup
    from src.tokenization.finalvocab import FrozenArtifacts

    module = exporter()

    chain(stage, EARLY, QUIET_SNAPSHOT)

    artifacts = FrozenArtifacts.load()
    row = module.find_row("val", RAW_CLIENT)
    history = Group("val").history(RAW_CLIENT, TemporalGroup("val").cutoff)
    zone = PreprocessingConfig.load(None).bank_timezone()

    return module, artifacts, row, history, zone


def test_vocabulary_summary_covers_every_token():

    module = exporter()

    vocab = {"[PAD]": 0, "[UNK]": 1, "key:a": 2, "value:a=x": 3, "bucket:a_bucket_1": 4, "bpe:ab": 5}

    summary = module.vocabulary_summary(vocab)

    assert summary["size"] == 6
    assert sum(item["count"] for item in summary["kinds"].values()) == 6
    assert summary["kinds"]["bpe"] == {"count": 1, "first": 5, "last": 5}
    assert summary["specials"] == {"[PAD]": 0, "[UNK]": 1}

    with pytest.raises(module.ExportError, match="неизвестного вида"):
        module.vocabulary_summary({"odd": 0})


def test_the_exported_client_is_the_dataset_row(stage):

    module, artifacts, row, history, zone = val_client(stage)

    client = module.client_export(artifacts, row, history, zone, 256)

    assert client["client_id"] == RAW_CLIENT
    assert client["n_events"] == len(row["event_starts"]) == len(client["timeline"]) == len(EARLY)
    assert client["n_tokens"] == len(row["key_ids"]) == sum(item["n_tokens"] for item in client["timeline"])
    assert client["n_profile_tokens"] == len(client["profile"]) == len(row["profile_key_ids"])
    assert client["dropped_old_events"] == 0

    assert [item["time_log"] for item in client["timeline"]] == pytest.approx(
        [float(value) for value in row["event_time_log"]], abs=1e-5
    )

    hero = client["highlighted"]["hero"]

    # Единственная покупка мира — 5000 списанием.
    assert hero["type"] == "purchase" and hero["raw"]["transaction_amount"] == 5000

    start, length = row["event_starts"][hero["index"]], row["event_lengths"][hero["index"]]

    assert [(item["key_id"], item["value_id"], item["position"]) for item in hero["tokens"]] == list(zip(
        row["key_ids"][start:start + length],
        row["value_ids"][start:start + length],
        row["positions"][start:start + length],
    ))

    assert all(artifacts.describe(item["value_id"]) == item["value"] for item in hero["tokens"])
    assert hero["calendar"] == pytest.approx(
        [float(value) for value in row["calendar"][hero["index"] * 6:(hero["index"] + 1) * 6]], abs=1e-6
    )

    # [USR] и вехи: у анкеты давность 0, у вехи — до T.
    assert client["profile"][0]["value"] == "[USR]" and client["profile"][0]["time_log"] == 0.0


def test_a_mismatch_between_02_and_05_stops_the_export(stage):
    """
    Токены набора, которые не выходят из события ленты тем же
    кодированием, — сломанная сверка: экспорт отказывает.
    """

    module, artifacts, row, history, zone = val_client(stage)

    start = row["event_starts"][0]

    broken = dict(row, value_ids=list(row["value_ids"]))
    broken["value_ids"][start + 1] = artifacts.special("[UNK]") if broken["value_ids"][start + 1] != artifacts.special("[UNK]") else artifacts.special("[MASK]")

    with pytest.raises(module.ExportError, match="по-разному"):
        module.client_export(artifacts, broken, history, zone, 256)


def test_mlm_candidates_are_the_amount_scale_of_the_hero():
    """
    Кандидаты — все диапазоны суммы при том же direction, что у
    сквозного события, в порядке нижних границ; цель — токен его
    суммы. Чужие шкалы и другой direction в кандидаты не попадают.
    """

    module = exporter()

    buckets = {
        10: {"key": "transaction_amount", "name": "transaction_amount_debit_bucket_2", "min": 0.0, "max": 100.0, "when": "debit"},
        11: {"key": "transaction_amount", "name": "transaction_amount_debit_bucket_1", "min": None, "max": 0.0, "when": "debit"},
        12: {"key": "transaction_amount", "name": "transaction_amount_credit_bucket_1", "min": None, "max": 50.0, "when": "credit"},
        13: {"key": "balance_after", "name": "balance_after_bucket_1", "min": None, "max": 10.0, "when": None},
        14: {"key": "transaction_amount", "name": "transaction_amount_debit_bucket_3", "min": 100.0, "max": None, "when": "debit"},
    }

    hero = {"tokens": [
        {"key": "event_type", "kind": "value", "value_id": 3},
        {"key": "transaction_amount", "kind": "bucket", "value_id": 10, "range": buckets[10]},
    ]}

    found = module.mlm_candidates(buckets, hero)

    assert found["key"] == "transaction_amount" and found["target"] == 10
    assert [item["value_id"] for item in found["buckets"]] == [11, 10, 14]


def test_a_hero_without_an_encoded_amount_stops_the_mlm_step(stage):
    """
    В крошечном словаре мира суммы нет: покупка — это [EVT] и тип
    события. Шагу MLM нечего скрыть — экспорт отказывает понятной
    ошибкой.
    """

    module, artifacts, row, history, zone = val_client(stage)

    client = module.client_export(artifacts, row, history, zone, 256)

    with pytest.raises(module.ExportError, match="не закодирована"):
        module.mlm_candidates(module.bucket_index(artifacts), client["highlighted"]["hero"])


def test_the_export_is_compact_and_reads_back_the_same():
    """
    Контейнер из одних скаляров — одна строка: строка весов внимания,
    событие ленты. Остальное — с отступами. Читается тот же объект,
    что дал бы json.dumps, с теми же строковыми ключами.
    """

    import json

    module = exporter()

    payload = {
        "attention": {"blocks": [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]},
        "timeline": [{"t": "2026-01-01", "type": "purchase", "n": 3}, {"t": "2026-01-02", "type": "app_screen", "n": 1}],
        "candidates": {1623: 0.25, "1624": 0.5},
        "nested": {7: {"pieces": ["магнум", " express"], "ids": [1, [2]]}},
        "empty": {"list": [], "dict": {}},
        "none": None,
    }

    text = module.dumps(payload)
    lines = [line.strip() for line in text.splitlines()]

    assert json.loads(text) == json.loads(json.dumps(payload))
    assert "[0.1, 0.2, 0.3]," in lines and "[0.4, 0.5, 0.6]" in lines
    assert sum('"type"' in line for line in lines) == 2
    assert '"pieces": ["магнум", " express"],' in lines
    assert len(lines) < 30



def test_the_usr_diagnostic_is_taken_only_from_a_current_probe_against_init(stage):
    """
    Диагностика [USR] на экране вывода — из отчёта пробы прогона с
    --control init, без test и по векторам текущей выгрузки. Иначе её
    в экспорте нет: числа не пишутся руками и не берутся из старых
    отчётов.
    """

    import json

    from src.downstream.settings import downstream_dir
    from tests.test_downstream import raw_groups, write_vectors

    module = exporter()

    def delta(mean: float) -> dict:
        one = {"mean": mean, "low": mean - 0.1, "high": mean + 0.1, "not_better": 0.5}
        return {"val": {"pr_auc": one, "roc_auc": one}}

    def cell(pr: float) -> dict:
        return {"val": {"pr_auc": pr, "roc_auc": pr + 0.5, "f1": pr / 2, "rows": 3}}

    report = {
        "tag": "m", "control": "init", "final_test": False, "groups": ["train", "val"],
        "tasks": {"churn_active90": {"rows": {"val": 3}, "positives": {"val": 1}, "results": {
            "usr": {**cell(0.24), "vs_control": delta(-0.05)},
            "init:usr": cell(0.3),
            "catboost_usr": {**cell(0.18), "vs_lr": delta(-0.06)},
            "init:catboost_usr": cell(0.28),
            "catboost": cell(0.58),
        }}},
    }

    def write(record: dict) -> None:
        (downstream_dir("m") / "report.json").write_text(json.dumps(record))

    raw_groups("train", "val")
    write_vectors("m", ("train", "val"))
    write_vectors("init", ("train", "val"), seed=1)

    assert module.usr_diagnostic("m") is None

    write(report)
    found = module.usr_diagnostic("m")["tasks"]["churn_active90"]

    assert set(found["cells"]) == {"usr", "init:usr", "catboost_usr", "init:catboost_usr"}
    assert found["cells"]["usr"] == {"pr_auc": 0.24, "roc_auc": 0.74, "f1": 0.12}
    assert found["deltas"]["usr_vs_control"]["pr_auc"] == {"mean": -0.05, "low": -0.15, "high": 0.05}
    assert found["deltas"]["catboost_vs_lr"]["roc_auc"]["mean"] == -0.06
    assert (found["rows"], found["positives"]) == (3, 1)

    for damaged in (dict(report, control=None), dict(report, final_test=True)):
        write(damaged)
        assert module.usr_diagnostic("m") is None

    write(report)
    meta = downstream_dir("init") / "meta.json"
    record = json.loads(meta.read_text())
    record["groups"]["val"]["raw_events_sha256"] = "previous-generation"
    meta.write_text(json.dumps(record))
    assert module.usr_diagnostic("m") is None
