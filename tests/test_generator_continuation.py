from __future__ import annotations

import json
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.generator import config, emit
from src.generator.config import DatasetGroup
from src.generator.continuation import FUTURE_FILE, ContinuationError, _file_sha256, extend


# ============================================================
# ИДЕЯ
# ============================================================
#
# Продолжение группы — те же клиенты, прожитые дальше конца
# выгрузки. Оно годится для меток, только если прошлое выгрузки
# в нём то же самое, строка в строку; иначе оно отвергается и
# ничего не пишет. На диск ложится один хвост — события не раньше
# конца выгрузки, — и только вне data/.
# ============================================================


GROUP = DatasetGroup(
    clients=8,
    history_start=datetime(2024, 1, 1),
    history_end=datetime(2024, 4, 1),
    seed=100,
    registration_end=datetime(2024, 4, 1),
)

DAYS = 30

COMMUNITY = 4


@pytest.fixture(scope="module")
def horizon():
    """
    Окно генератора — процессный глобал: вернуть прежнее после модуля.
    """

    kept = (config.HISTORY_START, config.HISTORY_END, config.REGISTRATION_END)

    yield

    config.activate_horizon(*kept)


def source(root):

    emit.generate_dataset(
        total_clients=GROUP.clients,
        out_dir=root / "raw" / "train",
        seed=GROUP.seed,
        world_seed=42,
        history_start=GROUP.history_start,
        history_end=GROUP.history_end,
        registration_end=GROUP.registration_end,
        workers=1,
        community_size=COMMUNITY,
        quiet=True,
    )

    return root / "raw" / "train"


def rows(path) -> list[tuple]:

    table = pq.read_table(path)

    return list(zip(*[table.column(name).to_pylist() for name in ("client_id", "event_time", "source", "payload")]))


def test_the_continuation_keeps_the_past_and_writes_only_the_tail(tmp_path, horizon):

    raw = source(tmp_path)
    out = tmp_path / "future" / "train"

    record = extend(GROUP, raw, out, DAYS, quiet=True)

    past = rows(raw / "events.parquet")
    tail = rows(out / "events.parquet")
    end = config.event_time_text(GROUP.history_end)

    assert sorted(path.name for path in out.iterdir()) == ["events.parquet", FUTURE_FILE]
    assert not (out.parent / f".{out.name}.work").exists()

    # Прошлое совпало целиком, хвост — только с конца выгрузки, и
    # клиенты в нём только прежние.
    assert record["prefix"]["rows_export"] == record["prefix"]["rows_continuation"] == len(past)
    assert record["diverged_clients"] == []
    assert tail and all(item[1][:19] >= end[:19] for item in tail)
    assert {item[0] for item in tail} <= {item[0] for item in past}
    assert record["events_rows"] == len(tail)
    assert record["events_sha256"] == _file_sha256(out / "events.parquet")

    manifest = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    saved = json.loads((out / FUTURE_FILE).read_text(encoding="utf-8"))

    assert saved["source"] == {key: manifest[key] for key in ("period_end", "events_sha256", "profile_sha256")}
    assert saved["period_start"] == manifest["period_end"]
    assert saved["period_end"] == config.event_time_text(datetime(2024, 5, 2))


def test_a_continuation_that_changes_the_past_is_refused(tmp_path, horizon):
    """
    Выгрузка, чьё прошлое продолжение не повторяет (здесь — одна
    запись подменена), не получает хвоста.
    """

    raw = source(tmp_path)

    table = pq.read_table(raw / "events.parquet")
    payload = table.column("payload").to_pylist()
    changed = payload[0][:-1] + ',"note":"подменено"}'
    assert changed != payload[0]
    payload[0] = changed
    table = table.set_column(table.schema.get_field_index("payload"), "payload", pa.array(payload, pa.string()))
    pq.write_table(table, raw / "events.parquet")

    manifest = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    manifest["events_sha256"] = _file_sha256(raw / "events.parquet")
    (raw / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    out = tmp_path / "future" / "train"

    with pytest.raises(ContinuationError, match="меняет прошлое у 1 клиентов из 8"):
        extend(GROUP, raw, out, DAYS, quiet=True)

    assert not (out / "events.parquet").exists() and not (out / FUTURE_FILE).exists()

    # Если такая доля допустима, клиент с другим прошлым исключается:
    # его строк в хвосте нет, а сам он записан в future.json.
    client = table.column("client_id")[0].as_py()
    record = extend(GROUP, raw, out, DAYS, quiet=True, max_diverged=0.5)

    assert record["diverged_clients"] == [client]
    assert record["diverged_details"][0]["client_id"] == client
    assert client not in {item[0] for item in rows(out / "events.parquet")}


def test_the_continuation_refuses_another_group_and_the_data_directory(tmp_path, horizon):

    raw = source(tmp_path)

    with pytest.raises(ContinuationError, match="не той группы"):
        extend(DatasetGroup(**{**GROUP.__dict__, "seed": 101}), raw, tmp_path / "future" / "train", DAYS)

    # Вывод внутри data/ отвергается до генерации: generate_dataset
    # стёр бы каталог.
    with pytest.raises(ContinuationError, match="вне"):
        extend(GROUP, raw, config.DATA_DIR / "01_raw" / "train", DAYS)


def test_a_finished_continuation_is_verified_again_without_regenerating(tmp_path, horizon, monkeypatch):
    """
    Сверка оборвалась после генерации (на полном масштабе её убила
    нехватка памяти): повторный запуск берёт готовое продолжение с тем
    же паспортом и не генерирует заново.
    """

    from src.generator import continuation

    raw = source(tmp_path)
    out = tmp_path / "future" / "train"

    def crash(*args, **kwargs):
        raise MemoryError("сверка оборвалась")

    with monkeypatch.context() as patch:
        patch.setattr(continuation, "_client_digests", crash)
        with pytest.raises(MemoryError):
            extend(GROUP, raw, out, DAYS, quiet=True)

    assert (out.parent / f".{out.name}.work" / "manifest.json").exists()

    def regenerate(*args, **kwargs):
        raise AssertionError("готовое продолжение сгенерировано заново")

    monkeypatch.setattr(continuation, "generate_dataset", regenerate)

    record = extend(GROUP, raw, out, DAYS, quiet=True)

    assert record["prefix"]["rows_export"] == len(rows(raw / "events.parquet"))
    assert not (out.parent / f".{out.name}.work").exists()
