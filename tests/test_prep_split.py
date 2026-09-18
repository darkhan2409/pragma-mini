from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.generator.emit import PRODUCTS_SCHEMA
from src.preprocessing.canonical.build import build_group
from src.preprocessing.run import EXIT_BLOCKED, EXIT_CONTRACT_MISMATCH, EXIT_OK
from src.preprocessing.run import main as run_main
from src.preprocessing.settings import PreprocessingConfig
from src.preprocessing.split import (
    SPLIT_MANIFEST_FILE,
    STATUS_BLOCKED,
    STATUS_BLOCKED_BY_INPUT,
    STATUS_OK,
    TRAIN_INDEX_FILE,
    GroupInput,
    SplitError,
    TrainCorpus,
    build_split,
)

from tests.prep_fixtures import MiniRaw, purchase_payload  # noqa: F401


CONFIG = PreprocessingConfig()

FULL_HORIZON = datetime(2023, 1, 1)

# Мир, общий для трёх выгрузок: один world_seed и одинаковые
# справочники. Разные seed популяций — обязательное условие.
WORLD_SEED = 77

SEEDS = {"train": 1, "val": 2, "test": 3}

FIT_END = CONFIG.windows["train"].final_cutoff


def _cli(*args: str) -> int:
    with pytest.raises(SystemExit) as result:
        run_main(list(args))
    return int(result.value.code)


def _group(root: Path, name: str, world_seed: int | None = WORLD_SEED) -> MiniRaw:
    return MiniRaw(root / name, history_start=FULL_HORIZON, seed=SEEDS[name], world_seed=world_seed)


def _client(mini: MiniRaw, client_id: str, amount: int = 12500, at: str = "2025-06-01 10:00:00") -> None:
    """
    Клиент с покрытием и одной покупкой.
    """

    mini.cover_all(client_id, first_seen="2023-01-01")
    mini.event(client_id, "purchase", at, payload=purchase_payload(amount=amount))


def _prepare(root: Path, out: Path, groups: dict[str, MiniRaw]) -> list[GroupInput]:
    """
    Пишет группы, собирает их canonical и возвращает вход этапа 4.
    """

    sources: list[GroupInput] = []

    for name, mini in groups.items():

        raw_dir = mini.write()
        canonical_dir = out / "canonical" / name

        build_group(raw_dir, canonical_dir, CONFIG, name)

        sources.append(
            GroupInput(
                group=name,
                raw_dir=raw_dir,
                canonical_dir=canonical_dir,
                canonical_fingerprint=f"fingerprint-{name}",
            )
        )

    return sources


def _three_groups(root: Path, out: Path, train: MiniRaw | None = None, **kwargs) -> list[GroupInput]:
    """
    Три независимые группы по одному клиенту, если не сказано иное.
    """

    groups: dict[str, MiniRaw] = {}

    for name in ("train", "val", "test"):

        if name == "train" and train is not None:
            groups[name] = train
            continue

        mini = _group(root, name, **kwargs)
        _client(mini, f"{name}_c1")
        groups[name] = mini

    return _prepare(root, out, groups)


def _train_checksum(report: dict) -> str:
    return report["train_corpus"]["checksum"]["content_sha256"]


# ============================================================
# ПОЛНЫЙ ПУТЬ
# ============================================================


def test_pipeline_split_keeps_silent_clients_and_excludes_test_accounts(tmp_path, capsys):
    """
    Сквозной путь паспорт → canonical → разделение на трёх
    независимых выгрузках общего мира.
    """

    root = tmp_path / "raw"
    out = tmp_path / "processed"

    train = _group(root, "train")

    _client(train, "train_c1")
    _client(train, "train_c2", amount=700)

    # Молчащий клиент: покрытие есть, событий нет. Он остаётся в
    # группе — отсутствие событий не повод убрать человека.
    train.cover_all("train_silent", first_seen="2023-01-01")

    # Технический аккаунт: остаётся в canonical, но не в рабочей группе.
    train.cover_all("train_test_account", first_seen="2023-01-01")
    train.event(
        "train_test_account", "purchase", "2025-06-01 10:00:00",
        payload=purchase_payload(), is_test_account=True,
    )

    groups = {"train": train}

    for name in ("val", "test"):
        mini = _group(root, name)
        _client(mini, f"{name}_c1")
        groups[name] = mini

    for mini in groups.values():
        mini.write()

    assert _cli("passport", "--raw-root", str(root), "--out", str(out), "--name", "x") == EXIT_OK
    assert _cli("canonical", "--raw-root", str(root), "--out", str(out), "--name", "x") == EXIT_OK

    capsys.readouterr()

    assert _cli("split", "--raw-root", str(root), "--out", str(out), "--name", "x") == EXIT_OK

    printed = capsys.readouterr().out
    assert "train" in printed and "train-интерфейс" in printed

    manifest = json.loads((out / "split" / SPLIT_MANIFEST_FILE).read_text(encoding="utf-8"))

    assert manifest["status"] == STATUS_OK
    assert manifest["usable"] is True
    assert manifest["shared_world"] is True

    train_group = manifest["groups"]["train"]

    assert train_group["clients_total"] == 4
    assert train_group["clients_working"] == 3
    assert train_group["excluded"] == {"test_account": 1}
    assert train_group["excluded_clients"] == ["train_test_account"]

    # Молчащий клиент в группе есть, событий не даёт.
    assert "train_silent" in train_group["clients"]
    assert train_group["silent_clients"] == 1

    index = pq.read_table(out / "split" / TRAIN_INDEX_FILE)

    assert index.num_rows == train_group["visible_events"] == 2
    assert set(index.column("mlm_target_eligible").to_pylist()) == {True}

    # Запись этапа в общем манифесте.
    entry = json.loads((out / "preprocessing_manifest.json").read_text(encoding="utf-8"))["stages"]["split"]
    assert entry["status"] == STATUS_OK and entry["train_rows"] == 2

    # Повторный запуск ничего не пересчитывает.
    capsys.readouterr()
    assert _cli("split", "--raw-root", str(root), "--out", str(out), "--name", "x") == EXIT_OK
    assert "этап пропущен" in capsys.readouterr().out

    # --- интерфейс корпуса держит границу сам ---

    corpus = TrainCorpus.open(out / "split", out / "canonical" / "train")

    # fit_end взят из манифеста, а не из переданного конфига.
    assert corpus.fit_end == FIT_END
    assert corpus.client_ids == train_group["clients"]

    assert corpus.history("train_c1").events.num_rows == 1
    # Молчащий клиент разрешён и просто не даёт строк.
    assert corpus.history("train_silent").events.num_rows == 0

    for refused in ("train_test_account", "val_c1", "нет_такого"):
        with pytest.raises(SplitError, match="разрешённую train-группу"):
            corpus.history(refused)

    # Canonical, пересобранный после разделения, корпус не подтверждает.
    manifest_path = out / "split" / SPLIT_MANIFEST_FILE
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["groups"]["train"]["canonical_fingerprint"] = "другой-отпечаток"
    manifest_path.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(SplitError, match="не тот, на котором построено разделение"):
        TrainCorpus.open(out / "split", out / "canonical" / "train")


# ============================================================
# НЕЗАВИСИМОСТЬ ПОПУЛЯЦИЙ
# ============================================================


def test_shared_client_blocks_the_split(tmp_path):
    """
    Общий клиент у двух групп это не ограничение, а поломка:
    обучение и проверка перестают быть независимыми.
    """

    root = tmp_path / "raw"
    out = tmp_path / "processed"

    groups: dict[str, MiniRaw] = {}

    for name in ("train", "val", "test"):
        mini = _group(root, name)
        _client(mini, f"{name}_c1")
        groups[name] = mini

    # Один и тот же человек попал в две выгрузки.
    _client(groups["val"], "train_c1")

    report = build_split(_prepare(root, out, groups), CONFIG, out / "split").report

    assert report["status"] == STATUS_BLOCKED
    assert report["population"]["shared_clients"] == 1
    assert any("делят 1 клиентов" in item for item in report["errors"])

    # Разрешённый корпус на сломанном входе не выдаётся.
    assert report["train_corpus"] is None
    assert not (out / "split" / TRAIN_INDEX_FILE).exists()


# ============================================================
# ОБЩИЙ МИР
# ============================================================


def test_world_mismatch_is_an_input_dependency(tmp_path):
    """
    Разные справочники — свойство входа, а не ошибка этапа:
    артефакты пишутся, но набор объявлен непригодным.
    """

    root = tmp_path / "raw"
    out = tmp_path / "processed"

    sources = _three_groups(root, out)

    report = build_split(sources, CONFIG, out / "split").report

    assert report["status"] == STATUS_OK and report["shared_world"] is True

    # Без world_seed мир остаётся общим, если справочники, хронология
    # и конфигурация совпали: неподтверждённое происхождение это
    # ограничение, а не блокировка.
    nameless = build_split(
        _three_groups(tmp_path / "raw_ns", tmp_path / "out_ns", world_seed=None),
        CONFIG,
        tmp_path / "out_ns" / "split",
    ).report

    assert nameless["status"] == STATUS_OK and nameless["shared_world"] is True
    assert any("происхождение общего мира" in item for item in nameless["limitations"])

    # Тот же вход, но у test свой каталог продуктов.
    other = pa.Table.from_pylist(
        [{name: None for name in PRODUCTS_SCHEMA.names} | {"product_id": "prd_x", "product_code": "X"}],
        schema=PRODUCTS_SCHEMA,
    )

    test_raw = next(item.raw_dir for item in sources if item.group == "test")
    pq.write_table(other, test_raw / "catalog" / "products.parquet", compression="zstd")

    report = build_split(sources, CONFIG, out / "split_2").report

    assert report["status"] == STATUS_BLOCKED_BY_INPUT
    assert report["shared_world"] is False
    assert report["usable"] is False
    assert any("справочник products различается" in item for item in report["input_dependencies"])

    # Артефакты всё равно записаны: это диагностика входа.
    assert (out / "split_2" / SPLIT_MANIFEST_FILE).exists()
    assert (out / "split_2" / TRAIN_INDEX_FILE).exists()

    # Но обучению непригодный корпус не открывается.
    with pytest.raises(SplitError, match="непригодно для обучения"):
        TrainCorpus.open(out / "split_2", out / "canonical" / "train")

    diagnostic = TrainCorpus.open(out / "split_2", out / "canonical" / "train", allow_unusable=True)

    assert diagnostic.fit_end == FIT_END
    assert diagnostic.client_ids == report["groups"]["train"]["clients"]


# ============================================================
# TRAIN-СОДЕРЖИМОЕ
# ============================================================


def test_train_content_does_not_depend_on_other_groups(tmp_path):
    """
    Замена validation и test другими выгрузками не меняет того,
    что разрешено видеть на train.
    """

    first = build_split(
        _three_groups(tmp_path / "raw_a", tmp_path / "out_a"), CONFIG, tmp_path / "out_a" / "split"
    ).report

    root = tmp_path / "raw_b"
    out = tmp_path / "out_b"

    groups = {"train": _group(root, "train")}
    _client(groups["train"], "train_c1")

    # Другие val и test: другие клиенты, другие суммы, другие даты.
    for name in ("val", "test"):
        mini = _group(root, name)
        _client(mini, f"{name}_other", amount=999, at="2025-09-09 09:00:00")
        _client(mini, f"{name}_more", amount=5, at="2024-02-02 08:00:00")
        groups[name] = mini

    second = build_split(_prepare(root, out, groups), CONFIG, out / "split").report

    assert _train_checksum(first) == _train_checksum(second)
    assert first["groups"]["val"]["visible_events"] != second["groups"]["val"]["visible_events"]


def test_future_event_does_not_change_train_content_but_edit_does(tmp_path):
    """
    Контрольная сумма считается по известному на fit_end: то, что
    произойдёт позже, в неё не входит, а правка видимой строки —
    входит.
    """

    def checksum(folder: str, *, future: bool = False, amount: int = 12500) -> str:

        root = tmp_path / folder
        out = tmp_path / f"{folder}_out"

        train = _group(root, "train")
        _client(train, "train_c1", amount=amount)

        if future:
            # Позже fit_end: на этом срезе события ещё нет.
            train.event("train_c1", "purchase", "2026-03-01 10:00:00", payload=purchase_payload(amount=4242))

        sources = _three_groups(root, out, train=train)

        report = build_split(sources, CONFIG, out / "split").report

        assert report["train_corpus"]["checksum"]["cutoff"] == FIT_END.isoformat()

        return _train_checksum(report)

    base = checksum("raw_base")

    assert checksum("raw_future", future=True) == base
    assert checksum("raw_edited", amount=13000) != base


# ============================================================
# ЦЕЛОСТНОСТЬ КОРПУСА И ДВА ВЕРДИКТА
# ============================================================


def _pipeline(tmp_path: Path, history_start=FULL_HORIZON) -> tuple[Path, Path]:
    """
    Три группы, пройденные паспортом, canonical и разделением.
    """

    root = tmp_path / "raw"
    out = tmp_path / "processed"

    for name in ("train", "val", "test"):
        mini = MiniRaw(root / name, history_start=history_start, seed=SEEDS[name], world_seed=WORLD_SEED)
        _client(mini, f"{name}_c1")
        mini.write()

    assert _cli("passport", "--raw-root", str(root), "--out", str(out), "--name", "x") in (
        EXIT_OK,
        EXIT_CONTRACT_MISMATCH,
    )
    assert _cli("canonical", "--raw-root", str(root), "--out", str(out), "--name", "x") == EXIT_OK
    assert _cli("split", "--raw-root", str(root), "--out", str(out), "--name", "x") == EXIT_OK

    return root, out


def test_edited_canonical_is_refused_at_open(tmp_path):
    """
    Корпус читает ФАЙЛЫ canonical, поэтому сверяются они, а не
    только отпечаток.

    Отпечаток подтверждает согласие маркера с манифестом и
    остаётся прежним, если файл правили на месте. Обучение
    пошло бы по строкам, которых разделение не видело.
    """

    _root, out = _pipeline(tmp_path)

    # До правки корпус открывается.
    TrainCorpus.open(out / "split", out / "canonical" / "train")

    events_path = out / "canonical" / "train" / "events.parquet"

    table = pq.read_table(events_path)

    amounts = table.column("amount").to_pylist()

    assert amounts and amounts[0] == 12500

    changed = table.set_column(
        table.column_names.index("amount"),
        "amount",
        pa.array([12775 if value is not None else None for value in amounts], table.column("amount").type),
    )

    pq.write_table(changed, events_path)

    with pytest.raises(SplitError, match="файлы canonical изменились"):
        TrainCorpus.open(out / "split", out / "canonical" / "train")

    # Режим диагностики проверок не делает — он для разбора.
    TrainCorpus.open(out / "split", out / "canonical" / "train", allow_unusable=True)


def test_edited_train_index_is_refused(tmp_path):
    """
    Индекс корпуса это разрешение на строки обучения, и выдаёт его
    разделение, а не тот, кто правил parquet.

    Проверялось только число строк: индекс с перевёрнутым
    mlm_target_eligible при том же числе строк открывался как
    родной. Теперь манифест несёт sha256 файла индекса.
    """

    _root, out = _pipeline(tmp_path)

    TrainCorpus.open(out / "split", out / "canonical" / "train")

    manifest = json.loads((out / "split" / SPLIT_MANIFEST_FILE).read_text(encoding="utf-8"))

    assert manifest["train_corpus"]["index_sha256"]

    index_path = out / "split" / TRAIN_INDEX_FILE

    table = pq.read_table(index_path)

    eligible = table.column("mlm_target_eligible").to_pylist()

    changed = table.set_column(
        table.column_names.index("mlm_target_eligible"),
        "mlm_target_eligible",
        pa.array([not value for value in eligible], pa.bool_()),
    )

    pq.write_table(changed, index_path)

    # Число строк то же: прежняя проверка этого не заметила бы.
    assert pq.read_table(index_path).num_rows == manifest["train_corpus"]["rows"]

    with pytest.raises(SplitError, match="индекс корпуса изменён"):
        TrainCorpus.open(out / "split", out / "canonical" / "train")


def test_short_horizon_needs_explicit_flag(tmp_path):
    """
    Техническая исправность и выполнение договорённости о
    горизонте это РАЗНЫЕ вердикты.

    Разделение с короткой историей исправно: файлы целы, группы
    независимы, мир общий. Но согласованный горизонт оно не
    выполняет, и молча принимать это нельзя.
    """

    _root, out = _pipeline(tmp_path, history_start=datetime(2025, 1, 1))

    manifest = json.loads((out / "split" / SPLIT_MANIFEST_FILE).read_text(encoding="utf-8"))

    assert manifest["usable"] is True
    assert manifest["contract_met"] is False
    assert manifest["contract"]["horizon_ok"] is False
    assert manifest["contract"]["reasons"]

    with pytest.raises(SplitError, match="договорённый горизонт не выполнен"):
        TrainCorpus.open(out / "split", out / "canonical" / "train")

    corpus = TrainCorpus.open(
        out / "split", out / "canonical" / "train", allow_short_horizon=True
    )

    assert corpus.fit_end == FIT_END


# ============================================================
# ВЕРСИЯ ЭТАПА
# ============================================================


def test_stage_version_tracks_sources():

    from src.preprocessing import history as history_module
    from src.preprocessing import run as run_module
    from src.preprocessing import settings as settings_module
    from src.preprocessing import split as split_module

    modules = {
        "split": split_module,
        "history": history_module,
        "settings": settings_module,
        "run": run_module,
    }

    stored = json.loads(Path("tests/prep_stage_sources.json").read_text(encoding="utf-8"))["split"]

    assert stored["version"] == split_module.STAGE_VERSION

    actual = {
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in modules.items()
    }

    assert stored["modules"] == actual, (
        "модули этапа изменены: поднимите STAGE_VERSION и обновите tests/prep_stage_sources.json"
    )
