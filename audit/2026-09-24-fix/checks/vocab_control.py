"""
Контрольный эксперимент: словарь и статистики учатся только на train.

    python audit/2026-09-24-fix/checks/vocab_control.py --source regression

Утверждение проверяется не чтением кода, а вмешательством:

  1. собирается словарь по неизменённым train, val, test;
  2. ПОРТЯТСЯ val и test — суммы, категории, состав клиентов;
     словарь собирается снова: артефакты обязаны совпасть до
     байта;
  3. отрицательный контроль: так же портится TRAIN, и словарь
     обязан измениться. Без этого шага совпадение в пункте 2
     ничего не доказывало бы — оно могло бы означать, что
     эксперимент вообще ни на что не влияет.

Настоящий data/ не участвует: каталоги этапов переставлены на
временный корень внутри runs/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from importlib import import_module
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[1]
RUNS = AUDIT / "runs"

sys.path.insert(0, str(ROOT))

GROUPS = ("train", "val", "test")

# (модуль, имя глобала, подкаталог)
PLACES = (
    ("src.preprocessing.settings", "RAW_DIR", "01_raw"),
    ("src.preprocessing.settings", "PREPROCESSED_DIR", "02_preprocessed"),
    ("src.tokenization.settings", "VOCAB_DIR", "03_vocab"),
    ("src.tokenization.settings", "TOKENIZED_DIR", "04_tokenized"),
    ("src.tokenization.finalvocab", "VOCAB_DIR", "03_vocab"),
)


def redirect(root: Path) -> None:

    for name, attribute, folder in PLACES:
        module = import_module(name)
        assert hasattr(module, attribute), f"{name}.{attribute} больше нет"
        setattr(module, attribute, root / folder)


def preprocess(group: str) -> int:

    from src.preprocessing.canonical.build import build_group
    from src.preprocessing.settings import PreprocessingConfig, group_dir, raw_group_dir

    out = group_dir(group)

    if out.exists():
        shutil.rmtree(out)

    result = build_group(raw_group_dir(group), out, PreprocessingConfig.load(None), group)

    return int(result.events_rows)


def fit() -> dict:
    """
    Словарь целиком и отпечаток каждого его файла.
    """

    from src.tokenization.run import build_parser, run_fit
    from src.tokenization.settings import VOCAB_DIR

    if VOCAB_DIR.exists():
        shutil.rmtree(VOCAB_DIR)

    args = build_parser().parse_args(["fit"])

    code = run_fit(args)

    if code != 0:
        raise SystemExit(f"словарь не собран, код {code}")

    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(VOCAB_DIR.rglob("*"))
        if path.is_file()
    }


def damage(directory: Path) -> dict:
    """
    Порча выгрузки группы: суммы, категории и состав клиентов.

    Портится RAW, потому что именно с него начинается путь к
    словарю. Изменения крупные нарочно: мелкие могли бы не
    дойти до границ квантилей.
    """

    events = directory / "events.parquet"

    table = pq.read_table(events)

    ids = table.column("client_id").to_pylist()
    payloads = [json.loads(raw) for raw in table.column("payload").to_pylist()]

    touched = 0

    for payload in payloads:
        if isinstance(payload.get("amount"), int):
            payload["amount"] = payload["amount"] * 7 + 13
            touched += 1
        if payload.get("merchant_category"):
            payload["merchant_category"] = "audit_unseen_category"
            touched += 1

    # И состав клиентов: выкидывается каждый третий.
    keep = {value for index, value in enumerate(sorted(set(ids))) if index % 3}

    # Колонки разворачиваются ОДИН раз: обращение к to_pylist()
    # внутри перебора давало бы квадрат от числа строк.
    times = table.column("event_time").to_pylist()
    sources = table.column("source").to_pylist()

    rows = [
        (client, times[index], sources[index], json.dumps(payloads[index], ensure_ascii=False))
        for index, client in enumerate(ids)
        if client in keep
    ]

    pq.write_table(
        pa.table(
            {
                "client_id": pa.array([item[0] for item in rows], pa.string()),
                "event_time": pa.array([item[1] for item in rows], pa.string()),
                "source": pa.array([item[2] for item in rows], pa.string()),
                "payload": pa.array([item[3] for item in rows], pa.string()),
            },
            schema=table.schema,
        ),
        events,
        compression="zstd",
    )

    profile = directory / "profile.parquet"

    snapshot = pq.read_table(profile)

    mask = [value in keep for value in snapshot.column("client_id").to_pylist()]

    pq.write_table(snapshot.filter(pa.array(mask)), profile, compression="zstd")

    # Манифест больше не описывает выгрузку: препроцессинг обязан
    # читать её по факту, поэтому счётчики строк обновляются.
    card = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    card["events_rows"] = len(rows)
    card["profile_rows"] = int(sum(mask))
    card["events_sha256"] = hashlib.sha256(events.read_bytes()).hexdigest()
    card["profile_sha256"] = hashlib.sha256(profile.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {"payload_fields_changed": touched, "clients_left": len(keep), "rows_left": len(rows)}


def lay_out(root: Path, source: Path) -> None:
    """
    Свежая копия выгрузок во временный корень.
    """

    raw = root / "01_raw"

    if raw.exists():
        shutil.rmtree(raw)

    for group in GROUPS:
        shutil.copytree(source / group, raw / group)


def main() -> int:

    parser = argparse.ArgumentParser(prog="vocab_control")
    parser.add_argument("--source", default="regression", help="каталог выгрузок под runs/")
    parser.add_argument("--out", default="evidence/vocab-control.json")

    args = parser.parse_args()

    source = (RUNS / args.source).resolve()

    root = (RUNS / "vocab-control-root").resolve()

    redirect(root)

    report: dict = {"source": str(source), "root": str(root)}

    # --- 1. Эталон ---

    lay_out(root, source)

    report["rows"] = {group: preprocess(group) for group in GROUPS}

    baseline = fit()

    report["vocab_files"] = len(baseline)

    # --- 2. Портим val и test ---

    lay_out(root, source)

    report["damage_val_test"] = {
        group: damage(root / "01_raw" / group) for group in ("val", "test")
    }

    for group in GROUPS:
        preprocess(group)

    after_holdout = fit()

    changed = sorted(
        name for name in set(baseline) | set(after_holdout)
        if baseline.get(name) != after_holdout.get(name)
    )

    report["changed_after_holdout_damage"] = changed

    # --- 3. Отрицательный контроль: портим train ---

    lay_out(root, source)

    report["damage_train"] = {"train": damage(root / "01_raw" / "train")}

    for group in GROUPS:
        preprocess(group)

    after_train = fit()

    changed_train = sorted(
        name for name in set(baseline) | set(after_train)
        if baseline.get(name) != after_train.get(name)
    )

    report["changed_after_train_damage"] = changed_train

    report["verdict"] = (
        "PASS" if not changed and changed_train else "FAIL"
    )

    report["note"] = (
        "порча val и test словарь не изменила; порча train изменила — "
        "значит эксперимент достаёт до словаря, и совпадение в пункте 2 "
        "не следствие бездействия"
        if report["verdict"] == "PASS"
        else "см. списки изменившихся файлов"
    )

    (AUDIT / args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"файлов словаря: {report['vocab_files']}")
    print(f"после порчи val и test изменилось файлов: {len(changed)} {changed}")
    print(f"после порчи train изменилось файлов: {len(changed_train)}")
    print("ИТОГ:", report["verdict"], "—", report["note"])

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
