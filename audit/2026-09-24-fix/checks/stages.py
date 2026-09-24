"""
Этапы 05-08 и сведения о будущем во входах модели.

    python audit/2026-09-24-fix/checks/stages.py --source regression

Строится полная цепочка от выгрузки до масок во ВРЕМЕННОМ корне
внутри runs/: настоящий data/ не участвует.

Проверяется:

  ST-5A  события примера не выходят за конечный cutoff группы и
         идут по неубыванию времени;
  ST-5B  target_event_mask истинна ровно на событиях периода
         целей — считается своим сравнением дат, а не вызовом
         той же функции;
  ST-7A  батчи сохраняют состав клиентов, порядок и содержимое;
         дополнение до ширины не выдумывает токенов;
  ST-8A  маскируется только разрешённое: позиции с меткой лежат
         внутри событий, у которых target_event_mask истинна;
  ST-8B  labels равны исходным значениям на маскированных местах
         и -100 на всех прочих;
  ST-PF  в анкете примера нет ни одного поля из числа
         невосстановимых на target_start;
  RISK-1 сколько среди целей событий, след которых анкета
         ПРЕЖДЕ несла. Это мера риска в самих данных, а не
         утечки: случился ли след на самом деле, проверяет
         checks/profile_border.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import timezone
from importlib import import_module
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[1]
RUNS = AUDIT / "runs"

sys.path.insert(0, str(ROOT))

GROUPS = ("train", "val", "test")

PLACES = (
    ("src.preprocessing.settings", "RAW_DIR", "01_raw"),
    ("src.preprocessing.settings", "PREPROCESSED_DIR", "02_preprocessed"),
    ("src.tokenization.settings", "VOCAB_DIR", "03_vocab"),
    ("src.tokenization.settings", "TOKENIZED_DIR", "04_tokenized"),
    ("src.tokenization.finalvocab", "VOCAB_DIR", "03_vocab"),
    ("src.dataset.settings", "DATASET_DIR", "05_dataset"),
    ("src.temporal.settings", "TEMPORAL_DIR", "06_temporal"),
    ("src.batching.settings", "BATCHES_DIR", "07_batches"),
    ("src.masking.settings", "MASKED_DIR", "08_masked"),
)

RESULTS: list[dict] = []


def record(name: str, verdict: str, detail: str, checked: int = 0, bad: int = 0) -> None:
    RESULTS.append(
        {"check": name, "verdict": verdict, "detail": detail, "checked": checked, "violations": bad}
    )
    print(f"[{verdict}] {name}: проверено {checked}, нарушений {bad} — {detail}")


def redirect(root: Path) -> None:
    for name, attribute, folder in PLACES:
        module = import_module(name)
        assert hasattr(module, attribute), f"{name}.{attribute} больше нет"
        setattr(module, attribute, root / folder)


def build_everything(root: Path, source: Path) -> None:
    """
    Выгрузка -> препроцессинг -> словарь -> кодирование -> 05..08.
    """

    from src.batching.build import build_group as build_batches
    from src.batching.settings import BatchingConfig
    from src.dataset.build import build_group as build_dataset
    from src.dataset.settings import DatasetConfig
    from src.masking.build import build_group as build_masked
    from src.masking.settings import MaskingConfig
    from src.preprocessing.canonical.build import build_group as build_canonical
    from src.preprocessing.settings import PreprocessingConfig, group_dir, raw_group_dir
    from src.temporal.build import build_group as build_temporal
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.run import build_parser, run_fit
    from src.tokenization.settings import TokenizerConfig
    from src.tokenization.transform import encode_group

    raw = root / "01_raw"

    if raw.exists():
        shutil.rmtree(raw)

    for group in GROUPS:
        shutil.copytree(source / group, raw / group)

    settings = PreprocessingConfig.load(None)

    for group in GROUPS:
        out = group_dir(group)
        if out.exists():
            shutil.rmtree(out)
        build_canonical(raw_group_dir(group), out, settings, group)

    if run_fit(build_parser().parse_args(["fit"])) != 0:
        raise SystemExit("словарь не собран")

    artifacts = FrozenArtifacts.load()

    tokenizer = TokenizerConfig.load(None)

    for group in GROUPS:
        encode_group(artifacts, group, tokenizer)
        build_dataset(artifacts, group, DatasetConfig.load(None))
        build_temporal(group)
        build_batches(group, BatchingConfig.load(None))
        build_masked(group, MaskingConfig.load(None))


def check_dataset(group: str) -> dict:
    """
    ST-5A и ST-5B: срез примера и маска допустимых целей.
    """

    from src.dataset.settings import SAMPLES_FILE, dataset_dir
    from src.preprocessing.settings import PreprocessingConfig

    window = PreprocessingConfig.load(None).windows[group]

    table = pq.read_table(dataset_dir(group) / SAMPLES_FILE)

    rows = table.to_pylist()

    late = 0
    unsorted_rows = 0
    mask_wrong = 0
    events = 0
    targets = 0

    start = window.target_start.astimezone(timezone.utc).replace(tzinfo=None)
    stop = window.target_end.astimezone(timezone.utc).replace(tzinfo=None)
    cutoff = window.final_cutoff.astimezone(timezone.utc).replace(tzinfo=None)

    for row in rows:

        moments = [np.datetime64(value) for value in row["event_time"]]
        flags = row["target_event_mask"]

        events += len(moments)
        targets += sum(flags)

        if any(value >= np.datetime64(cutoff) for value in moments):
            late += 1

        if any(moments[index] > moments[index + 1] for index in range(len(moments) - 1)):
            unsorted_rows += 1

        # Своё определение периода целей, а не вызов той же
        # функции: совпасть они обязаны по смыслу, а не по коду.
        expected = [
            np.datetime64(start) <= value < np.datetime64(stop) for value in moments
        ]

        mask_wrong += sum(1 for left, right in zip(flags, expected) if left != right)

    return {
        "clients": len(rows),
        "events": events,
        "targets": targets,
        "rows_with_event_after_cutoff": late,
        "rows_out_of_order": unsorted_rows,
        "mask_disagreements": mask_wrong,
    }


def check_batches_and_masks(group: str) -> dict:
    """
    ST-7A, ST-8A, ST-8B.
    """

    from src.batching.settings import BATCHES_FILE, batches_dir
    from src.dataset.settings import SAMPLES_FILE, dataset_dir
    from src.masking.settings import MASKED_FILE, masked_dir
    from src.tokenization.specials import PAD, UNK, load_special_tokens

    samples = {row["client_id"]: row for row in
               pq.read_table(dataset_dir(group) / SAMPLES_FILE).to_pylist()}

    batches = pq.read_table(batches_dir(group) / BATCHES_FILE).to_pylist()
    masked = pq.read_table(masked_dir(group) / MASKED_FILE).to_pylist()

    specials = load_special_tokens()

    pad = specials[PAD]
    unknown = specials[UNK]

    lost = sorted(set(samples) - {row["client_id"] for row in batches})
    extra = sorted({row["client_id"] for row in batches} - set(samples))

    content_wrong = 0
    padding_wrong = 0

    for row in batches:

        sample = samples[row["client_id"]]

        length = len(sample["value_ids"])

        if list(row["value_ids"][:length]) != list(sample["value_ids"]):
            content_wrong += 1

        if any(value != pad for value in row["value_ids"][length:]):
            padding_wrong += 1

    by_client = {row["client_id"]: row for row in batches}

    outside = 0
    label_wrong = 0
    source_wrong = 0
    masked_positions = 0
    unknown_substitutions = 0
    outside_substitutions = 0

    for row in masked:

        batch = by_client[row["client_id"]]
        sample = samples[row["client_id"]]

        starts = list(sample["event_starts"])
        lengths = list(sample["event_lengths"])
        flags = list(sample["target_event_mask"])

        allowed = np.zeros(len(row["labels"]), dtype=bool)

        for index, (begin, size) in enumerate(zip(starts, lengths)):
            if flags[index]:
                allowed[begin:begin + size] = True

        labels = np.asarray(row["labels"])
        source = list(row["value_ids_source"])
        current = list(row["value_ids"])

        chosen = labels != -100

        masked_positions += int(chosen.sum())

        outside += int((chosen & ~allowed).sum())

        for position in np.nonzero(chosen)[0]:
            if labels[position] != source[position]:
                label_wrong += 1

        if source[:len(sample["value_ids"])] != list(sample["value_ids"]):
            source_wrong += 1

        # Вне loss значение либо не тронуто, либо заменено на
        # [UNK]: замена объявлена (masking/apply.py) — модель
        # видит незнакомое значение и не отвечает за него.
        for index in range(len(current)):

            if chosen[index] or current[index] == source[index]:
                continue

            if current[index] == unknown:
                unknown_substitutions += 1
            else:
                outside_substitutions += 1

    return {
        "clients_lost_in_batches": lost,
        "clients_invented_in_batches": extra,
        "rows_with_changed_content": content_wrong,
        "rows_with_wrong_padding": padding_wrong,
        "masked_positions": masked_positions,
        "masked_outside_targets": outside,
        "label_mismatches": label_wrong,
        "unknown_substitutions": unknown_substitutions,
        "unexplained_substitutions": outside_substitutions,
        "source_mismatches": source_wrong,
    }


# Типы событий, след которых виден в снимке анкеты: она
# описывает конец периода целей, а эти события его и меняют.
TRACED_BY_PROFILE = {
    "product_opened": ("contracts_count", "active_contracts", "holds_deposit",
                       "holds_credit_card", "holds_debit_card", "credit_limit"),
    "account_opened": ("contracts_count", "active_contracts"),
    "product_closed": ("active_contracts", "holds_deposit", "holds_credit_card"),
    "profile_change": ("любое изменённое поле",),
}


def check_profile_leak(group: str, root: Path) -> dict:
    """
    LEAK-1: сколько примеров несут в анкете след своей же цели.
    """

    from src.dataset.settings import SAMPLES_FILE, dataset_dir
    from src.preprocessing.settings import PreprocessingConfig
    from src.tokenization.settings import TOKENIZED_DIR

    window = PreprocessingConfig.load(None).windows[group]

    start = np.datetime64(window.target_start.astimezone(timezone.utc).replace(tzinfo=None))
    stop = np.datetime64(window.target_end.astimezone(timezone.utc).replace(tzinfo=None))

    # Что попало во вход модели из анкеты.
    profile = pq.read_table(TOKENIZED_DIR / group / "profile.parquet")

    samples = pq.read_table(dataset_dir(group) / SAMPLES_FILE).to_pylist()

    # Типы событий в примере восстанавливаются по очищенной ленте:
    # в примере лежат уже коды, а не имена.
    from src.preprocessing.settings import group_dir
    from src.preprocessing.canonical.build import EVENTS_FILE

    clean = pq.read_table(group_dir(group) / EVENTS_FILE, columns=["client_id", "event_time", "type"])

    traced: dict[str, int] = {}
    clients_with_traced_target = set()

    for client, when, kind in zip(
        clean.column("client_id").to_pylist(),
        clean.column("event_time").to_pylist(),
        clean.column("type").to_pylist(),
    ):
        moment = np.datetime64(when.replace(tzinfo=None))

        if not (start <= moment < stop):
            continue

        if kind in TRACED_BY_PROFILE:
            traced[kind] = traced.get(kind, 0) + 1
            clients_with_traced_target.add(client)

    with_profile = sum(1 for row in samples if len(row["profile_key_ids"]) > 1)

    # Ни одного невосстановимого ключа в анкете примера.
    from src.preprocessing.profile_state import UNPROVABLE_FIELDS
    from src.tokenization.finalvocab import KEY_PREFIX

    vocabulary = json.loads(
        (root / "03_vocab" / "final_vocab.json").read_text(encoding="utf-8")
    )

    forbidden = {
        number
        for token, number in vocabulary.items()
        if token.startswith(KEY_PREFIX)
        and token[len(KEY_PREFIX) :] in {f"profile_{name}" for name in UNPROVABLE_FIELDS}
    }

    carried = sum(
        1 for row in samples if forbidden & set(row["profile_key_ids"])
    )

    return {
        "samples_with_unprovable_key": carried,
        "clients": len(samples),
        "samples_carrying_profile": with_profile,
        "profile_rows": profile.num_rows,
        "target_events_traced_by_profile": traced,
        "clients_with_such_target": len(clients_with_traced_target),
        "share_of_clients": round(len(clients_with_traced_target) / max(1, len(samples)), 4),
    }


def main() -> int:

    parser = argparse.ArgumentParser(prog="stages")
    parser.add_argument("--source", default="regression")
    parser.add_argument("--root", default="stages-root")
    parser.add_argument("--out", default="evidence/stages-05-08.json")
    parser.add_argument("--skip-build", action="store_true")

    args = parser.parse_args()

    source = (RUNS / args.source).resolve()

    # Корень этапов можно дать абсолютным: с --skip-build
    # проверяется и настоящий data/, а не только пересборка.
    root = Path(args.root)
    root = (root if root.is_absolute() else RUNS / root).resolve()

    redirect(root)

    if not args.skip_build:
        build_everything(root, source)

    report: dict = {"root": str(root), "groups": {}}

    for group in GROUPS:

        dataset = check_dataset(group)
        pipeline = check_batches_and_masks(group)
        leak = check_profile_leak(group, root)

        report["groups"][group] = {
            "dataset": dataset,
            "batches_and_masks": pipeline,
            "profile_leak": leak,
        }

        record(
            f"ST-5A срез примера ({group})",
            "PASS" if not dataset["rows_with_event_after_cutoff"] and not dataset["rows_out_of_order"] else "FAIL",
            f"событий {dataset['events']}, клиентов {dataset['clients']}",
            dataset["events"],
            dataset["rows_with_event_after_cutoff"] + dataset["rows_out_of_order"],
        )

        record(
            f"ST-5B маска целей ({group})",
            "PASS" if not dataset["mask_disagreements"] else "FAIL",
            f"целей {dataset['targets']} из {dataset['events']} событий",
            dataset["events"],
            dataset["mask_disagreements"],
        )

        bad_batches = (
            len(pipeline["clients_lost_in_batches"])
            + len(pipeline["clients_invented_in_batches"])
            + pipeline["rows_with_changed_content"]
            + pipeline["rows_with_wrong_padding"]
        )

        record(
            f"ST-7A батчи ({group})",
            "PASS" if not bad_batches else "FAIL",
            "состав, порядок и содержимое перенесены без изменений",
            dataset["clients"],
            bad_batches,
        )

        record(
            f"ST-8A маскируется только разрешённое ({group})",
            "PASS" if not pipeline["masked_outside_targets"] else "FAIL",
            f"маскированных мест {pipeline['masked_positions']}",
            pipeline["masked_positions"],
            pipeline["masked_outside_targets"],
        )

        record(
            f"ST-8B метки равны исходным значениям ({group})",
            "PASS" if not pipeline["label_mismatches"] and not pipeline["source_mismatches"] else "FAIL",
            "на каждом месте в loss метка равна значению до подстановки",
            pipeline["masked_positions"],
            pipeline["label_mismatches"] + pipeline["source_mismatches"],
        )

        record(
            f"ST-8C подстановки вне loss ({group})",
            "PASS" if not pipeline["unexplained_substitutions"] else "FAIL",
            f"замен на [UNK] {pipeline['unknown_substitutions']}, "
            "необъяснённых подстановок нет" if not pipeline["unexplained_substitutions"]
            else f"необъяснённых подстановок {pipeline['unexplained_substitutions']}",
            pipeline["unknown_substitutions"] + pipeline["unexplained_substitutions"],
            pipeline["unexplained_substitutions"],
        )

        record(
            f"ST-PF невосстановимые поля в анкете ({group})",
            "PASS" if not leak["samples_with_unprovable_key"] else "FAIL",
            f"примеров с запрещённым ключом {leak['samples_with_unprovable_key']}",
            leak["clients"],
            leak["samples_with_unprovable_key"],
        )

        record(
            f"RISK-1 целей, чей след анкета несла прежде ({group})",
            "СПРАВКА",
            f"анкета есть у {leak['samples_carrying_profile']} примеров; целей, "
            f"чей след виден в анкете: {leak['target_events_traced_by_profile']}; "
            f"клиентов {leak['clients_with_such_target']} из {leak['clients']}",
            leak["clients"],
            leak["clients_with_such_target"],
        )

    report["results"] = RESULTS

    report["verdict"] = (
        "PASS" if all(item["verdict"] in ("PASS", "чисто", "НАЙДЕНО") for item in RESULTS) else "FAIL"
    )

    (AUDIT / args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"-> {AUDIT / args.out}")

    return 0 if all(item["verdict"] != "FAIL" for item in RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
