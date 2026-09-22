from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .artifacts import sha256_file, write_json, write_text
from .manifest import (
    fingerprint_path,
    load_fingerprint,
    output_digests,
    outputs_intact,
    save_fingerprint,
    stage_entry,
    stage_fingerprint,
    update_manifest,
)
from .manifest import MANIFEST_FILE
from .rawdata import MANIFEST_NAME
from .settings import GROUPS, PreprocessingConfig, normalize_group, processed_dir


# ============================================================
# ИДЕЯ
# ============================================================
#
# Точка входа препроцессинга: этап за этапом, каждый со своим
# отпечатком.
#
# Отпечаток считается по ФАКТИЧЕСКИМ файлам, а не по числам из
# manifest.json. Манифест это утверждение выгрузки о себе, и
# доверять ему при решении «можно ли пропустить проверку» нельзя:
# подменённый events.parquet оставил бы прежний отпечаток и
# протащил бы повреждение мимо этапа.
#
# Пропуск разрешён только когда совпал отпечаток входов И на
# месте лежат целые выходные файлы: удалённый или правленый
# отчёт восстанавливается, а не считается существующим.
#
# RAW задаётся либо одним каталогом с указанием группы, либо
# корнем с подкаталогами train/val/test.
#
# Выход: data/processed/<name>/. Ни путей, ни времени запуска в
# артефактах.
# ============================================================


# Коды возврата: отличают структурную поломку от расхождения с
# контрактом, чтобы вызывающий скрипт мог реагировать по-разному.
EXIT_OK = 0
EXIT_BLOCKED = 2
EXIT_CONTRACT_MISMATCH = 3

# Тяжесть вердикта, а не величина кода: у нескольких групп
# итогом обязан быть самый тяжёлый случай, а blocked (2)
# численно меньше contract_mismatch (3).
EXIT_SEVERITY: dict[int, int] = {
    EXIT_OK: 0,
    EXIT_CONTRACT_MISMATCH: 1,
    EXIT_BLOCKED: 2,
}


def clean_directory(path: Path) -> None:

    if not path.exists():
        return

    for item in sorted(path.rglob("*"), reverse=True):
        if item.is_file():
            item.unlink()
        else:
            item.rmdir()


# ============================================================
# ГРУППЫ
# ============================================================


def resolve_groups(args) -> list[tuple[str | None, Path]]:
    """
    Пары (группа, каталог RAW) из аргументов командной строки.
    """

    if args.raw is not None:
        group = normalize_group(args.group) if args.group else None
        return [(group, Path(args.raw))]

    root = Path(args.raw_root)

    pairs: list[tuple[str | None, Path]] = []

    for group in GROUPS:
        candidates = [root / group] + ([root / "validation"] if group == "val" else [])
        found = next((path for path in candidates if path.exists()), None)
        if found is not None:
            pairs.append((group, found))

    if not pairs:
        raise SystemExit(f"в {root} нет подкаталогов {GROUPS}")

    return pairs


def dataset_name(args) -> str:

    if args.name:
        return args.name

    if getattr(args, "raw", None) is not None:
        return Path(args.raw).resolve().name

    root = getattr(args, "raw_root", None)

    if root is None:
        raise SystemExit("нужно имя набора: укажите --name")

    return Path(root).resolve().name


# ============================================================
# ОТПЕЧАТОК ВХОДОВ И ЦЕЛОСТНОСТЬ ВЫХОДОВ
# ============================================================


def raw_inputs(raw_dir: Path) -> dict[str, str]:
    """
    sha256 фактических файлов RAW: манифест и все parquet-таблицы.

    Считается по содержимому на диске: именно это отличает
    «данные те же» от «манифест говорит, что те же». Отчёты и
    прочие файлы рядом с выгрузкой не входят: паспорт их не
    читает, и перегенерация отчёта не должна пересчитывать этап.
    """

    raw_dir = Path(raw_dir)

    if not raw_dir.exists():
        return {"<missing>": raw_dir.name}

    inputs: dict[str, str] = {}

    manifest = raw_dir / MANIFEST_NAME

    if manifest.exists():
        inputs[MANIFEST_NAME] = sha256_file(manifest)

    for path in sorted(raw_dir.rglob("*.parquet")):

        name = path.relative_to(raw_dir).as_posix()

        inputs[name] = sha256_file(path)

    return inputs


def can_skip(stored: dict | None, fingerprint: dict, root: Path) -> bool:
    """
    Результат этапа можно переиспользовать: совпал отпечаток
    входов, выходные файлы целы и маркер знает, что записать в
    общий манифест.
    """

    return (
        stored is not None
        and stored.get("fingerprint") == fingerprint["fingerprint"]
        and stored.get("manifest_entry") is not None
        and outputs_intact(stored, root)
    )


def restore_manifest_entry(root: Path, stage: str, group: str | None, stored: dict) -> bool:
    """
    Возвращает запись этапа в preprocessing_manifest.json, если
    её там нет или она разошлась с маркером. Общий манифест это
    тоже результат этапа: удалённый, он не должен оставаться
    пустым только потому, что отчёты на месте.
    """

    entry = stored["manifest_entry"]

    if stage_entry(root, stage, group) == entry:
        return False

    update_manifest(root, stage, group, entry)

    return True


def worst(first: int, second: int) -> int:
    """
    Более тяжёлый из двух вердиктов.
    """

    return first if EXIT_SEVERITY[first] >= EXIT_SEVERITY[second] else second


def exit_code_for(status: str, allow_contract_mismatch: bool) -> int:

    if status == "blocked":
        return EXIT_BLOCKED

    if status == "contract_mismatch":
        return EXIT_OK if allow_contract_mismatch else EXIT_CONTRACT_MISMATCH

    return EXIT_OK


# ============================================================
# ЭТАП 1
# ============================================================


def run_passport(args) -> int:

    from .passport import STAGE, STAGE_VERSION, build_passport, render_passport_md

    config = PreprocessingConfig.load(Path(args.config) if args.config else None)

    out_root = Path(args.out) if args.out else processed_dir(dataset_name(args))

    exit_code = EXIT_OK

    for group, raw_dir in resolve_groups(args):

        label = group or "—"

        fingerprint = stage_fingerprint(STAGE, STAGE_VERSION, raw_inputs(raw_dir), config.section(STAGE))
        marker = fingerprint_path(out_root, STAGE, group)
        stored = load_fingerprint(marker)

        if not args.force and can_skip(stored, fingerprint, out_root):
            status = stored.get("status", "unknown")
            print(f"[{STAGE}] группа {label}: входы и выходы не изменились, этап пропущен (статус {status})")
            if restore_manifest_entry(out_root, STAGE, group, stored):
                print(f"    запись этапа восстановлена в {MANIFEST_FILE}")
            exit_code = worst(exit_code, exit_code_for(status, args.allow_contract_mismatch))
            continue

        report = build_passport(raw_dir, config, group)

        target = out_root / "passport" / f"{group or 'raw'}"

        json_path = target.with_suffix(".json")
        md_path = target.with_suffix(".md")

        write_json(json_path, report)
        write_text(md_path, render_passport_md(report))

        fingerprint["outputs"] = output_digests([json_path, md_path], out_root)
        fingerprint["status"] = report["status"]

        entry = {
            "status": report["status"],
            "usable": report.get("usable", False),
            "stage_version": STAGE_VERSION,
            "fingerprint": fingerprint["fingerprint"],
            "errors": len(report.get("errors", [])),
            "contract_violations": len(report.get("contract_violations", [])),
            "limitations": len(report.get("limitations", [])),
            "config_sha256": fingerprint["config_sha256"],
        }

        # Маркер держит запись целиком: пропущенный этап обязан
        # уметь восстановить её без повторного счёта.
        fingerprint["manifest_entry"] = entry

        save_fingerprint(marker, fingerprint)

        update_manifest(out_root, STAGE, group, entry)

        print(f"[{STAGE}] группа {label}: статус {report['status']} → {md_path}")

        for item in report.get("errors", []):
            print(f"    ошибка структуры: {item}")

        for item in report.get("contract_violations", []):
            print(f"    нарушение контракта: {item}")

        if "events" in report:
            events = report["events"]
            print(
                f"    клиентов {events['clients']}, событий {events['rows']}, "
                f"event_time {events['event_time']['min']} … {events['event_time']['max']}"
            )

        for item in report.get("limitations", []):
            print(f"    ограничение: {item}")

        exit_code = worst(exit_code, exit_code_for(report["status"], args.allow_contract_mismatch))

    if exit_code == EXIT_CONTRACT_MISMATCH:
        print(
            "\nВыгрузка расходится с собственным каталогом ключей. Следующие этапы на ней "
            "запускать нельзя; для разработки используйте --allow-contract-mismatch: "
            "статус в отчёте останется contract_mismatch."
        )

    return exit_code


# ============================================================
# ЭТАП 2
# ============================================================


def passport_gate(
    out_root: Path,
    group: str | None,
    raw_dir: Path,
    allow_contract_mismatch: bool,
) -> tuple[dict | None, str | None, int]:
    """
    Этап 2 опирается на вердикт этапа 1, и вердикт обязан
    описывать ТЕ ЖЕ файлы.

    Паспорт, снятый с прежней выгрузки, ничего не говорит о
    нынешней: RAW мог измениться после него, и тогда canonical
    собрался бы на новых данных под старым вердиктом.
    """

    from .passport import STAGE as PASSPORT_STAGE

    entry = stage_entry(out_root, PASSPORT_STAGE, group)

    if entry is None:
        return None, "паспорт группы не построен: сначала выполните этап passport", EXIT_BLOCKED

    marker = load_fingerprint(fingerprint_path(out_root, PASSPORT_STAGE, group))

    if marker is None or marker.get("fingerprint") != entry.get("fingerprint"):
        return entry, (
            "маркер паспорта отсутствует или разошёлся с общим манифестом: "
            "выполните этап passport заново"
        ), EXIT_BLOCKED

    if marker.get("inputs") != raw_inputs(raw_dir):
        return entry, (
            "RAW изменился после паспорта: вердикт этапа 1 описывает другие файлы, "
            "выполните этап passport заново"
        ), EXIT_BLOCKED

    status = entry.get("status")

    if status == "blocked":
        return entry, "паспорт группы заблокирован: выгрузка повреждена", EXIT_BLOCKED

    if status == "contract_mismatch" and not allow_contract_mismatch:
        return entry, (
            "паспорт группы contract_mismatch: запустите с --allow-contract-mismatch, "
            "чтобы строить canonical в режиме диагностики"
        ), EXIT_CONTRACT_MISMATCH

    return entry, None, EXIT_OK


def run_canonical(args) -> int:

    from .canonical.build import STAGE, STAGE_VERSION, build_group
    from .canonical.events import CanonicalError

    config = PreprocessingConfig.load(Path(args.config) if args.config else None)

    out_root = Path(args.out) if args.out else processed_dir(dataset_name(args))

    exit_code = EXIT_OK

    for group, raw_dir in resolve_groups(args):

        label = group or "—"

        passport, refusal, refusal_code = passport_gate(
            out_root, group, raw_dir, args.allow_contract_mismatch
        )

        if refusal is not None:
            print(f"[{STAGE}] группа {label}: {refusal}")
            exit_code = worst(exit_code, refusal_code)
            continue

        inputs = raw_inputs(raw_dir)
        inputs["stage:passport"] = passport["fingerprint"]

        fingerprint = stage_fingerprint(STAGE, STAGE_VERSION, inputs, config.section(STAGE))
        marker = fingerprint_path(out_root, STAGE, group)
        stored = load_fingerprint(marker)

        if not args.force and can_skip(stored, fingerprint, out_root):
            print(f"[{STAGE}] группа {label}: входы и выходы не изменились, этап пропущен")
            if restore_manifest_entry(out_root, STAGE, group, stored):
                print(f"    запись этапа восстановлена в {MANIFEST_FILE}")
            continue

        target = out_root / STAGE / (group or "raw")

        clean_directory(target)

        try:
            result = build_group(raw_dir, target, config, group)
        except CanonicalError as error:
            # Необрабатываемый вход: этап останавливается с
            # названием поля и строки, а не роняет трассировку.
            print(f"[{STAGE}] группа {label}: {error}")
            clean_directory(target)
            exit_code = worst(exit_code, EXIT_BLOCKED)
            continue

        report = result.report

        md_path = target / "canonical_report.md"

        fingerprint["outputs"] = output_digests(result.outputs, out_root)
        fingerprint["status"] = report["status"]

        entry = {
            "status": report["status"],
            "stage_version": STAGE_VERSION,
            "schema_version": report["schema_version"],
            "fingerprint": fingerprint["fingerprint"],
            "passport_status": passport.get("status"),
            "diagnostic_mode": bool(args.allow_contract_mismatch and passport.get("status") == "contract_mismatch"),
            "rows": report["rows"]["canonical_events"],
            "repeated_ids": report["repeats"]["rows"],
            "rejects": report["rows"]["rejects"],
            "config_sha256": fingerprint["config_sha256"],
            "registry_digest": report["registry"]["digest"],
        }

        fingerprint["manifest_entry"] = entry

        save_fingerprint(marker, fingerprint)
        update_manifest(out_root, STAGE, group, entry)

        rows = report["rows"]

        print(f"[{STAGE}] группа {label}: статус {report['status']} → {md_path}")
        print(
            f"    строк RAW {rows['raw_events']} → canonical {rows['canonical_events']}, "
            f"повторов event_id {rows['repeated_ids']}, неразобранных {rows['rejects']}"
        )
        print(
            f"    упоминаний сущностей {rows['mentions']}, сторон переводов {rows['transfer_sides']}, "
            f"клиентов {rows['clients']} (без событий {rows['clients_without_events']})"
        )

        if report["status"] != "ok":
            exit_code = worst(exit_code, EXIT_BLOCKED)

    return exit_code


# ============================================================
# ЭТАП 3
# ============================================================


def canonical_gate(out_root: Path, group: str | None) -> tuple[dict | None, str | None]:
    """
    Этап 3 читает canonical, поэтому он обязан быть на месте и
    совпадать со своим маркером: правленый или недописанный слой
    отвечал бы на вопрос о другой выгрузке.
    """

    from .canonical.build import STAGE as CANONICAL_STAGE

    entry = stage_entry(out_root, CANONICAL_STAGE, group)

    if entry is None:
        return None, "canonical группы не построен: сначала выполните этап canonical"

    marker = load_fingerprint(fingerprint_path(out_root, CANONICAL_STAGE, group))

    if marker is None or marker.get("fingerprint") != entry.get("fingerprint"):
        return entry, "маркер canonical отсутствует или разошёлся с общим манифестом: выполните этап canonical заново"

    if not outputs_intact(marker, out_root):
        return entry, "файлы canonical изменились после сборки: выполните этап canonical заново"

    return entry, None


def canonical_source_gate(out_root: Path, group: str, raw_dir: Path) -> tuple[dict | None, str | None]:
    """
    Canonical группы на месте, цел и собран ИМЕННО из этой
    выгрузки: иначе разделение закрепило бы клиентов одной
    выгрузки под содержимым другой.
    """

    from .canonical.build import STAGE as CANONICAL_STAGE

    entry, refusal = canonical_gate(out_root, group)

    if refusal is not None:
        return entry, refusal

    marker = load_fingerprint(fingerprint_path(out_root, CANONICAL_STAGE, group))

    stored = {name: value for name, value in marker.get("inputs", {}).items() if not name.startswith("stage:")}

    if stored != raw_inputs(raw_dir):
        return entry, "canonical собран на других файлах RAW: выполните этап canonical заново"

    return entry, None



def resolve_cutoffs(args, config: PreprocessingConfig, group: str | None, period_end) -> list:

    from datetime import datetime

    from .temporal import choose_cutoffs

    if args.cutoff:
        return sorted(datetime.fromisoformat(value) for value in args.cutoff)

    window = config.windows.get(group or "train")

    return choose_cutoffs(
        window.history_start,
        window.final_cutoff,
        period_end,
        config.history_report_cutoffs,
    )


def run_history(args) -> int:

    from .canonical.build import STAGE as CANONICAL_STAGE
    from .history import STAGE, STAGE_VERSION, CanonicalStore, HistoryError, history_as_of
    from .temporal import render_history_md, render_temporal_md, temporal_report

    config = PreprocessingConfig.load(Path(args.config) if args.config else None)

    out_root = Path(args.out) if args.out else processed_dir(dataset_name(args))

    group = normalize_group(args.group) if args.group else None

    raw_dir = Path(args.raw) if args.raw else None

    if raw_dir is not None:
        # Справочник должен описывать ТУ ЖЕ выгрузку, из которой
        # собран canonical.
        canonical, refusal = canonical_source_gate(out_root, group or "train", raw_dir)
    else:
        canonical, refusal = canonical_gate(out_root, group)

    if refusal is not None:
        print(f"[{STAGE}] группа {group or '—'}: {refusal}")
        return EXIT_BLOCKED

    canonical_dir = out_root / CANONICAL_STAGE / (group or "raw")

    store = CanonicalStore(canonical_dir)

    cutoffs = resolve_cutoffs(args, config, group, store.period_end)

    marker = load_fingerprint(fingerprint_path(out_root, CANONICAL_STAGE, group))

    inputs = dict(marker.get("outputs", {}))
    inputs["stage:canonical"] = canonical["fingerprint"]

    section = dict(config.section(STAGE))
    section["cutoffs"] = [moment.isoformat() for moment in cutoffs]

    fingerprint = stage_fingerprint(STAGE, STAGE_VERSION, inputs, section)
    own_marker = fingerprint_path(out_root, STAGE, group)
    stored = load_fingerprint(own_marker)

    target = out_root / STAGE / (group or "raw")

    if not args.force and not args.client and can_skip(stored, fingerprint, out_root):
        print(f"[{STAGE}] группа {group or '—'}: входы и выходы не изменились, этап пропущен")
        if restore_manifest_entry(out_root, STAGE, group, stored):
            print(f"    запись этапа восстановлена в {MANIFEST_FILE}")
        return EXIT_OK

    # --- читаемые истории выбранного клиента ---

    examples: list[Path] = []

    if args.client:
        wanted = [int(args.client) if args.client.isdigit() else args.client]
    else:
        wanted = [row["client_id"] for row in store.clients[:1]]

    for client in wanted:
        for cutoff in cutoffs:
            try:
                history = history_as_of(store, client, cutoff)
            except HistoryError as error:
                print(f"    {error}")
                return EXIT_BLOCKED

            path = target / "examples" / f"{history.client_id}__{cutoff.date()}.md"
            write_text(path, render_history_md(history))
            examples.append(path)

            print(
                f"    {history.client_id} на {cutoff.date()}: видно {history.counts['visible']} из "
                f"{history.counts['rows']}, профиль {history.profile_meta['state']}, "
                f"сущностей {len(history.entities)}, переводов {len(history.transfers)}"
            )

    # --- отчёт о правилах ---

    sample = [row["client_id"] for row in store.clients[: config.history_sample_clients]]

    report = temporal_report(store, sample, cutoffs, group)

    json_path = target / "temporal_report.json"
    md_path = target / "temporal_report.md"

    write_json(json_path, report)
    write_text(md_path, render_temporal_md(report))

    fingerprint["outputs"] = output_digests([json_path, md_path], out_root)
    fingerprint["status"] = report["status"]

    entry = {
        "status": report["status"],
        "stage_version": STAGE_VERSION,
        "fingerprint": fingerprint["fingerprint"],
        "clients_checked": report["clients_checked"],
        "cutoffs": report["cutoffs"],
        "problems": report["problem_count"],
        "config_sha256": fingerprint["config_sha256"],
    }

    fingerprint["manifest_entry"] = entry

    save_fingerprint(own_marker, fingerprint)
    update_manifest(out_root, STAGE, group, entry)

    print(f"[{STAGE}] группа {group or '—'}: статус {report['status']} → {md_path}")
    print(
        f"    проверено клиентов {report['clients_checked']} на срезах "
        f"{', '.join(moment[:10] for moment in report['cutoffs'])}; нарушений {report['problem_count']}"
    )

    for item in report["problems"][:5]:
        print(f"    нарушение: {item['client_id']} на {item['cutoff']}: {item['problem']}")

    return EXIT_OK if report["status"] == "ok" else EXIT_BLOCKED


# ============================================================
# ЭТАП 4
# ============================================================


def run_corpus(args) -> int:

    from .canonical.build import STAGE as CANONICAL_STAGE
    from .history import STAGE_VERSION as HISTORY_VERSION
    from .corpus import (
        STAGE,
        STAGE_VERSION,
        STATUS_BLOCKED,
        STATUS_BLOCKED_BY_INPUT,
        GroupInput,
        build_corpus,
    )

    config = PreprocessingConfig.load(Path(args.config) if args.config else None)

    out_root = Path(args.out) if args.out else processed_dir(dataset_name(args))

    sources: list[GroupInput] = []
    inputs: dict[str, str] = {"code:history": HISTORY_VERSION}

    for group, raw_dir in resolve_groups(args):

        canonical, refusal = canonical_source_gate(out_root, group, raw_dir)

        if refusal is not None:
            print(f"[{STAGE}] группа {group}: {refusal}")
            return EXIT_BLOCKED

        sources.append(
            GroupInput(
                group=group,
                raw_dir=raw_dir,
                canonical_dir=out_root / CANONICAL_STAGE / group,
                canonical_fingerprint=canonical["fingerprint"],
                passport_status=canonical.get("passport_status"),
                diagnostic_mode=bool(canonical.get("diagnostic_mode")),
            )
        )

        inputs[f"stage:canonical:{group}"] = canonical["fingerprint"]

    fingerprint = stage_fingerprint(STAGE, STAGE_VERSION, inputs, config.section(STAGE))
    marker = fingerprint_path(out_root, STAGE, None)
    stored = load_fingerprint(marker)

    target = out_root / STAGE

    if not args.force and can_skip(stored, fingerprint, out_root):
        status = stored.get("status", "unknown")
        print(f"[{STAGE}] входы и выходы не изменились, этап пропущен (статус {status})")
        if restore_manifest_entry(out_root, STAGE, None, stored):
            print(f"    запись этапа восстановлена в {MANIFEST_FILE}")
        return corpus_exit_code(status, args.allow_input_mismatch)

    clean_directory(target)

    result = build_corpus(sources, config, target)

    report = result.report

    fingerprint["outputs"] = output_digests(result.outputs, out_root)
    fingerprint["status"] = report["status"]

    entry = {
        "status": report["status"],
        "usable": report["usable"],
        "stage_version": STAGE_VERSION,
        "schema_version": report["schema_version"],
        "fingerprint": fingerprint["fingerprint"],
        "shared_world": report["shared_world"],
        "groups": {name: item["clients_working"] for name, item in report["groups"].items()},
        "train_rows": (report["train_corpus"] or {}).get("rows"),
        "train_content_sha256": ((report["train_corpus"] or {}).get("checksum") or {}).get("content_sha256"),
        "errors": len(report["errors"]),
        "input_dependencies": len(report["input_dependencies"]),
        "config_sha256": fingerprint["config_sha256"],
    }

    fingerprint["manifest_entry"] = entry

    save_fingerprint(marker, fingerprint)
    update_manifest(out_root, STAGE, None, entry)

    print(f"[{STAGE}] статус {report['status']} → {target / 'corpus_report.md'}")

    for name, item in sorted(report["groups"].items()):
        print(
            f"    {name}: клиентов {item['clients_working']}, "
            f"видно событий {item['visible_events']} "
            f"на {item['window']['final_cutoff'][:10]}, в периоде целей {item['eligible_events']}"
        )

    corpus = report["train_corpus"]

    if corpus is not None:
        print(
            f"    train-интерфейс на {corpus['fit_end'][:10]}: строк {corpus['rows']}, "
            f"сумма содержимого {corpus['checksum']['content_sha256'][:16]}"
        )

    for item in report["errors"]:
        print(f"    ошибка: {item}")

    for item in report["input_dependencies"]:
        print(f"    входная зависимость: {item}")

    for item in report["limitations"]:
        print(f"    ограничение: {item}")

    if report["status"] == STATUS_BLOCKED_BY_INPUT and not args.allow_input_mismatch:
        print(
            "\nГруппы не делят один мир либо не независимы. Артефакты записаны как диагностика; "
            "для разработки следующих этапов используйте --allow-input-mismatch: "
            f"статус в отчёте останется {STATUS_BLOCKED_BY_INPUT}."
        )

    return corpus_exit_code(report["status"], args.allow_input_mismatch)


def corpus_exit_code(status: str, allow_input_mismatch: bool) -> int:

    from .corpus import STATUS_BLOCKED, STATUS_BLOCKED_BY_INPUT

    if status == STATUS_BLOCKED:
        return EXIT_BLOCKED

    if status == STATUS_BLOCKED_BY_INPUT:
        return EXIT_OK if allow_input_mismatch else EXIT_CONTRACT_MISMATCH

    return EXIT_OK


# ============================================================
# ЭТАП 5
# ============================================================


def run_semantic(args) -> int:

    from datetime import datetime

    from .canonical.build import STAGE as CANONICAL_STAGE
    from .history import STAGE_VERSION as HISTORY_VERSION
    from .semantic.build import STAGE, STAGE_VERSION, SemanticError, build_group

    config = PreprocessingConfig.load(Path(args.config) if args.config else None)

    out_root = Path(args.out) if args.out else processed_dir(dataset_name(args))

    group = normalize_group(args.group) if args.group else None

    raw_dir = Path(args.raw) if args.raw else None

    if raw_dir is not None:
        canonical, refusal = canonical_source_gate(out_root, group or "train", raw_dir)
    else:
        canonical, refusal = canonical_gate(out_root, group)

    if refusal is not None:
        print(f"[{STAGE}] группа {group or '—'}: {refusal}")
        return EXIT_BLOCKED

    canonical_dir = out_root / CANONICAL_STAGE / (group or "raw")

    window = config.windows.get(group or "train")

    cutoff = datetime.fromisoformat(args.cutoff) if args.cutoff else window.final_cutoff

    inputs = {
        "stage:canonical": canonical["fingerprint"],
        "code:history": HISTORY_VERSION,
    }

    section = dict(config.section(STAGE))
    section["cutoff"] = cutoff.isoformat()

    fingerprint = stage_fingerprint(STAGE, STAGE_VERSION, inputs, section)
    marker = fingerprint_path(out_root, STAGE, group)
    stored = load_fingerprint(marker)

    target = out_root / STAGE / (group or "raw")

    if not args.force and not args.client and can_skip(stored, fingerprint, out_root):
        print(f"[{STAGE}] группа {group or '—'}: входы и выходы не изменились, этап пропущен")
        if restore_manifest_entry(out_root, STAGE, group, stored):
            print(f"    запись этапа восстановлена в {MANIFEST_FILE}")
        return EXIT_OK

    # Разбор одного клиента это ЧТЕНИЕ, а не сборка этапа: он не
    # чистит каталог, не переписывает маркер и не подменяет
    # запись в манифесте. Иначе состояние группы становилось бы
    # состоянием одного клиента, а следующий запуск считал бы
    # этап собранным.
    single = bool(args.client)

    if not single:
        clean_directory(target)

    try:
        result = build_group(
            canonical_dir=canonical_dir,
            target=target if not single else target / "examples",
            config=config,
            group=group,
            cutoff=cutoff,
            clients=[args.client] if single else None,
        )
    except SemanticError as error:
        print(f"[{STAGE}] группа {group or '—'}: {error}")
        if not single:
            clean_directory(target)
        return EXIT_BLOCKED

    report = result.report

    if single:
        print(
            f"[{STAGE}] группа {group or '—'}: клиент {args.client} на срезе "
            f"{report['cutoff'][:10]} → {target / 'examples'}"
        )
        print("    разбор одного клиента: состояние этапа не менялось")
        return EXIT_OK

    fingerprint["outputs"] = output_digests(result.outputs, out_root)
    fingerprint["status"] = "ok"

    entry = {
        "status": "ok",
        "stage_version": STAGE_VERSION,
        "schema_version": report["schema_version"],
        "keys_version": report["keys_version"],
        "fingerprint": fingerprint["fingerprint"],
        "cutoff": report["cutoff"],
        "clients_checked": report["clients_checked"],
        "keys_declared": report["keys_declared"],
        "keys_used": report["keys_used"],
        "config_sha256": fingerprint["config_sha256"],
    }

    fingerprint["manifest_entry"] = entry

    save_fingerprint(marker, fingerprint)
    update_manifest(out_root, STAGE, group, entry)

    print(f"[{STAGE}] группа {group or '—'}: срез {report['cutoff'][:10]} → {target / 'semantic_report.md'}")
    print(
        f"    ключей объявлено {report['keys_declared']}, встретилось {report['keys_used']}; "
        f"клиентов {report['clients_checked']}, событий {report['events']}, значений {report['values']}"
    )
    print(
        f"    активность по месяцам: {report['activity']['by_state']}; "
        f"цепочек {report['chains']['chains']} (незавершённых {report['chains']['unfinished']}), "
        f"связей {report['relations']}"
    )

    for item in report["limitations"][:5]:
        print(f"    ограничение: {item}")

    return EXIT_OK


# ============================================================
# CLI
# ============================================================


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="python -m src.preprocessing.run")

    subparsers = parser.add_subparsers(dest="stage", required=True)

    passport = subparsers.add_parser("passport", help="этап 1: проверка входных данных")

    source = passport.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw", type=Path, help="каталог одной RAW-группы")
    source.add_argument("--raw-root", type=Path, help="корень с подкаталогами train/val/test")

    passport.add_argument("--group", default=None, help="имя группы для --raw: train, val, test")
    passport.add_argument("--name", default=None, help="имя набора в data/processed; по умолчанию имя каталога RAW")
    passport.add_argument("--out", type=Path, default=None, help="каталог вывода вместо data/processed/<name>")
    passport.add_argument("--config", type=Path, default=None, help="JSON с переопределениями конфига")
    passport.add_argument("--force", action="store_true", help="пересчитать, даже если отпечаток совпал")
    passport.add_argument(
        "--allow-contract-mismatch",
        action="store_true",
        help="режим диагностики: не считать расхождение с каталогом ключей поводом для ненулевого кода возврата",
    )
    passport.set_defaults(handler=run_passport)

    canonical = subparsers.add_parser("canonical", help="этап 2: структуризация в canonical")

    source = canonical.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw", type=Path, help="каталог одной RAW-группы")
    source.add_argument("--raw-root", type=Path, help="корень с подкаталогами train/val/test")

    canonical.add_argument("--group", default=None, help="имя группы для --raw: train, val, test")
    canonical.add_argument("--name", default=None, help="имя набора в data/processed")
    canonical.add_argument("--out", type=Path, default=None, help="каталог вывода вместо data/processed/<name>")
    canonical.add_argument("--config", type=Path, default=None, help="JSON с переопределениями конфига")
    canonical.add_argument("--force", action="store_true", help="пересчитать, даже если отпечаток совпал")
    canonical.add_argument(
        "--allow-contract-mismatch",
        action="store_true",
        help="режим диагностики: строить canonical на выгрузке, расходящейся с каталогом ключей",
    )
    canonical.set_defaults(handler=run_canonical)

    history = subparsers.add_parser("history", help="этап 3: история на дату")

    history.add_argument("--name", default=None, help="имя набора в data/processed")
    history.add_argument("--group", default=None, help="имя группы: train, val, test")
    history.add_argument("--out", type=Path, default=None, help="каталог вывода вместо data/processed/<name>")
    history.add_argument("--raw", type=Path, default=None, help="каталог RAW: нужен только для справочника продуктов")
    history.add_argument("--client", default=None, help="клиент для читаемой истории: client_id или client_idx")
    history.add_argument(
        "--cutoff",
        action="append",
        default=None,
        help="момент среза в формате ISO; можно повторять. По умолчанию берутся срезы окна группы",
    )
    history.add_argument("--config", type=Path, default=None, help="JSON с переопределениями конфига")
    history.add_argument("--force", action="store_true", help="пересчитать, даже если отпечаток совпал")
    history.set_defaults(handler=run_history)

    corpus = subparsers.add_parser("corpus", help="этап 4: реестр групп и разрешённый корпус train")

    corpus.add_argument("--raw-root", type=Path, required=True, help="корень с подкаталогами train/val/test")
    corpus.add_argument("--name", default=None, help="имя набора в data/processed")
    corpus.add_argument("--out", type=Path, default=None, help="каталог вывода вместо data/processed/<name>")
    corpus.add_argument("--config", type=Path, default=None, help="JSON с переопределениями конфига")
    corpus.add_argument("--force", action="store_true", help="пересчитать, даже если отпечаток совпал")
    corpus.add_argument(
        "--allow-input-mismatch",
        action="store_true",
        help="режим диагностики: не считать общий мир или повторный seed поводом для ненулевого кода возврата",
    )
    corpus.set_defaults(handler=run_corpus, raw=None, group=None)

    semantic = subparsers.add_parser("semantic", help="этап 5: смысловой слой")

    semantic.add_argument("--name", default=None, help="имя набора в data/processed")
    semantic.add_argument("--group", default=None, help="имя группы: train, val, test")
    semantic.add_argument("--out", type=Path, default=None, help="каталог вывода вместо data/processed/<name>")
    semantic.add_argument("--raw", type=Path, default=None, help="каталог RAW: нужен для справочника мерчантов")
    semantic.add_argument("--client", default=None, help="клиент для читаемого примера")
    semantic.add_argument("--cutoff", default=None, help="момент среза ISO; по умолчанию конечный cutoff группы")
    semantic.add_argument("--config", type=Path, default=None, help="JSON с переопределениями конфига")
    semantic.add_argument("--force", action="store_true", help="пересчитать, даже если отпечаток совпал")
    semantic.set_defaults(handler=run_semantic)

    return parser


def main(argv: list[str] | None = None) -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)

    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
