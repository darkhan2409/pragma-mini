from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.preprocessing.artifacts import write_json
from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK

from .settings import EMBEDDINGS_META, FINAL_GROUPS, cutoff, downstream_dir, groups


# ============================================================
# ВЕКТОРЫ КЛИЕНТОВ НА МОМЕНТ T
# ============================================================
#
#   python -m src.downstream.embed --checkpoint data/runs/w4-b0/best_checkpoint.pt --tag w4-b0
#
# По умолчанию — train и val; test только в финальной оценке
# (--final-test), пока идут эксперименты его векторы не нужны.
#
# Для каждой группы — клиенты на её момент T (settings.cutoff),
# вход собран из прошлого (at_cutoff), один проход модели без
# градиента, четыре вектора на клиента (Model.readouts) и две
# величины о самом входе: число событий и давность последнего
# события до T. Сравнение на задачах берёт из них [USR] (usr).
#
# Клиент без событий до T пропускается: модель такого входа не
# видела, и ни одна из задач его не содержит. Сколько их — в
# meta.json.
# ============================================================


READOUTS = ("usr", "profile", "mean_event", "last_event")

# Процессов сборки входа по умолчанию: сборка идёт на CPU, по
# клиенту, и в одном процессе train занимал около 25 минут.
WORKERS = 3


def schema() -> pa.Schema:
    return pa.schema(
        [
            ("client_id", pa.string()),
            ("cutoff", pa.timestamp("us", tz="UTC")),
            ("n_events", pa.int32()),
            ("n_tokens", pa.int32()),
            ("gap_seconds", pa.float64()),
            *[(name, pa.list_(pa.float32())) for name in READOUTS],
        ]
    )


def trained_model(checkpoint: str, device):
    """
    Обученная модель и её описание.
    """

    from src.mlm.settings import MlmConfig
    from src.mlm.train import CheckpointError, data_record, load_trained

    model, state = load_trained(Path(checkpoint), device)

    # Вход на T строится из текущих данных: модель, обученная на
    # других (03–05 пересобраны под следующий эксперимент — другой
    # BPE, другая точка отсчёта времени), получила бы чужой вход.
    learned = state.get("data") or {}
    current = data_record(("train",))

    changed = sorted(name for name in current if learned.get(name) != current[name])

    if changed:
        raise CheckpointError(
            f"{checkpoint} обучен не на текущем наборе train (разные {', '.join(changed)}): "
            "соберите данные, на которых он учился, или возьмите модель этих данных"
        )

    config = MlmConfig.from_dict(state["config"])

    return model, config, {"checkpoint": str(checkpoint), "epoch": int(state["epoch"])}


def embed_group(
    model, group: str, moment: datetime, device, token_budget: int, workers: int = WORKERS
) -> tuple[pa.Table, dict]:
    """
    Векторы всех клиентов группы с событиями до момента.
    """

    import torch

    from src.mlm.inputs import micro_batches
    from src.mlm.model import pack
    from src.mlm.varlen import autocast

    from .at_cutoff import ClientsAtCutoff, clients_at

    builder = ClientsAtCutoff(group, moment)

    skipped = 0

    def with_events():
        nonlocal skipped
        for client in clients_at(builder, workers):
            if client.n_events:
                yield client
            else:
                skipped += 1

    columns: dict[str, list] = {name: [] for name in schema().names}

    started = time.perf_counter()

    with torch.no_grad():

        for clients in micro_batches(with_events(), token_budget):

            with autocast(device):
                vectors = model.readouts(pack(clients, device))

            vectors = {name: vectors[name].cpu().numpy() for name in READOUTS}

            for number, client in enumerate(clients):
                columns["client_id"].append(client.client_id)
                columns["cutoff"].append(builder.cutoff)
                columns["n_events"].append(client.n_events)
                columns["n_tokens"].append(client.n_tokens)
                columns["gap_seconds"].append((builder.cutoff - client.event_time[-1]).total_seconds())
                for name in READOUTS:
                    columns[name].append(vectors[name][number].tolist())

    table = pa.Table.from_pydict(columns, schema=schema())

    return table, {
        "cutoff": builder.cutoff.isoformat(),
        "clients": table.num_rows,
        "skipped_without_events": skipped,
        "seconds": time.perf_counter() - started,
        **raw_record(group),
    }


def raw_record(group: str) -> dict:
    """
    Из какой выгрузки собран вход на T: sha256 событий и анкеты её
    manifest. Проба сверяет их с текущей выгрузкой — той же, из
    которой churn-бейзлайн строит признаки.
    """

    from src.preprocessing.rawdata import read_manifest
    from src.preprocessing.settings import raw_group_dir

    exported = read_manifest(raw_group_dir(group))

    return {"raw_events_sha256": exported.events_sha256, "raw_profile_sha256": exported.profile_sha256}


def embed_groups(requested: list[str] | None, final_test: bool) -> list[str]:
    """
    Группы съёма: по умолчанию train и val, в финальной оценке и
    test. Векторы test без --final-test не снимаются.
    """

    chosen = list(requested or groups(final_test))

    if "test" in chosen and not final_test:
        raise ValueError("test — только для финальной оценки: добавьте --final-test")

    return chosen


def run(args) -> int:

    try:
        chosen = embed_groups(args.groups, args.final_test)
    except ValueError as error:
        print(f"[embed] {error}")
        return EXIT_BLOCKED

    try:
        import torch

        from src.mlm.backbone import BackboneError
        from src.mlm.train import CheckpointError

    except ModuleNotFoundError as error:
        print(f"[embed] нет модуля {error.name}: pip install -e .[torch]")
        return EXIT_BLOCKED

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model, config, source = trained_model(args.checkpoint, device)
    except (CheckpointError, BackboneError, FileNotFoundError) as error:
        print(f"[embed] {error}")
        return EXIT_BLOCKED

    tag = args.tag or Path(args.checkpoint).stem

    directory = downstream_dir(tag)
    directory.mkdir(parents=True, exist_ok=True)

    meta = dict(
        source, tag=tag, device=str(device), attention=model.attention, final_test=args.final_test, groups={}
    )

    for group in chosen:

        table, counts = embed_group(model, group, cutoff(group), device, config.token_budget, args.workers)

        pq.write_table(table, directory / f"{group}.parquet", compression="zstd")

        meta["groups"][group] = counts

        print(
            f"[embed] {group}: T {counts['cutoff']}, клиентов {counts['clients']}, без событий "
            f"до T {counts['skipped_without_events']}, {counts['seconds'] / 60:.1f} мин"
        )

    # Отметка последней: прерванный съём её не получает.
    write_json(directory / EMBEDDINGS_META, meta)

    print(f"[embed] → {directory}")

    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="python -m src.downstream.embed")
    parser.add_argument(
        "--checkpoint", required=True,
        help="чекпойнт или веса эпохи обученной модели",
    )
    parser.add_argument(
        "--groups", nargs="+", choices=FINAL_GROUPS, default=None,
        help="группы; по умолчанию train и val, с --final-test — и test",
    )
    parser.add_argument(
        "--final-test", action="store_true", help="финальная оценка: векторы и для test",
    )
    parser.add_argument("--tag", default=None, help="имя каталога векторов (по умолчанию имя файла)")
    parser.add_argument(
        "--workers", type=int, default=WORKERS, help="процессов сборки входа на T; 0 — в этом процессе"
    )
    parser.set_defaults(handler=run)

    return parser


def main(argv: list[str] | None = None) -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = build_parser().parse_args(argv)

    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()


__all__ = ["READOUTS", "embed_group", "embed_groups", "main", "raw_record", "schema", "trained_model"]
