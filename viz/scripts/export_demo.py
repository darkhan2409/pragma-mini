from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

# Скрипт лежит в viz/scripts и импортирует код проекта из корня.
ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tokenization.finalvocab import FrozenArtifacts, load_final_vocab, vocabulary_digest  # noqa: E402


# ============================================================
# ФАКТЫ ПРОЕКТА ДЛЯ ВИЗУАЛИЗАЦИИ
# ============================================================
#
#   python viz/scripts/export_demo.py [--group train] [--client ID]
#                                     [--run data/runs/<имя>]
#                                     [--out viz/src/data/pragma_demo.json]
#
# Визуализация (viz/) не держит ни одного числа модели руками:
# размеры, параметры, конфиги, словарь, окна, план шагов и пример
# клиента берутся здесь, кодом проекта, из data/. Скрипт только
# читает data/ и пишет один JSON.
#
# Пример клиента — строка набора 05 и его же события из 02:
# сырые значения выделенных событий сверяются с токенами набора
# тем же кодированием (encode_event), что строило набор. Не
# совпало хоть одно событие — экспорт останавливается.
# ============================================================


FORMAT = 1

DEFAULT_CLIENT = "c000728448910"

# Какую покупку показывать сквозным событием, если она есть у
# клиента: узнаваемые названия из нескольких кусков BPE.
PREFERRED_MERCHANTS = ("magnum express", "yandex go", "додо пицца", "apple tv")

# Ещё по одному событию этих типов — для параллельных дорожек.
COMPANION_TYPES = ("salary_credit", "app_screen", "communication_sent")

KINDS = ("special", "key", "value", "bucket", "bpe")


class ExportError(RuntimeError):
    """
    Данные не сходятся с кодом: экспорт остановлен.
    """


def _round(value, digits: int = 6):
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else None
    return value


def token_kind(name: str) -> str:
    """
    Вид токена по имени финального словаря.
    """

    if name.startswith("["):
        return "special"

    for kind in ("key", "value", "bucket", "bpe"):
        if name.startswith(f"{kind}:"):
            return kind

    raise ExportError(f"токен неизвестного вида: {name!r}")


def vocabulary_summary(vocab: dict[str, int]) -> dict:
    """
    Словарь по видам: сколько токенов и какие номера.
    """

    ranges: dict[str, list[int]] = {}

    for name, token_id in vocab.items():
        ranges.setdefault(token_kind(name), []).append(token_id)

    kinds = {
        kind: {"count": len(ids), "first": min(ids), "last": max(ids)}
        for kind, ids in ((kind, ranges.get(kind, [])) for kind in KINDS)
        if ids
    }

    if sum(item["count"] for item in kinds.values()) != len(vocab):
        raise ExportError("виды токенов не покрывают словарь")

    return {
        "size": len(vocab),
        "kinds": kinds,
        "specials": {name: token_id for name, token_id in vocab.items() if name.startswith("[")},
    }


def bucket_index(artifacts: FrozenArtifacts) -> dict[int, dict]:
    """
    Номер диапазона → границы и условие.
    """

    found = {}

    for key, buckets in artifacts.buckets.items():
        for bucket in buckets:
            found[bucket.token_id] = {
                "key": key,
                "name": bucket.name,
                "min": bucket.minimum,
                "max": bucket.maximum,
                "when": bucket.when,
            }

    return found


def token_view(artifacts: FrozenArtifacts, buckets: dict[int, dict], key_names: dict[int, str],
               key_id: int, value_id: int, position: int) -> dict:
    """
    Один токен так, как его видит модель, с человеческим именем.
    """

    value_name = artifacts.describe(value_id)
    kind = token_kind(value_name)

    view = {
        "key_id": key_id,
        "value_id": value_id,
        "position": position,
        "key": key_names.get(key_id, artifacts.describe(key_id)),
        "value": value_name,
        "kind": kind,
    }

    if kind == "bucket":
        view["range"] = buckets[value_id]

    if kind == "bpe":
        view["piece"] = value_name[len("bpe:"):]

    if kind == "value":
        view["text"] = value_name.split("=", 1)[1] if "=" in value_name else value_name

    return view


def _slice(row: dict, start: int, length: int) -> list[tuple[int, int, int]]:
    return list(zip(
        row["key_ids"][start:start + length],
        row["value_ids"][start:start + length],
        row["positions"][start:start + length],
    ))


def _plain(value):
    """
    Сырое значение события для JSON.
    """

    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        return _round(value, 4)
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    return str(value)


def match_events(artifacts: FrozenArtifacts, row: dict, events: list, limit: int) -> int:
    """
    Сдвиг событий 02 относительно событий строки 05.

    Набор мог отбросить самые старые события (предел контекста),
    поэтому строка 05 — хвост ленты 02. Каждое событие хвоста
    обязано кодироваться в те же токены, что лежат в наборе.
    """

    from src.tokenization.encode import encode_event

    starts, lengths = row["event_starts"], row["event_lengths"]

    offset = len(events) - len(starts)

    if offset < 0:
        raise ExportError(f"в 05 событий {len(starts)}, а в 02 только {len(events)}")

    for index, (start, length) in enumerate(zip(starts, lengths)):

        record = encode_event(artifacts, events[offset + index], limit)
        stored = _slice(row, start, length)

        if list(zip(record.key_ids, record.value_ids, record.positions)) != stored:
            raise ExportError(
                f"событие {index} клиента {row['client_id']} в 02 и 05 кодируется по-разному"
            )

    return offset


def pieces_of(artifacts: FrozenArtifacts, row: dict, start: int, length: int, key: str) -> int:
    """
    Сколько кусков у значения ключа в событии.
    """

    key_id = artifacts.key_id(key)

    return sum(1 for k, _v, _p in _slice(row, start, length) if k == key_id)


def choose_events(artifacts: FrozenArtifacts, row: dict, events: list, offset: int) -> dict[str, int]:
    """
    Сквозное событие и по одному событию сопутствующих типов.
    Номера — в строке 05.

    Сквозное — покупка с суммой. Первыми — те, у которых сумма есть
    в токенах (её прячет шаг MLM), затем — с названием из нескольких
    кусков BPE (узнаваемые названия — первыми): на ней видно
    разбиение текста.
    """

    starts, lengths = row["event_starts"], row["event_lengths"]

    chosen: dict[str, int] = {}
    candidates: list[tuple[int, int, int]] = []

    for index in range(len(starts)):

        event = events[offset + index]
        kind = event.values.get("event_type")

        if kind == "purchase" and "transaction_amount" in event.values:

            name = str(event.values.get("merchant_name") or "")
            encoded = pieces_of(artifacts, row, starts[index], lengths[index], "transaction_amount") > 0
            split = pieces_of(artifacts, row, starts[index], lengths[index], "merchant_name") >= 2
            known = PREFERRED_MERCHANTS.index(name) if name in PREFERRED_MERCHANTS else len(PREFERRED_MERCHANTS)

            candidates.append((0 if encoded else 1, 0 if split else 1, known, index))

        if kind in COMPANION_TYPES and kind not in chosen:
            chosen[kind] = index

    if not candidates:
        raise ExportError("у клиента нет покупки с суммой: выберите другого клиента")

    chosen["hero"] = min(candidates)[-1]

    return chosen


def event_view(artifacts: FrozenArtifacts, buckets: dict, key_names: dict[int, str], row: dict,
               event, index: int, timezone_name) -> dict:
    """
    Событие целиком: сырые значения из 02, календарь и токены 05.
    """

    start, length = row["event_starts"][index], row["event_lengths"][index]

    local = event.event_time.astimezone(timezone_name)

    return {
        "index": index,
        "type": event.values.get("event_type"),
        "source": event.source,
        "time": event.event_time.isoformat(),
        "local_time": local.strftime("%Y-%m-%d %H:%M:%S"),
        "raw": {key: _plain(value) for key, value in sorted(event.values.items())},
        "calendar": [_round(float(x), 6) for x in row["calendar"][index * 6:(index + 1) * 6]],
        "time_log": _round(float(row["event_time_log"][index]), 6),
        "target": bool(row["target_event_mask"][index]),
        "tokens": [
            token_view(artifacts, buckets, key_names, k, v, p) for k, v, p in _slice(row, start, length)
        ],
    }


def profile_view(artifacts: FrozenArtifacts, buckets: dict, key_names: dict[int, str], row: dict) -> list[dict]:
    """
    Токены анкеты с их временем и давностью до T.
    """

    tokens = []

    for index, (k, v, p) in enumerate(zip(row["profile_key_ids"], row["profile_value_ids"], row["profile_positions"])):

        view = token_view(artifacts, buckets, key_names, k, v, p)

        moment = row["profile_time"][index]
        view["time"] = moment.isoformat() if moment is not None else None
        view["time_log"] = _round(float(row["profile_time_log"][index]), 6)

        tokens.append(view)

    return tokens


def client_export(artifacts: FrozenArtifacts, row: dict, history, timezone_name, limit: int) -> dict:
    """
    Клиент для визуализации: лента целиком (время, тип, источник,
    длина, давность), выделенные события целиком и анкета.
    """

    offset = match_events(artifacts, row, history.events, limit)

    key_names = {token_id: key for key, token_id in artifacts.keys.items()}
    buckets = bucket_index(artifacts)

    starts, lengths = row["event_starts"], row["event_lengths"]
    events = history.events[offset:]

    timeline = [
        {
            "t": event.event_time.isoformat(),
            "type": event.values.get("event_type"),
            "source": event.source,
            "n_tokens": int(lengths[index]),
            "time_log": _round(float(row["event_time_log"][index]), 5),
            "target": bool(row["target_event_mask"][index]),
        }
        for index, event in enumerate(events)
    ]

    chosen = choose_events(artifacts, row, history.events, offset)

    highlighted = {
        role: event_view(artifacts, buckets, key_names, row, events[index], index, timezone_name)
        for role, index in chosen.items()
    }

    profile = profile_view(artifacts, buckets, key_names, row)

    if len(timeline) != len(starts) or sum(item["n_tokens"] for item in timeline) != len(row["key_ids"]):
        raise ExportError("лента клиента не сходится со строкой набора")

    return {
        "client_id": row["client_id"],
        "n_events": len(starts),
        "n_tokens": len(row["key_ids"]),
        "n_profile_tokens": len(row["profile_key_ids"]),
        "dropped_old_events": offset,
        "type_counts": dict(Counter(item["type"] for item in timeline).most_common()),
        "source_counts": dict(Counter(item["source"] for item in timeline).most_common()),
        "timeline": timeline,
        "highlighted": highlighted,
        "profile": profile,
    }


def mlm_candidates(buckets: dict[int, dict], hero: dict) -> dict:
    """
    Кандидаты шага MLM для суммы сквозного события: все диапазоны её
    шкалы при том же условии (direction), в порядке границ, и
    настоящий токен суммы. Суммы нет в токенах — скрывать на шаге
    MLM нечего, и экспорт отказывает.
    """

    amount = next(
        (token for token in hero["tokens"] if token["key"] == "transaction_amount" and token["kind"] == "bucket"), None
    )

    if amount is None:
        raise ExportError("у сквозного события сумма не закодирована: шагу MLM нечего скрыть")

    when = amount["range"]["when"]

    candidates = sorted(
        (
            {"value_id": token_id, **item}
            for token_id, item in buckets.items()
            if item["key"] == "transaction_amount" and item["when"] == when
        ),
        key=lambda item: (item["min"] is not None, item["min"] or 0.0),
    )

    return {"key": "transaction_amount", "target": amount["value_id"], "buckets": candidates}


def find_row(group: str, client_id: str) -> dict:
    """
    Строка клиента в наборе 05 со временем, посчитанным при чтении.
    """

    import pyarrow.parquet as pq

    from src.temporal.samples import TemporalGroup

    samples = TemporalGroup(group)

    handle = pq.ParquetFile(samples.path)

    for number in range(handle.num_row_groups):

        ids = handle.read_row_group(number, columns=["client_id"]).column(0).to_pylist()

        if client_id in ids:
            table = samples.row_group(number)
            return table.slice(ids.index(client_id), 1).to_pylist()[0]

    raise ExportError(f"клиента {client_id} нет в наборе {group}")


def architecture_summary(backbone_dir: Path, embeddings_dir: Path, head_inputs: int = 3) -> dict:
    """
    Размеры и параметры модели: из lineage и из самих весов.
    """

    import torch

    lineage = json.loads((backbone_dir / "lineage.json").read_text(encoding="utf-8"))
    embedding = json.loads((embeddings_dir / "lineage.json").read_text(encoding="utf-8"))

    encoders = {}

    for name, item in lineage["encoders"].items():

        state = torch.load(backbone_dir / f"{name}.pt", map_location="cpu", weights_only=True)["state_dict"]

        parts: dict[str, int] = {}

        for tensor_name, tensor in state.items():
            head = tensor_name.split(".")
            part = ".".join(head[:2]) if head[0] == "layers" else head[0]
            parts[part] = parts.get(part, 0) + tensor.numel()

        total = sum(parts.values())

        if total != item["parameters"]:
            raise ExportError(f"{name}: в весах {total} параметров, в lineage {item['parameters']}")

        encoders[name] = {"config": item["config"], "blocks": item["blocks"], "parameters": total, "parts": parts}

    dim = int(lineage["embedding"]["dim"])
    vocab = int(lineage["embedding"]["vocab_size"])

    table = vocab * dim
    head = head_inputs * dim * dim + dim

    return {
        "dim": dim,
        "vocab_size": vocab,
        "embedding_seed": embedding.get("seed", lineage["embedding"].get("seed")),
        "embedding_parameters": table,
        "head_parameters": head,
        "encoders": encoders,
        "total_parameters": table + head + sum(item["parameters"] for item in encoders.values()),
        "implementation": lineage.get("implementation"),
    }


def batching_summary(group: str, token_budget: int) -> dict:
    """
    Сколько micro-batch'ей в эпохе и сколько в них токенов — по
    длинам клиентов, без масок и модели.
    """

    from src.mlm.inputs import Source, micro_batches

    tokens = [
        sum(size.n_tokens + size.profile_n_tokens for size in batch)
        for batch in micro_batches(Source(group).sizes(), token_budget)
    ]

    clients = sum(1 for _ in Source(group).sizes())

    return {
        "clients": clients,
        "micro_batches": len(tokens),
        "tokens": sum(tokens),
        "mean_tokens": _round(sum(tokens) / len(tokens), 1) if tokens else None,
        "max_tokens": max(tokens) if tokens else None,
    }


def run_summary(run: Path) -> dict | None:
    """
    Прогон обучения по его телеметрии и заголовку лога.
    """

    from src.dashboard.telemetry import TelemetryReader, progress
    from src.mlm.settings import TELEMETRY_FILE

    telemetry = TelemetryReader(run / TELEMETRY_FILE).poll()

    if not telemetry.runs and not telemetry.epochs:
        return None

    where = progress(telemetry)

    log = run / "train.log"
    header = []

    if log.exists():
        with open(log, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("[train] "):
                    header.append(line.rstrip())
                if len(header) >= 3:
                    break

    return {
        "run": run.name,
        "plan": telemetry.runs[0] if telemetry.runs else None,
        "epochs": [
            {
                key: _round(record.get(key)) for key in (
                    "epoch", "step", "train_loss", "val_loss", "learning_rate", "train_seconds",
                    "val_seconds", "grad_norm_mean", "clipped_share", "cuda_peak_allocated_gib",
                )
            }
            for _, record in sorted(telemetry.epochs.items())
        ],
        "steps": where["step"],
        "total_steps": where["total_steps"],
        "best_val_loss": _round(where["best_val_loss"]),
        "header": header,
    }


def scenarios(tag: str) -> dict | None:
    """
    Три сценария прогона на val — из отчёта его пробы (probe --tag
    <прогон>): handcrafted → CatBoost, [USR] → регрессия, handcrafted
    и [USR] → CatBoost, с парной разницей к первому. Нет отчёта, в нём
    test или векторы сняты не с текущей выгрузки — None.
    """

    from src.downstream.probe import SCENARIOS
    from src.downstream.settings import EMBEDDINGS_META, REPORT_FILE, downstream_dir
    from src.preprocessing.rawdata import read_manifest
    from src.preprocessing.settings import raw_group_dir

    path = downstream_dir(tag) / REPORT_FILE

    if not path.exists():
        return None

    report = json.loads(path.read_text(encoding="utf-8"))

    if report.get("final_test"):
        return None

    meta = json.loads((downstream_dir(tag) / EMBEDDINGS_META).read_text(encoding="utf-8"))

    for group in report["groups"]:
        exported = read_manifest(raw_group_dir(group))
        recorded = meta.get("groups", {}).get(group, {})
        if (recorded.get("raw_events_sha256"), recorded.get("raw_profile_sha256")) != (
            exported.events_sha256, exported.profile_sha256
        ):
            return None

    def cell(result: dict) -> dict:
        delta = result.get("vs_reference", {}).get("val")
        return {
            **{metric: _round(result["val"][metric], 4) for metric in ("pr_auc", "roc_auc", "log_loss", "f1")},
            "vs_reference": {
                metric: {key: _round(delta[metric][key], 3) for key in ("mean", "low", "high")}
                for metric in ("pr_auc", "roc_auc")
            } if delta else None,
        }

    tasks = {
        task: {
            "rows": block["rows"]["val"],
            "positives": block["positives"]["val"],
            "reference": block["reference"],
            "cells": {name: cell(block["results"][name]) for name in SCENARIOS if name in block["results"]},
        }
        for task, block in report["tasks"].items()
    }

    return {
        "group": "val",
        "tag": tag,
        "tasks": tasks,
        "source": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
    }


# Точек кривой ранней остановки в экспорте не больше этого.
CURVE_POINTS = 200


def thinned(curve: list[float], best: int, limit: int = CURVE_POINTS) -> list[list[float]]:
    """
    Кривая [итерация, значение], прорежённая до limit точек: первая,
    лучшая и последняя итерации остаются всегда.
    """

    stride = max(1, -(-len(curve) // limit))
    keep = sorted(set(range(0, len(curve), stride)) | {best, len(curve) - 1})
    return [[index, curve[index]] for index in keep]


def current_churn(metrics: dict) -> bool:
    """
    Отчёт churn_baseline собран на текущей выгрузке train и val и без
    test.
    """

    from src.preprocessing.rawdata import read_manifest
    from src.preprocessing.settings import raw_group_dir

    if metrics.get("final_test"):
        return False

    for group in ("train", "val"):
        exported = read_manifest(raw_group_dir(group))
        built = metrics.get("sources", {}).get(group, {})
        if (built.get("feature_history_events_sha256"), built.get("feature_profile_sha256")) != (
            exported.events_sha256, exported.profile_sha256
        ):
            return False

    return True


def training(tag: str) -> dict | None:
    """
    Процесс обучения голов для шагов 17 и 18: перебор C регрессии над
    [USR] — из отчёта пробы прогона; ранняя остановка, порог и
    переобучение CatBoost на handcrafted-признаках и на них же с [USR]
    — из отчётов churn_baseline. Только текущие отчёты без test, иначе
    None.
    """

    from src.downstream.probe import FOLDS
    from src.downstream.settings import CHURN_REPORTS, REPORT_FILE, downstream_dir

    # Отчёт пробы — тот же, что у трёх сценариев, с теми же проверками.
    found = scenarios(tag)

    if found is None:
        return None

    # CatBoost на handcrafted + [USR] — только если проба его приняла.
    paths = {"catboost": CHURN_REPORTS / "metrics.json"}
    if all("catboost_plus_usr" in block["cells"] for block in found["tasks"].values()):
        paths["catboost_plus_usr"] = CHURN_REPORTS / "plus_usr" / tag / "metrics.json"

    if not all(path.exists() for path in paths.values()):
        return None

    metrics = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}

    if not all(current_churn(item) for item in metrics.values()):
        return None

    report = json.loads((downstream_dir(tag) / REPORT_FILE).read_text(encoding="utf-8"))

    def val(result: dict) -> dict:
        return {key: _round(result[key], 4) for key in ("pr_auc", "roc_auc", "f1")} | {
            "rows": result["rows"], "positives": result["positives"],
        }

    tasks = {}

    for task, block in report["tasks"].items():
        usr = block["results"]["usr"]
        lr = {
            "folds": FOLDS,
            "grid": [{"C": point["C"], "log_loss": _round(point["log_loss"], 4)} for point in usr["cv"]],
            "C": usr["C"],
            "threshold": _round(usr["threshold"], 4),
            "train": {"rows": block["rows"]["train"], "positives": block["positives"]["train"]},
            "val": val(usr["val"]),
        }

        catboost = {}
        for name, item in metrics.items():
            fitted = item["tasks"][task]
            features = item["features"]
            catboost[name] = {
                "features": features if isinstance(features, int) else features["total"],
                "usr": 0 if isinstance(features, int) else features["usr"],
                "holdout_share": item.get("holdout_share", metrics["catboost"].get("holdout_share")),
                "inner_train": {key: fitted["inner_train"][key] for key in ("rows", "positives")},
                "inner_holdout": {key: fitted["inner_holdout"][key] for key in ("rows", "positives")},
                "best_iteration": fitted["best_iteration"],
                "trees": fitted["trees"],
                "threshold": _round(fitted["threshold"], 4),
                "curve": thinned(fitted["holdout_curve"], fitted["best_iteration"]),
                "val": val(fitted["groups"]["val"]),
            }

        tasks[task] = {"lr": lr, "catboost": catboost}

    return {"group": "val", "tag": tag, "seed": metrics["catboost"]["params"]["random_seed"], "tasks": tasks}


def downstream_summary(run: Path | None) -> dict:
    """
    Задачи и пробы стенда оценки.
    """

    from src.downstream.probe import COMPARED, REFERENCE, TASKS
    from src.downstream.settings import HORIZON_DAYS, cutoff
    from src.preprocessing.rawdata import read_manifest
    from src.preprocessing.settings import raw_group_dir

    # CatBoost-бейзлайн — только на val: test во время экспериментов не
    # показывается. Отчёт по другой выгрузке не берётся.
    metrics_path = ROOT / "churn_baseline" / "reports" / "metrics.json"
    catboost = None

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        exported = read_manifest(raw_group_dir("val"))
        built = metrics.get("sources", {}).get("val", {})
        current = (exported.events_sha256, exported.profile_sha256)
        if (built.get("feature_history_events_sha256"), built.get("feature_profile_sha256")) == current and "tasks" in metrics:
            catboost = {
                "group": "val",
                "tasks": {
                    task: {
                        "pr_auc": _round(block["groups"]["val"]["pr_auc"], 4),
                        "roc_auc": _round(block["groups"]["val"]["roc_auc"], 4),
                        "rows": block["groups"]["val"]["rows"],
                        "positives": block["groups"]["val"]["positives"],
                    }
                    for task, block in metrics["tasks"].items()
                },
                "source": "churn_baseline/reports/metrics.json",
            }

    return {
        "tasks": list(TASKS),
        "reference": dict(REFERENCE),
        "compared": list(COMPARED),
        "horizon_days": HORIZON_DAYS,
        "cutoffs": {group: cutoff(group).isoformat() for group in ("train", "val", "test")},
        "catboost_churn": catboost,
        "scenarios": scenarios(run.name) if run is not None else None,
        "training": training(run.name) if run is not None else None,
    }


def model_facts(checkpoint: Path, group: str, row: dict, event_index: int, key_id: int, candidates: list[int],
                describe: Callable[[int], str]) -> dict:
    """
    Настоящие числа обученной модели для одного клиента, на CPU:

      attention  строки внимания [USR] энкодера истории по блокам и
                 головам: вес [USR] и каждого события (без маски);
      mlm        значение ключа key_id в событии event_index скрыто
                 [MASK], как делает маскер: top-5 по всему словарю
                 с именами токенов, вероятности кандидатов и
                 настоящего значения.
    """

    import torch

    from src.masking.choose import NONE, VALUE
    from src.mlm.diagnostics import usr_attention_rows
    from src.mlm.inputs import IGNORE, Source
    from src.mlm.model import pack
    from src.mlm.train import load_trained

    device = torch.device("cpu")

    model, state = load_trained(checkpoint, device, attention_backend="sdpa")
    source = Source(group)

    width = len(row["value_ids"])
    plain = {"value_ids": list(row["value_ids"]), "labels": [IGNORE] * width, "reason": [NONE] * width}

    rows, error = usr_attention_rows(model, source._client(0, row, plain), device)

    start, length = row["event_starts"][event_index], row["event_lengths"][event_index]
    place = next(i for i in range(start, start + length) if row["key_ids"][i] == key_id and row["positions"][i] == 0)

    masked = {key: list(value) for key, value in plain.items()}
    target = masked["value_ids"][place]
    masked["labels"][place] = target
    masked["value_ids"][place] = source.mask_id
    masked["reason"][place] = VALUE

    with torch.no_grad():
        out = model(pack([source._client(0, row, masked)], device), logits=True)

    probabilities = torch.softmax(out.logits[0].float(), dim=-1)
    top = torch.topk(probabilities, 5)
    rank = int((probabilities > probabilities[target]).sum()) + 1

    history = state.get("history") or []
    last = history[-1] if history else {}

    return {
        "checkpoint": str(checkpoint.relative_to(ROOT) if checkpoint.is_relative_to(ROOT) else checkpoint),
        "epoch": int(state.get("epoch", 0)),
        "val_loss": _round(last.get("val", {}).get("loss") if isinstance(last, dict) else None),
        "attention": {
            "row_error": _round(error, 8),
            "blocks": [[[_round(float(w), 6) for w in head] for head in block] for block in rows],
        },
        "mlm": {
            "event_index": event_index,
            "target": int(target),
            "target_probability": _round(float(probabilities[target]), 6),
            "target_rank": rank,
            "top5": [
                {"value_id": int(i), "name": describe(int(i)), "p": _round(float(p), 6)}
                for p, i in zip(top.values.tolist(), top.indices.tolist())
            ],
            "candidates": {str(i): _round(float(probabilities[i]), 6) for i in candidates},
        },
    }


def export(group: str, client_id: str, run: Path | None, checkpoint: Path | None = None) -> dict:

    from src.dataset.settings import DatasetConfig
    from src.masking.settings import MaskingConfig
    from src.mlm.settings import BACKBONE_DIR, MlmConfig
    from src.embedding.settings import EMBEDDINGS_DIR
    from src.preprocessing.read import Group
    from src.preprocessing.settings import CALENDAR_ENCODING, PreprocessingConfig
    from src.temporal.samples import TemporalGroup
    from src.tokenization.settings import TokenizerConfig

    artifacts = FrozenArtifacts.load()
    vocab = load_final_vocab()

    samples = TemporalGroup(group)
    row = find_row(group, client_id)

    history = Group(group).history(client_id, samples.cutoff)

    zone = PreprocessingConfig.load(None).bank_timezone()

    client = client_export(
        artifacts, row, history, zone, TokenizerConfig.load(None).max_pieces_per_value
    )
    client["mlm_candidates"] = mlm_candidates(bucket_index(artifacts), client["highlighted"]["hero"])

    training = MlmConfig()
    architecture = architecture_summary(BACKBONE_DIR, EMBEDDINGS_DIR / "train")

    if architecture["vocab_size"] != len(vocab):
        raise ExportError("веса собраны под другой словарь: пересоберите 06 и 07")

    return {
        "format": FORMAT,
        "sources": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "vocabulary_digest": vocabulary_digest(),
            "group": group,
        },
        "vocabulary": vocabulary_summary(vocab),
        "architecture": architecture,
        "training": training.as_dict(),
        "masking": MaskingConfig().as_dict(),
        "dataset": {
            "group": group,
            "cutoff": samples.cutoff.isoformat(),
            "time_anchor": samples.anchor,
            "window": samples.meta.get("window"),
            "context": samples.meta.get("context"),
            "profile_fields": samples.meta.get("profile_fields"),
            "lifelong_types": samples.meta.get("profile_lifelong_types"),
            "calendar": CALENDAR_ENCODING,
            "bank_timezone": str(zone),
            # Фиксированный сдвиг, как в коде проекта: IANA-пояс
            # Asia/Almaty до 2024-03 дал бы другой календарь.
            "bank_utc_offset_hours": PreprocessingConfig.load(None).timezone_hours,
            "default_config": DatasetConfig().as_dict(),
        },
        "batching": batching_summary(group, training.token_budget),
        "client": client,
        "downstream": downstream_summary(run),
        "run": run_summary(run) if run is not None else None,
        "model": model_facts(
            checkpoint, group, row, client["highlighted"]["hero"]["index"], artifacts.key_id("transaction_amount"),
            [item["value_id"] for item in client["mlm_candidates"]["buckets"]], artifacts.describe,
        ) if checkpoint is not None else None,
    }


def dumps(value, level: int = 0) -> str:
    """
    JSON, где контейнер из одних чисел, строк и флагов — одна строка:
    строка весов внимания, событие ленты, токен. Иначе по строке на
    число, и экспорт весил бы 11 тысяч строк. Остальное — с отступом
    в один пробел на уровень.
    """

    def flat(items) -> bool:
        return all(not isinstance(item, (dict, list)) for item in items)

    if isinstance(value, dict) and not flat(value.values()):
        inner = " " * (level + 1)
        # Ключ — всегда строка, как у json.dumps: 1623 → "1623".
        rows = [
            f"{inner}{json.dumps(key if isinstance(key, str) else json.dumps(key), ensure_ascii=False)}: "
            f"{dumps(item, level + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(rows) + "\n" + " " * level + "}"

    if isinstance(value, list) and not flat(value):
        inner = " " * (level + 1)
        return "[\n" + ",\n".join(inner + dumps(item, level + 1) for item in value) + "\n" + " " * level + "]"

    return json.dumps(value, ensure_ascii=False)


def write(payload: dict, path: Path) -> None:
    """
    Запись целиком через временный файл.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".tmp")
    temporary.write_text(dumps(payload) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="python viz/scripts/export_demo.py")
    parser.add_argument("--group", default="train")
    parser.add_argument("--client", default=DEFAULT_CLIENT)
    parser.add_argument("--run", type=Path, default=None, help="каталог прогона, например data/runs/<имя>")
    parser.add_argument("--out", type=Path, default=ROOT / "viz" / "src" / "data" / "pragma_demo.json")
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="чекпойнт обученной модели: настоящие веса внимания [USR] и вероятности MLM",
    )
    args = parser.parse_args(argv)

    try:
        payload = export(args.group, args.client, args.run, args.checkpoint)
    except ExportError as error:
        print(f"[export] {error}")
        return 2

    write(payload, args.out)

    client = payload["client"]
    print(f"[export] {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")
    print(
        f"    словарь {payload['vocabulary']['size']}, параметров {payload['architecture']['total_parameters']:,}; "
        f"клиент {client['client_id']}: событий {client['n_events']}, токенов {client['n_tokens']}, "
        f"анкета {client['n_profile_tokens']}; сверено событий с 02: {client['n_events']}"
    )
    print(f"    выделено: " + ", ".join(f"{role} #{item['index']} {item['type']}" for role, item in client["highlighted"].items()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
