from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np

from .inputs import Client


# ============================================================
# ДИАГНОСТИКА ПРЕДСТАВЛЕНИЯ КЛИЕНТА
# ============================================================
#
# Только оценка: обучения, записи в data/ и изменений прохода
# модели здесь нет.
#
# client_embedding — вектор [USR] клиента после последнего блока
# энкодера истории и его нормы (Model.client_embeddings): тот же
# вектор, что получает голова MLM.
#
# Абляции — варианты ОДНОГО клиента из его же настоящих данных,
# без единого выдуманного события или токена:
#
#   full              всё как есть;
#   no_events         та же анкета, событий нет — так выглядит
#                     настоящий клиент без истории;
#   shuffled_content  те же события и те же моменты, но содержимое
#                     событий переставлено по моментам: хронология
#                     нарушена;
#   shuffled_order    события вместе со своими моментами в другом
#                     порядке строк. Позиция в энкодере истории —
#                     время (TimeRoPE), порядкового номера нет, и
#                     вектор обязан не измениться: это проверка
#                     самой диагностики, а не вопрос к модели;
#   partial_50/25     последние 50 % / 25 % событий;
#   no_profile        анкета из одного [USR] — так выглядит
#                     настоящий клиент без анкеты. Насколько это
#                     в распределении обучения, решает число таких
#                     клиентов в train (его печатает отчёт).
#
# Вход — немаскированные value_ids: маскирование с нулевыми
# вероятностями ничего не закрывает.
#
# Внимание [USR] в энкодере истории: для каждого блока и головы —
# доля массы внимания запроса [USR] на сам [USR] и на события.
# Считается на пути SDPA в fp32 с теми же весами: вход каждого
# блока и углы TimeRoPE снимаются хуком с настоящего прохода, по
# ним строится одна строка softmax(QKᵀ/√d). Эта строка обязана
# воспроизвести настоящий выход внимания в позиции [USR] — расхождение
# печатается. Веса внимания не доказывают причинного влияния:
# главная проверка — абляции.
# ============================================================


VARIANTS = ("no_events", "shuffled_content", "shuffled_order", "partial_50", "partial_25", "no_profile")


def _events(client: Client, order: np.ndarray, times: np.ndarray) -> Client:
    """
    Клиент с событиями в порядке order; моменты (event_time_log,
    календарь, event_time) берутся из слотов times.
    """

    chunks = [slice(int(client.event_starts[i]), int(client.event_starts[i] + client.event_lengths[i])) for i in order]

    def take(values: np.ndarray) -> np.ndarray:
        return np.concatenate([values[chunk] for chunk in chunks]) if chunks else values[:0]

    lengths = client.event_lengths[order]

    return replace(
        client,
        key_ids=take(client.key_ids),
        value_ids=take(client.value_ids),
        positions=take(client.positions),
        labels=take(client.labels),
        reason=[client.reason[index] for chunk in chunks for index in range(chunk.start, chunk.stop)],
        event_starts=np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(client.event_starts.dtype)
        if lengths.size else client.event_starts[:0],
        event_lengths=lengths,
        event_time_log=client.event_time_log[times],
        calendar=client.calendar[times],
        event_time=[client.event_time[i] for i in times],
    )


def variant(client: Client, name: str, rng: np.random.Generator) -> Client:
    """
    Вариант клиента для абляции (см. шапку).
    """

    n = client.n_events
    every = np.arange(n)

    if name == "no_events":
        return _events(client, every[:0], every[:0])

    if name == "shuffled_content":
        return _events(client, rng.permutation(n), every)

    if name == "shuffled_order":
        order = rng.permutation(n)
        return _events(client, order, order)

    if name.startswith("partial_"):
        kept = every[n - math.ceil(n * int(name.split("_")[1]) / 100):]
        return _events(client, kept, kept)

    if name == "no_profile":
        # Первый токен анкеты — всегда [USR].
        return replace(
            client,
            profile_key_ids=client.profile_key_ids[:1],
            profile_value_ids=client.profile_value_ids[:1],
            profile_positions=client.profile_positions[:1],
            profile_time_log=client.profile_time_log[:1],
        )

    raise ValueError(f"неизвестный вариант {name!r}: известны {VARIANTS}")


def embeddings(model, clients: list[Client], device) -> "torch.Tensor":
    """
    client_embedding каждого клиента, по одному клиенту на проход:
    [B, d] в fp32 на CPU.
    """

    import torch

    from .model import pack
    from .varlen import autocast

    model.eval()

    rows = []

    with torch.no_grad():
        for client in clients:
            with autocast(device) if model.attention == "flash" else nullcontext():
                rows.append(model.client_embeddings(pack([client], device)).float().cpu())

    return torch.cat(rows, dim=0)


def attention_to_events(model, client: Client, device) -> dict:
    """
    Доли массы внимания [USR] -> [USR] и [USR] -> события по блокам
    и головам энкодера истории: {"usr": [слои][головы], "events": ...,
    "row_error": наибольшее расхождение строки с настоящим выходом}.

    Модель обязана идти путём SDPA в fp32: хуки снимают вход блоков
    настоящего прохода.
    """

    import torch

    from .model import pack

    if model.attention != "sdpa":
        raise ValueError("диагностика внимания идёт путём SDPA: соберите модель с attention_backend='sdpa'")

    model.eval()

    seen: list[tuple] = []
    mixed: list[torch.Tensor] = []
    handles = []

    for block in model.history.layers:
        handles.append(block.register_forward_pre_hook(lambda module, args: seen.append(args)))
        handles.append(block.out.register_forward_pre_hook(lambda module, args: mixed.append(args[0])))

    try:
        with torch.no_grad():
            model.client_embeddings(pack([client], device))
    finally:
        for handle in handles:
            handle.remove()

    usr, events, error = [], [], 0.0

    with torch.no_grad():
        for block, (x, rope, cos, sin, keys), real in zip(model.history.layers, seen, mixed):

            batch, length, dim = x.shape

            if batch != 1:
                raise ValueError("внимание снимается по одному клиенту")

            qkv = block.qkv(block.norm1(x)).view(batch, length, 3, block.heads, block.head_dim)

            query = rope.rotate(qkv[:, :, 0].transpose(1, 2), cos, sin)
            key = rope.rotate(qkv[:, :, 1].transpose(1, 2), cos, sin)
            value = qkv[:, :, 2].transpose(1, 2)

            # Строка запроса [USR] — позиция 0 истории клиента.
            scores = (query[:, :, :1] @ key.transpose(-1, -2)) / math.sqrt(block.head_dim)

            if keys is not None:
                scores = scores.masked_fill(~keys, float("-inf"))

            weights = scores.softmax(dim=-1)                     # [1, H, 1, L]

            row = (weights @ value).transpose(1, 2).reshape(1, dim)

            error = max(error, float((row - real[:, 0]).abs().max()))

            usr.append(weights[0, :, 0, 0].tolist())
            events.append(weights[0, :, 0, 1:].sum(dim=-1).tolist())

    return {"usr": usr, "events": events, "row_error": error}


def cosine_l2(left: "torch.Tensor", right: "torch.Tensor") -> tuple[np.ndarray, np.ndarray]:

    import torch

    cos = torch.nn.functional.cosine_similarity(left, right, dim=-1)

    return cos.numpy(), (left - right).norm(dim=-1).numpy()


def summary(values: np.ndarray) -> dict:

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def diagnose(
    checkpoint: Path,
    group: str,
    clients: int,
    min_events: int,
    attention_clients: int,
    seed: int,
) -> dict:
    """
    Абляции client_embedding и внимание [USR] на выборке группы.
    """

    import torch

    from src.masking.settings import MaskingConfig
    from src.preprocessing.artifacts import read_json
    from src.tokenization.settings import tokenized_dir
    from src.tokenization.transform import META_FILE

    from .inputs import Source
    from .model import load_model
    from .settings import MlmConfig

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)

    config = MlmConfig.from_dict(state["config"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def model(backend: str):
        built = load_model(config.seed, config.events_per_chunk, config.label_smoothing, device, backend)
        built.load_state_dict(state["model_state_dict"])
        return built.eval()

    # Нулевые вероятности: маскер не закрывает ничего, вход — как есть.
    nothing = MaskingConfig(
        value_probability=0.0, event_probability=0.0, key_probability=0.0, unknown_probability=0.0
    )

    rng = np.random.default_rng(seed)

    pool = [client for client in Source(group, masking=nothing).clients() if client.n_events >= min_events]

    chosen = [pool[i] for i in sorted(rng.choice(len(pool), size=min(clients, len(pool)), replace=False))]

    reference = model("sdpa")

    full = embeddings(reference, chosen, device)

    report: dict = {
        "checkpoint": str(checkpoint),
        "epoch": int(state["epoch"]),
        "group": group,
        "clients": len(chosen),
        "pool": len(pool),
        "min_events": min_events,
        "events_median": float(np.median([c.n_events for c in chosen])),
        "dim": int(full.shape[-1]),
        # Клиентов train с анкетой из одного [USR]: сколько такого
        # входа модель видела при обучении.
        "profile_only_clients_in_train": read_json(tokenized_dir("train") / META_FILE)[
            "clients_with_empty_profile"
        ],
    }

    # Тот же вектор настоящим путём обучения (flash под bf16, если он есть).
    production = model(config.attention_backend)

    if production.attention != "sdpa":
        fast = embeddings(production, chosen, device)
        cos, l2 = cosine_l2(full, fast)
        report["production_vs_reference"] = {
            "attention": production.attention, "cosine_min": float(cos.min()), "l2_max": float(l2.max()),
        }

    for name in VARIANTS:
        changed = embeddings(reference, [variant(c, name, rng) for c in chosen], device)
        cos, l2 = cosine_l2(full, changed)
        report[name] = {"cosine": summary(cos), "l2": summary(l2)}

    report["norm_full"] = summary(full.norm(dim=-1).numpy())

    masses = [attention_to_events(reference, c, device) for c in chosen[:attention_clients]]

    report["attention"] = {
        "clients": len(masses),
        "row_error_max": max(m["row_error"] for m in masses) if masses else None,
        "usr_to_usr": np.mean([m["usr"] for m in masses], axis=0).tolist() if masses else None,
        "usr_to_events": np.mean([m["events"] for m in masses], axis=0).tolist() if masses else None,
    }

    return report


def main(argv: list[str] | None = None) -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="python -m src.mlm.diagnostics")
    parser.add_argument("--checkpoint", type=Path, required=True)
    # test здесь не читается: его оценка — отдельное решение.
    parser.add_argument("--group", choices=("train", "val"), default="val")
    parser.add_argument("--clients", type=int, default=300)
    parser.add_argument("--min-events", type=int, default=100)
    parser.add_argument("--attention-clients", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)

    report = diagnose(args.checkpoint, args.group, args.clients, args.min_events, args.attention_clients, args.seed)

    text = json.dumps(report, ensure_ascii=False, indent=2)

    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")

    print(text)


if __name__ == "__main__":
    main()


__all__ = [
    "VARIANTS",
    "attention_to_events",
    "cosine_l2",
    "diagnose",
    "embeddings",
    "summary",
    "variant",
]
