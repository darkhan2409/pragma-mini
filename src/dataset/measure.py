from __future__ import annotations

from pathlib import Path

from src.preprocessing.artifacts import write_json, write_text

from .build import _lengths
from .encoding import encode_history
from .inputs import DatasetInputs
from .report import render_measure_md
from .settings import DEFAULT_MILESTONES
from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION


# ============================================================
# ИДЕЯ
# ============================================================
#
# Сначала измерить, потом решать.
#
# Бюджет контекста это решение человека, и принимать его вслепую
# нельзя: число 4096 из архива ничего не говорит о наших данных.
# Эта команда кодирует группу на её конечном срезе, считает
# длины и НИЧЕГО не сохраняет из примеров.
#
# Измерять полагается на train: выбирать предел по validation
# или test значило бы подглядывать в данные, на которых потом
# меряют.
# ============================================================


MEASURE_DIRNAME = "measurements"

# Кандидаты в пределы, по которым считается доля затронутых
# клиентов. Это не настройка, а линейка отчёта.
EVENT_CANDIDATES: tuple[int, ...] = (128, 256, 512, 1024, 2048)
TOKEN_CANDIDATES: tuple[int, ...] = (4_000, 8_000, 16_000, 32_000, 64_000)


def measure_group(inputs: DatasetInputs, group: str) -> dict:
    """
    Длины историй группы на её конечном срезе.
    """

    entry = inputs.groups[group]

    cutoff = entry.cutoffs[-1]

    limit = inputs.tokenizer_config.max_pieces_per_value

    events: list[int] = []
    tokens: list[int] = []
    profile_tokens: list[int] = []
    event_tokens: list[int] = []
    milestones: dict[str, int] = {}

    for client_id in entry.clients:

        encoded = encode_history(inputs.artifacts, entry.history(client_id, cutoff), limit)

        events.append(encoded.n_events)
        tokens.append(sum(item.n_tokens for item in encoded.events))
        profile_tokens.append(encoded.profile.n_tokens)

        for item in encoded.events:

            event_tokens.append(item.n_tokens)

            if item.event_type in DEFAULT_MILESTONES:
                milestones[item.event_type] = milestones.get(item.event_type, 0) + 1

    candidates = [
        {
            "limit": f"{events_limit} событий / {tokens_limit} токенов",
            "events_above": sum(1 for value in events if value > events_limit),
            "tokens_above": sum(1 for value in tokens if value > tokens_limit),
        }
        for events_limit, tokens_limit in zip(EVENT_CANDIDATES, TOKEN_CANDIDATES)
    ]

    return {
        "stage": "measure",
        "format_version": FORMAT_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "group": group,
        "cutoff": cutoff.isoformat(),
        "readiness": inputs.readiness,
        "clients": len(entry.clients),
        "events": sum(events),
        "tokens": sum(tokens),
        "lengths": {
            "events_per_client": _lengths(events),
            "tokens_per_client": _lengths(tokens),
            "tokens_per_event": _lengths(event_tokens),
            "tokens_per_profile": _lengths(profile_tokens),
        },
        "candidates": candidates,
        "milestones": dict(sorted(milestones.items())),
    }


def write_measurement(directory: Path, report: dict) -> list[Path]:

    directory = Path(directory) / MEASURE_DIRNAME

    name = f"{report['group']}__{report['cutoff'][:10]}"

    outputs = [directory / f"{name}.json", directory / f"{name}.md"]

    write_json(outputs[0], report)
    write_text(outputs[1], render_measure_md(report))

    return outputs


__all__ = [
    "EVENT_CANDIDATES",
    "MEASURE_DIRNAME",
    "TOKEN_CANDIDATES",
    "measure_group",
    "write_measurement",
]
