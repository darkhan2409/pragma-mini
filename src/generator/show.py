from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import pandas as pd

from .config import FEATURE_END, HISTORY_START, LABEL_END
from .coverage import first_seen
from .history import STREAM_FIELDS, generate_client_history, observed
from .persona import draw_persona
from .profile import snapshot_row
from .timeline import timeline_rows
from .version import DEFAULT_VERSION, VERSIONS


# ============================================================
# ИДЕЯ
# ============================================================
#
# Посмотреть ОДИН поток одного клиента, не собирая датасет.
#
#     python -m src.generator.show transactions --client 7
#
# Показывается ровно то, что попало бы в RAW: окно признаков,
# фильтр покрытия и наблюдательный шум уже применены.
# ============================================================


STREAMS = tuple(STREAM_FIELDS) + ("timeline", "persona")


def stream_rows(history, stream: str) -> list[dict]:

    if stream == "timeline":
        return timeline_rows(history)

    if stream == "profile":
        return [snapshot_row(snapshot) for snapshot in history.profile]

    return [asdict(event) for event in history.events(stream)]


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Показать один поток одного клиента"
    )

    parser.add_argument("stream", choices=STREAMS)
    parser.add_argument("--client", type=int, default=7)
    parser.add_argument("--limit", type=int, default=20, help="0 = все строки")
    parser.add_argument("--month", default=None, help="фильтр YYYY-MM")
    parser.add_argument(
        "--future",
        action="store_true",
        help="показать окно метки вместо окна признаков",
    )
    parser.add_argument("--version", choices=VERSIONS, default=DEFAULT_VERSION)

    args = parser.parse_args()

    client_id = args.client

    # ========================================================
    # ПЕРСОНА
    # ========================================================

    if args.stream == "persona":

        persona = draw_persona(client_id)

        print(f"КЛИЕНТ {client_id}")
        print()
        print("наблюдаемое банком:")
        for field in (
            "birth_date", "gender", "family_status", "children", "education",
            "region", "housing_type", "income_type", "industry",
            "declared_income", "salary_day", "relationship_start",
        ):
            print(f"  {field:22s}{getattr(persona, field)}")

        print()
        print("скрытое, в RAW не попадает:")
        for field in (
            "activity", "digital_affinity", "mobility",
            "credit_need", "risk", "volatility", "push_reachable",
        ):
            value = getattr(persona, field)
            value = round(value, 3) if isinstance(value, float) else value
            print(f"  {field:22s}{value}")

        return

    # ========================================================
    # ПОТОК
    # ========================================================

    history = generate_client_history(client_id, version=args.version)

    window = (
        history.since(FEATURE_END) if args.future else history.before(FEATURE_END)
    )

    rows = stream_rows(observed(window), args.stream)

    frame = pd.DataFrame(rows)

    print(f"КЛИЕНТ {client_id}  |  поток {args.stream}  |  {args.version}")

    if args.stream in STREAM_FIELDS:
        seen = first_seen(client_id, args.stream)
        print(f"виден в источнике с: {seen if seen else 'никогда'}")

    print(
        f"окно: {'метка' if args.future else 'признаки'} "
        f"[{HISTORY_START:%Y-%m-%d} .. "
        f"{(LABEL_END if args.future else FEATURE_END):%Y-%m-%d})"
    )
    print(f"строк: {len(frame)}")

    if frame.empty:
        return

    if args.month:
        frame = frame[frame.ts.astype(str).str.startswith(args.month)]
        print(f"после фильтра {args.month}: {len(frame)}")

    if args.limit:
        frame = frame.head(args.limit)

    print()

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", 60,
    ):
        print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
