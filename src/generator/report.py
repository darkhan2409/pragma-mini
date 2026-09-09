from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    BASE_DIR,
    FEATURE_END,
    HISTORY_START,
    LABEL_END,
    PROFILE_FIELDS,
    RAW_DIR,
    SOURCES,
)
from .persona import draw_persona


# ============================================================
# ИДЕЯ
# ============================================================
#
# Выгрузка одного клиента из готового датасета в читаемый
# markdown: что о нём знает банк, что с ним происходило
# и как это выглядит в единой ленте.
#
#     python -m src.generator.report --client 0
#
# Скрытые поля генератора вынесены в отдельный раздел
# и помечены: в RAW их нет.
# ============================================================


STREAMS = (
    "transactions",
    "product_events",
    "communications",
    "app_screens",
    "app_operations",
    "banners",
)


# ============================================================
# MARKDOWN
# ============================================================


def cell(value: Any) -> str:

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(value, (bool, np.bool_)):
        return "да" if value else "нет"

    if isinstance(value, float):
        # Целые из колонок с пропусками приходят как float.
        if value.is_integer():
            return f"{int(value):,}".replace(",", " ")

        return f"{value:,.2f}".replace(",", " ")

    if isinstance(value, int):
        return f"{value:,}".replace(",", " ")

    return str(value)


def table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    """
    Markdown-таблица без внешних зависимостей.
    """

    if frame.empty:
        return "_нет строк_\n"

    columns = columns or list(frame.columns)

    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"

    rows = [
        "| " + " | ".join(cell(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]

    return "\n".join([header, divider, *rows]) + "\n"


def pairs(items: dict[str, Any], left: str, right: str) -> str:

    header = f"| {left} | {right} |\n|---|---|\n"

    rows = "".join(f"| {name} | {cell(value)} |\n" for name, value in items.items())

    return header + rows


def counts(series: pd.Series, limit: int = 10) -> str:

    frequency = series.value_counts()

    total = int(frequency.sum())

    lines = ["| значение | строк | доля |", "|---|---|---|"]

    for name, value in frequency.head(limit).items():
        lines.append(f"| {name} | {value:,} | {value / total:.1%} |".replace(",", " "))

    if len(frequency) > limit:
        lines.append(f"| _ещё {len(frequency) - limit} значений_ |  |  |")

    return "\n".join(lines) + "\n"


# ============================================================
# ОТЧЁТ
# ============================================================


def build_report(raw_dir: Path, client_id: int) -> str:

    tables = {
        name: pd.read_parquet(raw_dir / f"{name}.parquet")
        for name in list(STREAMS) + ["profile", "timeline", "labels", "source_coverage"]
    }

    data = {
        name: frame[frame.client_id == client_id].reset_index(drop=True)
        for name, frame in tables.items()
    }

    out: list[str] = []
    add = out.append

    # --------------------------------------------------------
    # ШАПКА
    # --------------------------------------------------------

    add(f"# Клиент {client_id}\n")
    add(
        f"Выгрузка из `{raw_dir.as_posix()}`. "
        f"Окно признаков `{HISTORY_START:%Y-%m-%d}` .. `{FEATURE_END:%Y-%m-%d}`, "
        f"окно метки до `{LABEL_END:%Y-%m-%d}`.\n"
    )
    add(
        "В RAW попадает только окно признаков. Всё, что видно ниже, "
        "клиент и банк уже пережили к моменту среза.\n"
    )

    # --------------------------------------------------------
    # ПРОФИЛЬ НА СРЕЗ
    # --------------------------------------------------------

    profile = data["profile"].sort_values("ts")

    add("## Профиль на срез\n")

    if profile.empty:
        add("_профиля нет_\n")
    else:
        last = profile.iloc[-1]

        add(
            f"Последний снимок: `{last.snapshot_month:%Y-%m}`, "
            f"рассчитан `{last.ts:%Y-%m-%d %H:%M:%S}`.\n"
        )
        add(pairs({field: last[field] for field in PROFILE_FIELDS}, "поле", "значение"))
        add("")

    # --------------------------------------------------------
    # ПОКРЫТИЕ
    # --------------------------------------------------------

    add("## Где банк его видит\n")
    add(
        "`availability_start` это дата подключения источника, "
        "`first_seen` это момент появления в нём клиента. "
        "Прочерк значит, что клиент в источнике не появляется никогда.\n"
    )

    coverage = data["source_coverage"].copy()
    coverage["source"] = pd.Categorical(coverage.source, categories=list(SOURCES), ordered=True)
    coverage = coverage.sort_values("source")

    add(table(coverage, ["source", "availability_start", "first_seen"]))
    add("")

    # --------------------------------------------------------
    # ОБЪЁМЫ
    # --------------------------------------------------------

    add("## Что произошло за 24 месяца\n")

    volumes = pd.DataFrame(
        [
            {
                "поток": name,
                "строк": len(data[name]),
                "первое": data[name].ts.min() if len(data[name]) else None,
                "последнее": data[name].ts.max() if len(data[name]) else None,
            }
            for name in STREAMS
        ]
    )

    add(table(volumes))
    add(f"\nВсего в ленте: **{len(data['timeline']):,}** событий.\n".replace(",", " "))

    # --------------------------------------------------------
    # ДОГОВОРЫ
    # --------------------------------------------------------

    add("## Договоры\n")
    add(
        "Реестр открытий. Он старше окна наблюдения, поэтому здесь есть "
        "договоры, заключённые до начала истории. `timestamp_quality` "
        "показывает, сохранилось ли время: у карт оно есть, у кредитов "
        "и страховок только дата.\n"
    )
    add(
        table(
            data["product_events"].sort_values("ts"),
            ["ts", "product_type", "product_subtype", "amount_or_limit", "term", "timestamp_quality"],
        )
    )
    add("")

    # --------------------------------------------------------
    # ТРАНЗАКЦИИ
    # --------------------------------------------------------

    transactions = data["transactions"]

    add("## Транзакции\n")

    if not transactions.empty:

        add(
            pairs(
                {
                    "всего": len(transactions),
                    "списаний": int((transactions.direction == "debit").sum()),
                    "зачислений": int((transactions.direction == "credit").sum()),
                    "онлайн": f"{transactions.is_online.mean():.1%}",
                    "подписок": int(transactions.is_subscription.sum()),
                    "за рубежом": int((transactions.merchant_country != "KZ").sum()),
                    "город не разобран": int(transactions.merchant_city.isna().sum()),
                    "медианная сумма": float(transactions.amount.median()),
                },
                "показатель",
                "значение",
            )
        )

        add("\n**Частые MCC**\n")
        add(counts(transactions.mcc, limit=8))

        add("\n**Регулярные списания (подписки)**\n")

        subscriptions = transactions[transactions.is_subscription].sort_values("ts")

        add(table(subscriptions.head(6), ["ts", "amount", "mcc", "merchant_city"]))

        add("\n**Первые операции окна**\n")
        add(
            table(
                transactions.sort_values("ts").head(10),
                ["ts", "amount", "direction", "mcc", "merchant_city", "merchant_country", "is_online", "is_subscription"],
            )
        )

    add("")

    # --------------------------------------------------------
    # КОММУНИКАЦИИ
    # --------------------------------------------------------

    communications = data["communications"]

    add("## Коммуникации банка\n")
    add(
        "Кампании в контракте нет: бизнес-категория лежит в недоступной "
        "витрине, и отличить кредитное предложение от сервисного "
        "сообщения можно только по префиксу шаблона.\n"
    )

    if not communications.empty:

        delivery = {
            channel: f"{group.delivered.mean():.1%} из {len(group)}"
            for channel, group in communications.groupby("channel")
        }

        add("**Доставка по каналам**\n")
        add(pairs(delivery, "канал", "доставлено"))

        add("\n**Частые шаблоны**\n")
        add(counts(communications.template, limit=8))

        add("\n**Примеры**\n")
        add(
            table(
                communications.sort_values("ts").head(8),
                ["ts", "channel", "template", "day_of_week", "hour", "delivered"],
            )
        )

    add("")

    # --------------------------------------------------------
    # ПРИЛОЖЕНИЕ
    # --------------------------------------------------------

    screens = data["app_screens"]
    operations = data["app_operations"]

    add("## Приложение\n")

    if screens.empty:
        add("Клиент приложением не пользуется: экранов и операций нет.\n")
    else:
        add(
            pairs(
                {
                    "экранов": len(screens),
                    "сессий": int(screens.session_id.nunique()),
                    "экранов за сессию": round(len(screens) / max(1, screens.session_id.nunique()), 1),
                    "операций": len(operations),
                    "заявок в воронке": int((screens.funnel_stage == "application").sum()),
                    "одобрено": int((screens.funnel_stage == "approved").sum()),
                    "отказов": int((screens.funnel_stage == "rejected").sum()),
                },
                "показатель",
                "значение",
            )
        )

        funnel = screens[screens.funnel_stage.notna()].sort_values("ts")

        if not funnel.empty:
            add("\n**Воронка заявки**\n")
            add(
                table(
                    funnel,
                    ["ts", "session_id", "firebase_screen", "product", "funnel_stage", "reject_reason"],
                )
            )

        add("\n**Одна сессия целиком**\n")

        session_id = screens.session_id.value_counts().index[0]
        session = screens[screens.session_id == session_id].sort_values("ts")

        add(table(session, ["ts", "firebase_screen", "product"]))

        if not operations.empty:
            add("\n**Операции по доменам**\n")
            add(counts(operations.domain))

    add("")

    # --------------------------------------------------------
    # БАННЕРЫ
    # --------------------------------------------------------

    banners = data["banners"]

    add("## Баннеры\n")

    if banners.empty:
        add("_показов не было_\n")
    else:
        shown = int((banners.action == "shown").sum())
        clicked = int((banners.action == "clicked").sum())

        add(
            pairs(
                {
                    "показов": shown,
                    "кликов": clicked,
                    "CTR": f"{clicked / max(1, shown):.2%}",
                },
                "показатель",
                "значение",
            )
        )

        clicks = banners[banners.action == "clicked"].sort_values("ts")

        if not clicks.empty:
            add("\n**Клики**\n")
            add(table(clicks, ["ts", "slot", "offer"]))

    add("")

    # --------------------------------------------------------
    # ПРОФИЛЬ ПО МЕСЯЦАМ
    # --------------------------------------------------------

    add("## Профиль по месяцам\n")
    add(
        "Пропуски приходят блоками: отваливается источник, а не отдельное "
        "поле. Прочерк не означает «значения нет», он означает «витрина его "
        "не отдала».\n"
    )

    columns = [
        "snapshot_month", "age", "declared_income", "income_type",
        "relationship_months", "contracts_count", "active_contracts",
        "holds_credit_card", "holds_deposit", "credit_limit", "credit_utilization",
    ]

    add(table(profile, columns))
    add("")

    # --------------------------------------------------------
    # ЛЕНТА
    # --------------------------------------------------------

    timeline = data["timeline"].sort_values("seq")

    # seq это порядковый номер, а не величина: без разрядов.
    timeline = timeline.assign(seq=timeline.seq.astype(str))

    add("## Единая лента\n")
    add(
        "Контракт события: `client_id | ts | event_type | payload`. "
        "`seq` разрешает совпадающие метки времени и информации не несёт.\n"
    )

    add("**Состав**\n")
    add(counts(timeline.event_type))

    same = timeline[timeline.duplicated(subset=["ts"], keep=False)]

    add(f"\nСобытий с неуникальным `ts`: **{len(same)}** из {len(timeline)}.\n")

    if not same.empty:
        add("\n**Пример совпадающих меток времени**\n")

        first_ts = same.ts.iloc[0]

        add(table(same[same.ts == first_ts], ["seq", "ts", "event_type", "payload"]))

    add("\n**Начало ленты**\n")
    add(table(timeline.head(15), ["seq", "ts", "event_type", "payload"]))

    add("\n**Конец ленты**\n")
    add(table(timeline.tail(10), ["seq", "ts", "event_type", "payload"]))

    add("")

    # --------------------------------------------------------
    # МЕТКА
    # --------------------------------------------------------

    add("## Метка\n")

    label = data["labels"].iloc[0]

    add(
        pairs(
            {
                "окно": f"{label.label_start:%Y-%m-%d} .. {label.label_end:%Y-%m-%d}",
                "product_open_90d": label.product_open_90d,
            },
            "поле",
            "значение",
        )
    )

    add("")

    # --------------------------------------------------------
    # СКРЫТОЕ
    # --------------------------------------------------------

    persona = draw_persona(client_id)

    add("## Скрытое: чего в RAW нет\n")
    add(
        "Эти поля порождают поведение клиента, но модели не показываются "
        "никогда. Раздел приведён, чтобы было видно причину происходящего. "
        "Имена в коде оставлены как есть, рядом дан смысл.\n"
    )

    latent = (
        (
            "activity",
            "Общая активность",
            persona.activity,
            "как часто клиент вообще что-то делает: покупки, заходы в приложение",
        ),
        (
            "digital_affinity",
            "Цифровая привычка",
            persona.digital_affinity,
            "онлайн-покупки, приложение, готовность читать push",
        ),
        (
            "mobility",
            "Подвижность",
            persona.mobility,
            "такси, топливо, поездки и операции за рубежом",
        ),
        (
            "credit_need",
            "Потребность в заёмных деньгах",
            persona.credit_need,
            "тяга к кредитам и картам, интерес к разделу займов",
        ),
        (
            "risk",
            "Рискованность",
            persona.risk,
            "постоянный фон проблем с платежами и отказов по заявкам",
        ),
        (
            "volatility",
            "Неустойчивость",
            persona.volatility,
            "насколько вероятен спад активности вплоть до ухода",
        ),
        (
            "push_reachable",
            "Уведомления включены",
            persona.push_reachable,
            "дойдёт ли push, если приложение установлено",
        ),
    )

    lines = ["| поле в коде | что это | значение | смысл |", "|---|---|---|---|"]

    for name, title, value, meaning in latent:
        shown = cell(round(value, 3) if isinstance(value, float) else value)
        lines.append(f"| `{name}` | {title} | {shown} | {meaning} |")

    add("\n".join(lines) + "\n")

    return "\n".join(out)


def main() -> None:

    parser = argparse.ArgumentParser(description="Выгрузка одного клиента в markdown")

    parser.add_argument("--client", type=int, default=0)
    parser.add_argument("--raw", type=Path, default=RAW_DIR / "smoke")
    parser.add_argument("--out", type=Path, default=None)

    args = parser.parse_args()

    out = args.out or BASE_DIR / f"client_{args.client}.md"

    text = build_report(args.raw, args.client)

    out.write_text(text, encoding="utf-8")

    print(f"записано: {out}  ({len(text):,} знаков)".replace(",", " "))


if __name__ == "__main__":
    main()
