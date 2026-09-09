from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.generator.config import EVENT_TYPES
from src.preprocessing.artifacts import write_json, write_text
from src.tokenizer.build import check_order, client_runs, iter_client_blocks

from .data import select_clients


# ============================================================
# ИДЕЯ
# ============================================================
#
# Лимит истории задан в СОБЫТИЯХ, а поведение клиента живёт в
# КАЛЕНДАРЕ. Сколько месяцев видит модель при лимите 128, из
# длины окна не следует: у активного клиента это недели, у
# спящего годы.
#
# Две величины, которые легко перепутать:
#
#   доля обрезанных ПРИМЕРОВ   у скольких историй сработал лимит
#   доля отброшенных СОБЫТИЙ   какая часть ленты не дошла до модели
#
# Они разные, и отчёт печатает обе рядом.
#
# Модель здесь не участвует вовсе. Считается по processed, где
# значения лежат в исходном виде: там есть session_id и сырые
# суммы, которых в токенах нет.
# ============================================================


WINDOWS: tuple[int | None, ...] = (128, 512, 1024, None)

DAY = np.timedelta64(1, "D")

FULL = "full"


def window_name(window: int | None) -> str:
    return FULL if window is None else str(window)


# ============================================================
# ПРАВИЛА СВЯЗЕЙ
# ============================================================
#
# Объявляются до счёта. Достоверный идентификатор в этих данных
# ровно один: session_id у экранов приложения. Всё остальное это
# эвристика, и она названа эвристикой.

STATUS_IDENTIFIER = "identifier"
STATUS_SERIES = "series_identifier"
STATUS_HEURISTIC = "heuristic"
STATUS_WEAK = "weak_heuristic"


LINK_RULES: tuple[dict, ...] = (
    {
        "name": "subscription_repeat",
        "successor": "transaction",
        "status": STATUS_SERIES,
        "key": "(клиент, mcc, сумма) при is_subscription = true",
        "why": (
            "у клиента не больше одной подписки на категорию, наборы mcc категорий не "
            "пересекаются, сумма повторяется до тенге. Город в ключ не входит: он "
            "перерисовывается на каждое списание"
        ),
        "not_proof": "сохранённого идентификатора платежа в данных нет",
    },
    {
        "name": "merchant_repeat",
        "successor": "transaction",
        "status": STATUS_WEAK,
        "key": "(клиент, mcc, merchant_city, is_online) при is_subscription = false",
        "why": "повтор категории в том же городе тем же способом оплаты",
        "not_proof": (
            "поля магазина в данных нет вовсе, поэтому это не «тот же магазин», а «та же "
            "категория в том же городе». Сумма в ключ не входит: она рисуется заново"
        ),
    },
    {
        "name": "funnel_session",
        "successor": "app_screen",
        "status": STATUS_IDENTIFIER,
        "key": "app_screen__session_id у экранов с непустым funnel_stage",
        "why": "session_id общий у всех экранов одной заявки и заполнен всегда",
        "not_proof": "у app_operation session_id отсутствует, связь экран → операция им не проверяется",
    },
    {
        "name": "funnel_to_contract",
        "successor": "product_event",
        "status": STATUS_HEURISTIC,
        "key": "app_screen(funnel_stage=approved, product=X) → product_event(product_type=X)",
        "why": (
            "договор открывается после одобрения того же продукта; окно двустороннее, от суток "
            "назад до семи часов вперёд, потому что timestamp_quality = date_only округляет "
            "время договора вниз до полуночи и он оказывается в ленте раньше экранов"
        ),
        "not_proof": (
            "договоры приходят и без воронки: офлайн-канал и debit_card, которому нет "
            "соответствия среди значений app_screen__product"
        ),
    },
    {
        "name": "screen_to_operation",
        "successor": "app_operation",
        "status": STATUS_HEURISTIC,
        "key": "app_screen домена D → app_operation того же домена через 1–39 секунд",
        "why": "операция и порождается экраном своего домена внутри сессии просмотра",
        "not_proof": (
            "общего ключа у экрана и операции нет; вход домена auth стоит в начале сессии и "
            "экраном не вызван, поэтому из правила исключён"
        ),
    },
)


# Домен операции по префиксу кода экрана.
SCREEN_DOMAIN: dict[str, str] = {
    "s_1": "cards",
    "s_2": "transfers",
    "s_3": "payments",
    "s_4": "loans",
    "s_5": "deposits",
    "s_7": "market",
    "s_9": "support",
}

OPERATION_WINDOW_SECONDS = (1, 39)

CONTRACT_WINDOW_DAYS = (-1.0, 7.0 / 24.0)

AUTH_DOMAIN = "auth"


# ============================================================
# ЗАГРУЗКА
# ============================================================


COLUMNS: tuple[str, ...] = (
    "client_id",
    "seq",
    "ts",
    "event_type",
    "transaction__mcc",
    "transaction__merchant_city",
    "transaction__amount",
    "transaction__is_subscription",
    "transaction__is_online",
    "app_screen__session_id",
    "app_screen__firebase_screen",
    "app_screen__product",
    "app_screen__funnel_stage",
    "app_operation__domain",
    "product_event__product_type",
)


@dataclass(frozen=True)
class Timeline:
    """
    Ленты выбранных клиентов и их примеры.
    """

    client_ids: list[int]
    starts: dict[int, int]
    ends: dict[int, int]
    columns: dict[str, np.ndarray]
    examples: dict[str, np.ndarray]

    @property
    def n_examples(self) -> int:
        return int(self.examples["client_id"].size)

    def slice_of(self, client: int, seq_end: int) -> slice:
        lo = self.starts[client]
        return slice(lo, lo + int(seq_end))


def load_timeline(processed_dir: Path, split: str, max_clients: int | None) -> Timeline:
    """
    Примеры сплита и полные ленты его клиентов, один проход по файлу.
    """

    processed_dir = Path(processed_dir)

    table = pq.read_table(
        processed_dir / split / "examples.parquet",
        columns=["client_id", "cutoff", "seq_end", "n_events", "client_group"],
    )

    clients = select_clients(table, max_clients)

    if not clients:
        raise ValueError(f"{split}: не нашлось ни одного клиента")

    wanted = set(clients)

    group = table.column("client_group").to_pylist()[0]

    keep = np.isin(table.column("client_id").to_numpy().astype(np.int64), np.array(clients))

    rows = table.filter(pa.array(keep))

    order = np.lexsort(
        (
            rows.column("cutoff").to_numpy().astype("datetime64[us]").astype(np.int64),
            rows.column("client_id").to_numpy().astype(np.int64),
        )
    )

    examples = {
        "client_id": rows.column("client_id").to_numpy().astype(np.int64)[order],
        "cutoff": rows.column("cutoff").to_numpy().astype("datetime64[s]")[order],
        "seq_end": rows.column("seq_end").to_numpy().astype(np.int64)[order],
    }

    # --------------------------------------------------------

    path = processed_dir / "clients" / f"{group}_clients" / "events.parquet"

    limit = max(wanted)

    pieces: dict[int, list[pa.Table]] = {}

    for block in iter_client_blocks(path, columns=list(COLUMNS)):

        cid = block.column("client_id").to_numpy()

        if cid.size == 0:
            continue

        if int(cid[0]) > limit:
            break

        for value, lo, hi in client_runs(cid):
            if value in wanted:
                pieces.setdefault(value, []).append(block.slice(lo, hi - lo))

    missing = sorted(wanted - set(pieces))

    if missing:
        raise ValueError(f"{split}: нет событий клиентов {missing[:5]}")

    joined = pa.concat_tables([piece for client in clients for piece in pieces[client]])

    columns = {name: _column(joined, name) for name in COLUMNS}

    check_order(columns["client_id"], columns["ts"], columns["seq"])

    starts: dict[int, int] = {}
    ends: dict[int, int] = {}

    for value, lo, hi in client_runs(columns["client_id"]):
        starts[value] = lo
        ends[value] = hi

    return Timeline(clients, starts, ends, columns, examples)


def _column(table: pa.Table, name: str) -> np.ndarray:

    column = table.column(name)

    if name == "ts":
        return column.to_numpy().astype("datetime64[s]")

    if name in ("client_id", "seq"):
        return column.to_numpy().astype(np.int64)

    return np.array(column.to_pylist(), dtype=object)


# ============================================================
# ПОКРЫТИЕ
# ============================================================


def _quantiles(values: np.ndarray) -> dict:

    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return {"p50": None, "p90": None, "p95": None, "max": None, "mean": None}

    return {
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def window_stats(timeline: Timeline, window: int | None) -> dict:
    """
    По каждому примеру: что осталось в окне и сколько это времени.
    """

    ts = timeline.columns["ts"]
    kind = timeline.columns["event_type"]

    clients = timeline.examples["client_id"]
    cutoffs = timeline.examples["cutoff"]
    seq_end = timeline.examples["seq_end"]

    n = clients.size

    original = seq_end.astype(np.int64)
    used = original if window is None else np.minimum(original, window)

    first_to_cutoff = np.full(n, np.nan)
    last_to_cutoff = np.full(n, np.nan)
    span = np.full(n, np.nan)

    counts = {name: np.zeros(n, dtype=np.int64) for name in EVENT_TYPES}

    empty = 0

    for index in range(n):

        if used[index] == 0:
            empty += 1
            continue

        client = int(clients[index])

        lo = timeline.starts[client]

        stop = lo + int(original[index])
        start = stop - int(used[index])

        window_ts = ts[start:stop]

        # Будущее исключено по построению: префикс это seq < seq_end,
        # а seq_end это число событий строго до cutoff.
        cutoff = cutoffs[index]

        first_to_cutoff[index] = (cutoff - window_ts[0]) / DAY
        last_to_cutoff[index] = (cutoff - window_ts[-1]) / DAY
        span[index] = (window_ts[-1] - window_ts[0]) / DAY

        types, amounts = np.unique(kind[start:stop], return_counts=True)

        for name, amount in zip(types, amounts):
            if name in counts:
                counts[str(name)][index] = amount

    truncated = original > used

    kept_share = np.divide(used, original, out=np.ones(n), where=original > 0)

    return {
        "window": window_name(window),
        "n_examples": int(n),
        "n_empty_histories": int(empty),
        "original_events": _quantiles(original),
        "used_events": _quantiles(used),
        "truncated_examples_share": float(truncated.mean()) if n else 0.0,
        "dropped_events_share": (
            float(1.0 - used.sum() / original.sum()) if original.sum() else 0.0
        ),
        "kept_events_share": _quantiles(kept_share),
        "days_first_to_cutoff": _quantiles(first_to_cutoff[~np.isnan(first_to_cutoff)]),
        "days_last_to_cutoff": _quantiles(last_to_cutoff[~np.isnan(last_to_cutoff)]),
        "days_span": _quantiles(span[~np.isnan(span)]),
        "events_by_type": {
            name: _quantiles(values) for name, values in sorted(counts.items())
        },
    }


# ============================================================
# СВЯЗИ
# ============================================================


def _predecessor_state(rule: str, timeline: Timeline, index: int) -> tuple | None:
    """
    Ключ связи события. None означает, что событие правилу не подходит.
    """

    columns = timeline.columns

    kind = columns["event_type"][index]

    if rule == "subscription_repeat":

        if kind != "transaction" or not columns["transaction__is_subscription"][index]:
            return None

        return (columns["transaction__mcc"][index], columns["transaction__amount"][index])

    if rule == "merchant_repeat":

        if kind != "transaction" or columns["transaction__is_subscription"][index]:
            return None

        return (
            columns["transaction__mcc"][index],
            columns["transaction__merchant_city"][index],
            columns["transaction__is_online"][index],
        )

    if rule == "funnel_session":

        if kind != "app_screen" or columns["app_screen__funnel_stage"][index] is None:
            return None

        return (columns["app_screen__session_id"][index],)

    return None


def _links_by_key(rule: str, timeline: Timeline, span: slice) -> list[tuple[int, int]]:
    """
    Пары (последователь, ближайший предшественник) внутри ленты.
    """

    last: dict[tuple, int] = {}

    pairs: list[tuple[int, int]] = []

    for index in range(span.start, span.stop):

        key = _predecessor_state(rule, timeline, index)

        if key is None:
            continue

        if key in last:
            pairs.append((index, last[key]))

        last[key] = index

    return pairs


def _links_funnel_to_contract(timeline: Timeline, span: slice) -> list[tuple[int, int]]:

    columns = timeline.columns

    kind = columns["event_type"]
    ts = columns["ts"]

    approved = [
        index
        for index in range(span.start, span.stop)
        if kind[index] == "app_screen"
        and columns["app_screen__funnel_stage"][index] == "approved"
    ]

    pairs: list[tuple[int, int]] = []

    low, high = CONTRACT_WINDOW_DAYS

    for index in range(span.start, span.stop):

        if kind[index] != "product_event":
            continue

        product = columns["product_event__product_type"][index]

        best = None

        for screen in approved:

            if columns["app_screen__product"][screen] != product:
                continue

            delta = (ts[index] - ts[screen]) / DAY

            # Окно двустороннее: date_only опускает время
            # договора до полуночи, и он встаёт раньше экрана.
            if low <= delta <= high and (best is None or abs(delta) < best[1]):
                best = (screen, abs(delta))

        if best is not None:
            pairs.append((index, best[0]))

    return pairs


def _links_screen_to_operation(timeline: Timeline, span: slice) -> list[tuple[int, int]]:

    columns = timeline.columns

    kind = columns["event_type"]
    ts = columns["ts"]

    low, high = OPERATION_WINDOW_SECONDS

    pairs: list[tuple[int, int]] = []

    screens = [index for index in range(span.start, span.stop) if kind[index] == "app_screen"]

    for index in range(span.start, span.stop):

        if kind[index] != "app_operation":
            continue

        domain = columns["app_operation__domain"][index]

        # Вход в начале сессии экраном не вызван.
        if domain == AUTH_DOMAIN:
            continue

        best = None

        for screen in screens:

            if screen >= index:
                break

            code = columns["app_screen__firebase_screen"][screen]

            if code is None or SCREEN_DOMAIN.get(str(code)[:3]) != domain:
                continue

            delta = (ts[index] - ts[screen]) / np.timedelta64(1, "s")

            if low <= delta <= high:
                best = screen

        if best is not None:
            pairs.append((index, best))

    return pairs


def link_stats(timeline: Timeline, windows=WINDOWS) -> dict:
    """
    Сколько найденных связей переживает обрезку каждого окна.
    """

    clients = timeline.examples["client_id"]
    seq_end = timeline.examples["seq_end"]

    tally = {
        rule["name"]: {
            window_name(window): {"successors": 0, "inside": 0, "cut_off": 0, "no_predecessor": 0}
            for window in windows
        }
        for rule in LINK_RULES
    }

    for index in range(clients.size):

        client = int(clients[index])

        lo = timeline.starts[client]

        stop = lo + int(seq_end[index])

        span = slice(lo, stop)

        found = {
            "subscription_repeat": _links_by_key("subscription_repeat", timeline, span),
            "merchant_repeat": _links_by_key("merchant_repeat", timeline, span),
            "funnel_session": _links_by_key("funnel_session", timeline, span),
            "funnel_to_contract": _links_funnel_to_contract(timeline, span),
            "screen_to_operation": _links_screen_to_operation(timeline, span),
        }

        for rule in LINK_RULES:

            name = rule["name"]

            pairs = dict(found[name])

            successors = _successors(rule, timeline, span)

            for window in windows:

                left = lo if window is None else max(lo, stop - int(window))

                cell = tally[name][window_name(window)]

                for successor in successors:

                    if successor < left:
                        continue

                    cell["successors"] += 1

                    predecessor = pairs.get(successor)

                    if predecessor is None:
                        cell["no_predecessor"] += 1
                    elif predecessor >= left:
                        cell["inside"] += 1
                    else:
                        cell["cut_off"] += 1

    return {
        rule["name"]: {
            "rule": rule,
            "windows": {
                name: {
                    **cell,
                    "with_predecessor": cell["inside"] + cell["cut_off"],
                    "inside_share": (
                        cell["inside"] / (cell["inside"] + cell["cut_off"])
                        if cell["inside"] + cell["cut_off"]
                        else None
                    ),
                    "cut_off_share": (
                        cell["cut_off"] / (cell["inside"] + cell["cut_off"])
                        if cell["inside"] + cell["cut_off"]
                        else None
                    ),
                    "reliable": (cell["inside"] + cell["cut_off"]) >= 30,
                }
                for name, cell in tally[rule["name"]].items()
            },
        }
        for rule in LINK_RULES
    }


def _successors(rule: dict, timeline: Timeline, span: slice) -> list[int]:
    """
    События, у которых предшественник в принципе может быть.
    """

    kind = timeline.columns["event_type"]

    columns = timeline.columns

    out: list[int] = []

    for index in range(span.start, span.stop):

        if kind[index] != rule["successor"]:
            continue

        if rule["name"] == "subscription_repeat":
            if not columns["transaction__is_subscription"][index]:
                continue

        elif rule["name"] == "merchant_repeat":
            if columns["transaction__is_subscription"][index]:
                continue

        elif rule["name"] == "funnel_session":
            if columns["app_screen__funnel_stage"][index] is None:
                continue

        elif rule["name"] == "screen_to_operation":
            if columns["app_operation__domain"][index] == AUTH_DOMAIN:
                continue

        out.append(index)

    return out


# ============================================================
# ЗАПУСК
# ============================================================


def run_history_coverage(
    processed_dir: Path,
    out_dir: Path,
    splits: dict[str, int | None],
    windows=WINDOWS,
    with_links: bool = True,
    quiet: bool = False,
) -> dict:
    """
    Глубина recent-окон и судьба наблюдаемых связей.
    """

    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    coverage: dict[str, dict] = {}
    links: dict[str, dict] = {}
    sizes: dict[str, dict] = {}

    for split, max_clients in splits.items():

        if not quiet:
            print(f"  сплит {split}")

        timeline = load_timeline(processed_dir, split, max_clients)

        sizes[split] = {
            "clients": len(timeline.client_ids),
            "examples": timeline.n_examples,
            "events_in_memory": int(timeline.columns["ts"].size),
        }

        coverage[split] = {window_name(w): window_stats(timeline, w) for w in windows}

        if with_links:
            links[split] = link_stats(timeline, windows)

    report = {
        "mode": "history_coverage",
        "processed_dir": str(processed_dir),
        "windows": [window_name(w) for w in windows],
        "splits": sizes,
        "coverage": coverage,
        "links": links,
        "link_rules": list(LINK_RULES),
        "note": (
            "доля обрезанных примеров и доля отброшенных событий это разные величины: "
            "лимит может срабатывать у всех историй и при этом отбрасывать разную часть ленты"
        ),
    }

    write_json(out_dir / "history_coverage.json", report)
    write_text(out_dir / "history_coverage.md", render_history_coverage(report))
    write_text(out_dir / "link_rules.md", render_link_rules(report))

    if not quiet:
        print()
        print(render_history_coverage(report))

    return report


# ============================================================
# ОТЧЁТЫ
# ============================================================


def _number(value, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_history_coverage(report: dict) -> str:

    lines: list[str] = []

    lines.append("# Календарная глубина recent-окон")
    lines.append("")
    lines.append(report["note"] + ".")
    lines.append("")

    lines.append("| сплит | клиентов | примеров | событий в лентах |")
    lines.append("|---|---|---|---|")

    for split, item in report["splits"].items():
        lines.append(
            f"| {split} | {item['clients']} | {item['examples']} | {item['events_in_memory']} |"
        )

    lines.append("")

    for split, windows in report["coverage"].items():

        lines.append(f"## {split}")
        lines.append("")

        first = windows[report["windows"][0]]

        lines.append(
            f"Длина истории: p50 {_number(first['original_events']['p50'], 0)}, "
            f"p90 {_number(first['original_events']['p90'], 0)}, "
            f"max {_number(first['original_events']['max'], 0)} событий. "
            f"Пустых историй: {first['n_empty_histories']}."
        )
        lines.append("")

        lines.append(
            "| окно | обрезано примеров | отброшено событий | сохранено, p50 | "
            "глубина p50 | p90 | p95 | max |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")

        for name in report["windows"]:

            item = windows[name]

            depth = item["days_first_to_cutoff"]

            lines.append(
                f"| {name} | {item['truncated_examples_share'] * 100:.0f} % | "
                f"{item['dropped_events_share'] * 100:.0f} % | "
                f"{item['kept_events_share']['p50'] * 100:.0f} % | "
                f"{_number(depth['p50'])} дн | {_number(depth['p90'])} | "
                f"{_number(depth['p95'])} | {_number(depth['max'])} |"
            )

        lines.append("")

        lines.append("Состав окна по типам событий, медиана:")
        lines.append("")

        types = sorted(windows[report["windows"][0]]["events_by_type"])

        lines.append("| окно | " + " | ".join(types) + " |")
        lines.append("|" + "---|" * (len(types) + 1))

        for name in report["windows"]:

            item = windows[name]["events_by_type"]

            lines.append(
                f"| {name} | "
                + " | ".join(_number(item[kind]["p50"], 0) for kind in types)
                + " |"
            )

        lines.append("")

        lines.append(
            f"Последнее событие отстоит от cutoff на "
            f"{_number(windows[report['windows'][0]]['days_last_to_cutoff']['p50'], 2)} дня "
            "по медиане: окно всегда упирается в cutoff справа, обрезается только левый край."
        )
        lines.append("")

    # --------------------------------------------------------

    if not report["links"]:
        return "\n".join(lines) + "\n"

    lines.append("## Судьба наблюдаемых связей")
    lines.append("")
    lines.append(
        "Считаются только события внутри окна. `внутри` это предшественник, оставшийся в окне, "
        "`потеряно` это предшественник левее границы. События без предшественника во всей "
        "доступной на cutoff истории в доли не входят и показаны отдельно."
    )
    lines.append("")

    for split, rules in report["links"].items():

        lines.append(f"### {split}")
        lines.append("")

        for name, section in rules.items():

            rule = section["rule"]

            lines.append(f"**{name}** ({rule['status']}): {rule['key']}")
            lines.append("")
            lines.append("| окно | событий | с предшественником | внутри | потеряно | без связи |")
            lines.append("|---|---|---|---|---|---|")

            for window in report["windows"]:

                cell = section["windows"][window]

                inside = (
                    "—"
                    if cell["inside_share"] is None
                    else f"{cell['inside_share'] * 100:.0f} %"
                )
                lost = (
                    "—"
                    if cell["cut_off_share"] is None
                    else f"{cell['cut_off_share'] * 100:.0f} %"
                )

                note = "" if cell["reliable"] else " (мало данных)"

                lines.append(
                    f"| {window} | {cell['successors']} | {cell['with_predecessor']}{note} | "
                    f"{inside} | {lost} | {cell['no_predecessor']} |"
                )

            lines.append("")

    return "\n".join(lines) + "\n"


def render_link_rules(report: dict) -> str:

    lines: list[str] = []

    lines.append("# Правила поиска связей")
    lines.append("")
    lines.append(
        "Правила объявлены до счёта. Достоверный идентификатор в этих данных ровно один, "
        "остальное это эвристики, и они названы эвристиками. Latent-поля генератора не "
        "используются. Совпадение корзины суммы доказательством одного платежа не считается, "
        "временное соседство доказанной цепочкой не называется."
    )
    lines.append("")

    for rule in report["link_rules"]:

        lines.append(f"## {rule['name']}")
        lines.append("")
        lines.append(f"- статус: **{rule['status']}**")
        lines.append(f"- последователь: `{rule['successor']}`")
        lines.append(f"- ключ: {rule['key']}")
        lines.append(f"- на чём основано: {rule['why']}")
        lines.append(f"- чего правило не доказывает: {rule['not_proof']}")
        lines.append("")

    lines.append("## Чего в данных нет")
    lines.append("")
    lines.append(
        "- идентификатора магазина, транзакции, заявки, договора и кампании; "
        "поле магазина не заводилось вовсе;"
    )
    lines.append(
        "- общего ключа у `app_screen` и `app_operation`: `session_id` есть только у экранов;"
    )
    lines.append(
        "- цепочки `app_screen → app_operation → product_event`: сессия заявки состоит только "
        "из экранов и операций не порождает, поэтому проверяются две отдельные связи."
    )

    return "\n".join(lines) + "\n"
