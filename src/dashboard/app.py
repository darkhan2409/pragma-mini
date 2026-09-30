from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# streamlit run кладёт в sys.path каталог страницы, а не проекта.
ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.dashboard.telemetry import Telemetry, TelemetryReader, progress  # noqa: E402
from src.mlm.settings import TELEMETRY_FILE, TRAIN_DIR  # noqa: E402


# ============================================================
# СТРАНИЦА ДАШБОРДА
# ============================================================
#
# Запуск — python -m src.dashboard (src/dashboard/__main__.py).
#
# Страница перечитывает telemetry.jsonl выбранного каталога раз в
# несколько секунд: перерисовывается только фрагмент с данными, а
# файл дочитывается с того места, где остановился. Потери val —
# точки после эпох, между ними значений нет и не выдумываются.
# ============================================================


# Больше точек на график не отдаётся: шаги сворачиваются в
# средние по соседним окнам, чтобы браузер не тормозил на
# десятках тысяч шагов.
MAX_POINTS = 2000

TRAIN_COLOR = "#1f77b4"
VAL_COLOR = "#d62728"


def arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run", default=None)

    return parser.parse_known_args(sys.argv[1:])[0]


def runs() -> list[Path]:
    """
    Каталоги прогонов: каталог по умолчанию и data/runs/*.
    С телеметрией — первыми.
    """

    found = [TRAIN_DIR]

    folder = TRAIN_DIR.parent / "runs"

    if folder.is_dir():
        found += sorted(path for path in folder.iterdir() if path.is_dir())

    return sorted(found, key=lambda path: not (path / TELEMETRY_FILE).exists())


def label(path: Path) -> str:

    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def thin(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Не больше MAX_POINTS строк: соседние шаги — средним, шаг —
    последний в окне.
    """

    if len(frame) <= MAX_POINTS:
        return frame

    size = -(-len(frame) // MAX_POINTS)
    groups = frame.reset_index(drop=True).groupby(lambda index: index // size)

    thinned = groups.mean(numeric_only=True)
    thinned["step"] = groups["step"].max()

    return thinned


def steps_frame(telemetry: Telemetry, smoothing: int) -> pd.DataFrame:

    frame = pd.DataFrame(
        [telemetry.steps[number] for number in sorted(telemetry.steps)],
        columns=["step", "epoch", "loss", "lr", "grad_norm", "tokens", "seconds"],
    )

    if frame.empty:
        return frame

    if smoothing > 1:
        frame["loss"] = frame["loss"].rolling(smoothing, min_periods=1).mean()

    seconds = frame["seconds"].where(frame["seconds"] > 0)
    frame["tokens_per_second"] = frame["tokens"] / seconds

    return thin(frame)


def loss_chart(steps: pd.DataFrame, telemetry: Telemetry, smoothing: int) -> alt.Chart | None:
    """
    train по шагам — линия, val — только настоящие точки после эпох.
    """

    name = "train" if smoothing <= 1 else f"train (среднее {smoothing})"

    train = pd.DataFrame({"step": steps["step"], "loss": steps["loss"], "series": name}) if not steps.empty else None

    val = pd.DataFrame(
        [
            {"step": record["step"], "epoch": number, "loss": record["val_loss"], "series": "val"}
            for number, record in sorted(telemetry.epochs.items())
            if record.get("val_loss") is not None
        ],
        columns=["step", "epoch", "loss", "series"],
    )

    colors = alt.Scale(domain=[name, "val"], range=[TRAIN_COLOR, VAL_COLOR])
    x = alt.X("step:Q", title="шаг оптимизатора")
    y = alt.Y("loss:Q", title="loss", scale=alt.Scale(zero=False))
    color = alt.Color("series:N", scale=colors, title=None, legend=alt.Legend(orient="top"))

    layers = []

    if train is not None:
        layers.append(
            alt.Chart(train).mark_line(strokeWidth=1.5).encode(
                x=x, y=y, color=color, tooltip=["step", alt.Tooltip("loss", format=".4f")]
            )
        )

    if not val.empty:
        # Отрезки только между соседними измерениями val; точки на
        # тех шагах, где val считался, и больше нигде.
        layers.append(
            alt.Chart(val).mark_line(strokeDash=[4, 3], strokeWidth=1.5, point=alt.OverlayMarkDef(size=70, filled=True)).encode(
                x=x, y=y, color=color,
                tooltip=["epoch", "step", alt.Tooltip("loss", format=".4f")],
            )
        )

    resumes = pd.DataFrame(
        [{"step": record["step"]} for record in telemetry.runs if record.get("resumed")],
        columns=["step"],
    )

    if not resumes.empty:
        layers.append(
            alt.Chart(resumes).mark_rule(color="gray", strokeDash=[2, 2]).encode(
                x="step:Q", tooltip=[alt.Tooltip("step", title="продолжение с шага")]
            )
        )

    if train is None and val.empty:
        return None

    return alt.layer(*layers).properties(height=320).interactive(bind_y=False)


def step_chart(steps: pd.DataFrame, column: str, title: str, rule: float | None = None,
               scale: str = "linear") -> alt.Chart:

    chart = alt.Chart(steps).mark_line(strokeWidth=1.2, color=TRAIN_COLOR).encode(
        x=alt.X("step:Q", title="шаг оптимизатора"),
        y=alt.Y(f"{column}:Q", title=title, scale=alt.Scale(type=scale, zero=False)),
        tooltip=["step", alt.Tooltip(column, format=".4g")],
    )

    if rule is not None:
        chart = chart + alt.Chart(pd.DataFrame({"rule": [rule]})).mark_rule(
            color=VAL_COLOR, strokeDash=[4, 3]
        ).encode(y="rule:Q")

    return chart.properties(height=220).interactive(bind_y=False)


def memory_chart(telemetry: Telemetry) -> alt.Chart | None:
    """
    Пики памяти CUDA по эпохам — то, что обучение уже собирает.
    """

    rows = []

    for number, record in sorted(telemetry.epochs.items()):
        for kind, key in (("allocated", "cuda_peak_allocated_gib"), ("reserved", "cuda_peak_reserved_gib")):
            if record.get(key) is not None:
                rows.append({"epoch": number, "kind": kind, "gib": record[key]})

    if not rows:
        return None

    return alt.Chart(pd.DataFrame(rows)).mark_line(point=True).encode(
        x=alt.X("epoch:O", title="эпоха"),
        y=alt.Y("gib:Q", title="пик памяти, ГиБ"),
        color=alt.Color("kind:N", title=None, legend=alt.Legend(orient="top")),
        tooltip=["epoch", "kind", alt.Tooltip("gib", format=".2f")],
    ).properties(height=220)


def number(value: float | None, pattern: str) -> str:
    return "—" if value is None else format(value, pattern)


def status(telemetry: Telemetry, where: dict) -> str:

    if telemetry.modified is None:
        return "Файла телеметрии нет: в этом каталоге обучение ещё не начиналось."

    age = time.time() - telemetry.modified
    parts = [f"последняя запись {age:.0f} с назад"]

    if where["epochs"] is not None and where["completed_epochs"] >= where["epochs"]:
        parts.append("все эпохи пройдены")
    elif age > 120 and telemetry.steps:
        parts.append("шагов давно нет: идёт val, обучение остановлено или упало")

    if where["resumes"]:
        parts.append(f"продолжений {where['resumes']}")

    if telemetry.skipped:
        parts.append(f"битых строк пропущено {telemetry.skipped}")

    return "; ".join(parts)


def render(path: Path, smoothing: int) -> None:

    key = f"reader:{path}"

    if key not in st.session_state:
        st.session_state[key] = TelemetryReader(path / TELEMETRY_FILE)

    telemetry = st.session_state[key].poll()
    where = progress(telemetry)

    st.caption(f"{label(path / TELEMETRY_FILE)} — {status(telemetry, where)}")

    if not telemetry.runs and not telemetry.steps and not telemetry.epochs:
        return

    last = where["last_step"] or {}
    recent = [telemetry.steps[number] for number in sorted(telemetry.steps)[-20:]]
    speeds = [item["tokens"] / item["seconds"] for item in recent if item.get("seconds") and item.get("tokens")]
    speed = sorted(speeds)[len(speeds) // 2] if speeds else None

    total = where["total_steps"]
    total_text = "—" if total is None else f"{'≈' if where['estimated'] else ''}{total:,}"

    columns = st.columns(8)
    columns[0].metric("Эпоха", f"{where['epoch'] or '—'} / {where['epochs'] or '—'}")
    columns[1].metric("Шаг", f"{where['step']:,} / {total_text}")
    columns[2].metric("Train loss", number(last.get("loss"), ".4f"))
    columns[3].metric("Val loss", number(where["val_loss"], ".4f"), help=f"лучший {number(where['best_val_loss'], '.4f')}")
    columns[4].metric("LR", number(last.get("lr"), ".2e"))
    columns[5].metric("Норма градиента", number(last.get("grad_norm"), ".3f"), help="до клипа")
    columns[6].metric("Токенов/с", number(speed, ",.0f"), help="медиана последних 20 шагов")

    memory = where["last_epoch"] or {}
    columns[7].metric(
        "Пик VRAM", number(memory.get("cuda_peak_allocated_gib"), ".2f") + " ГиБ",
        help=f"прошлая эпоха; reserved {number(memory.get('cuda_peak_reserved_gib'), '.2f')} ГиБ",
    )

    if total:
        st.progress(min(1.0, where["step"] / total))

    steps = steps_frame(telemetry, smoothing)

    losses = loss_chart(steps, telemetry, smoothing)

    if losses is not None:
        st.altair_chart(losses, width="stretch")

    if steps.empty:
        return

    run = telemetry.runs[-1] if telemetry.runs else {}

    left, right = st.columns(2)
    left.altair_chart(step_chart(steps, "lr", "learning rate"), width="stretch")
    right.altair_chart(
        step_chart(steps, "grad_norm", "норма градиента до клипа", rule=run.get("max_grad_norm")),
        width="stretch",
    )

    left, right = st.columns(2)
    left.altair_chart(step_chart(steps, "tokens_per_second", "токенов в секунду"), width="stretch")

    memory = memory_chart(telemetry)

    if memory is not None:
        right.altair_chart(memory, width="stretch")
    else:
        right.caption("Пики памяти CUDA появятся после первой эпохи на CUDA.")

    if telemetry.epochs:
        table = pd.DataFrame([telemetry.epochs[number] for number in sorted(telemetry.epochs)])
        shown = [column for column in (
            "epoch", "step", "train_loss", "val_loss", "learning_rate", "train_seconds", "val_seconds",
            "data_wait_seconds", "grad_norm_mean", "grad_norm_max", "clipped_share",
            "cuda_peak_allocated_gib", "cuda_peak_reserved_gib",
        ) if column in table]

        with st.expander("Эпохи", expanded=False):
            st.dataframe(table[shown], hide_index=True)


def main() -> None:

    st.set_page_config(page_title="PRAGMA: обучение", layout="wide")
    st.title("PRAGMA: обучение")

    given = arguments().run
    choices = runs()

    if given is not None and Path(given).resolve() not in [path.resolve() for path in choices]:
        choices.insert(0, Path(given))

    start = 0

    if given is not None:
        start = [path.resolve() for path in choices].index(Path(given).resolve())

    with st.sidebar:
        picked = st.selectbox("Каталог прогона", choices, index=start, format_func=label)
        typed = st.text_input("или путь", value="", placeholder="data/runs/w4-b0")
        refresh = st.slider("Обновление, с", min_value=2, max_value=5, value=3)
        smoothing = st.number_input("Сглаживание train loss, шагов", min_value=1, max_value=500, value=1)

    path = Path(typed).expanduser() if typed.strip() else Path(picked)

    if not path.is_absolute():
        path = ROOT / path

    @st.fragment(run_every=refresh)
    def live() -> None:
        render(path, int(smoothing))

    live()


# streamlit run исполняет страницу как __main__.
if __name__ == "__main__":
    main()
