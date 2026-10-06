from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from . import lifecycle as lc
from .config import FINAL_GROUP, FUTURE_DIR, FUTURE_LABEL_GROUPS, GROUPS, LOCAL_OFFSET, RAW_DIR, cutoff, period_end
from .raw import PAYLOAD, client_blocks, read_profile
from .sources import FUTURE_EVENTS, read_future


# ============================================================
# ЗАДАЧИ НА СТАДИЯХ CAPP
# ============================================================
#
# Две задачи для мобильной команды рядом с churn_active90 — его они не
# меняют и в TASKS обучения пока не входят. Бизнес-определение у обеих не
# выбрано: каждая неоднозначность — явный режим, и режима по умолчанию
# нет. Выбирает мобильная команда (audit/2026-10-06-lifecycle/README.md).
#
# active_to_at_risk — клиент на T в активной стадии (New, Activated,
# Growing, CORE, Loyal); что случится в окне (T, конец окна]:
#   A1 future_at_risk_strict          первый переход в At Risk в окне;
#                                     уход в Churn, минуя At Risk, — не
#                                     положительный;
#   A2 future_deterioration           At Risk или Churn в окне — любое
#                                     серьёзное ухудшение;
#   A3 stage_specific_early_warning   своя метка у каждой стадии на T
#                                     (A3_OUTCOMES): у Activated — At Risk
#                                     или Churn, у прочих — At Risk.
#                                     Оценивается по стадиям, не общей
#                                     метрикой.
# Тип перехода строки — первое ухудшение в окне: «стадия на T → at_risk»
# или «→ churn»; без ухудшения — none.
# Окно — число дней или NEXT_MONTH: до конца первого полного
# календарного месяца после T. У T на 1-м числе это сам месяц T: правило
# «Growing не в MAU» срабатывает ровно на его конце.
#
# at_risk_outcome — клиент на T уже At Risk. Исход — первый переход после
# T: в активную стадию (recovery) или в Churn (churn). Режимы:
#   B1 binary_completed_only   churn 1 / recovery 0; кто к пределу не
#                              вышел из At Risk, в задачу не входит;
#   B2 fixed_horizon_3class    в горизонте H: recovery 0, churn 1,
#                              still_at_risk 2;
#   B3 time_to_event           данные для анализа выживаемости: первый
#                              исход, время до него или до конца
#                              наблюдения, цензурирование. Не
#                              классификация.
#
# Признаки модели — только история строго до T (features.compute).
# События после T строят лишь историю стадий для метки. Стадия на T,
# прежняя стадия, причины и типы переходов — метаданные для аудита и
# стратификации; в матрицу признаков они не идут (task_matrix).
#
# Наблюдение. История стадий строится до followup_end — конца выгрузки
# (val, test) или конца продолжения (train). Окно, которое данными не
# покрыто, не становится «ничего не случилось»:
#   active_to_at_risk  строка цензурирована, если followup_end < конец
#                      окна: исключается целиком, даже когда ухудшение в
#                      обрезанном окне уже видно, — иначе положительные
#                      оставались бы, а неизвестные отрицательные
#                      выпадали;
#   at_risk_outcome    исход не наступил до предела наблюдения:
#                      still_at_risk, если окно T + H наблюдено целиком,
#                      иначе censored.
# ============================================================


TASK_A = "active_to_at_risk"
TASK_B = "at_risk_outcome"

ACTIVE = frozenset(lc.ACTIVE)

A1 = "future_at_risk_strict"
A2 = "future_deterioration"
A3 = "stage_specific_early_warning"
MODES_A: tuple[str, ...] = (A1, A2, A3)

# A3: какие исходы — положительные для стадии на T (постановка владельца
# 2026-10-06). У Activated правило At Risk действует только 14 дней от
# регистрации, дальше он уходит в Churn напрямую — поэтому и Churn.
A3_OUTCOMES: dict[str, frozenset[str]] = {
    lc.NEW: frozenset({lc.AT_RISK}),
    lc.ACTIVATED: frozenset({lc.AT_RISK, lc.CHURN}),
    lc.GROWING: frozenset({lc.AT_RISK}),
    lc.CORE: frozenset({lc.AT_RISK}),
    lc.LOYAL: frozenset({lc.AT_RISK}),
}

NEXT_MONTH = "next_month"
NONE = "none"

B1 = "binary_completed_only"
B2 = "fixed_horizon_3class"
B3 = "time_to_event"
MODES_B: tuple[str, ...] = (B1, B2, B3)

RECOVERY = "recovery"
CHURN = "churn"
STILL_AT_RISK = "still_at_risk"
CENSORED = "censored"

# Коды классов B2 (у B1 — те же 0 и 1).
CLASSES_B: dict[str, int] = {RECOVERY: 0, CHURN: 1, STILL_AT_RISK: 2}

# Колонки строки задачи: ключ, режим, окно, метка.
ROW_KEYS: tuple[str, ...] = (
    "client_id", "T", "mode", "horizon", "horizon_days", "window_end", "followup_end", "is_censored", "target",
)

# Метаданные lifecycle: аудит метки и стратификация, не признаки.
META_A: tuple[str, ...] = (
    "current_stage",
    "stage_since",
    "deterioration",
    "transition_type",
    "first_at_risk_at",
    "days_to_at_risk",
    "at_risk_reason",
    "at_risk_previous_stage",
    "first_churn_at",
    "days_to_churn",
    "churn_previous_stage",
)
META_B: tuple[str, ...] = (
    "current_stage",
    "previous_active_stage",
    "at_risk_since",
    "at_risk_reason",
    "outcome",
    "outcome_stage",
    "outcome_at",
    "days_to_outcome",
    "time_to_event",
    "time_to_recovery",
    "time_to_churn",
)
META: frozenset[str] = frozenset(ROW_KEYS) | frozenset(META_A) | frozenset(META_B)

# Поля payload, которые нужны стадиям: остальное в истории не читается.
LIFECYCLE_FIELDS = (
    "type", "reason", "channel", "direction", "counterparty", "migration_reason", "change_source",
    "operation", "status", "product_id", "contract_id",
)
LIFECYCLE_PAYLOAD = pa.schema([PAYLOAD.field(name) for name in LIFECYCLE_FIELDS])

DAY = pd.Timedelta(days=1)


def _utc(moment: datetime) -> pd.Timestamp:
    return pd.Timestamp(moment).tz_convert("UTC")


@dataclass
class GroupHistory:
    """
    История стадий группы до конца наблюдения и то, что из неё выпало.
    """

    group: str
    cutoff: pd.Timestamp
    followup_end: pd.Timestamp
    history: pd.DataFrame
    clients: int
    registered: int
    diverged: set[str] = field(default_factory=set)


def group_history(group: str, raw_dir: Path = RAW_DIR, future_dir: Path = FUTURE_DIR) -> GroupHistory:
    """
    Стадии клиентов группы до конца наблюдения. У группы с меткой из
    продолжения (train) события после T — только в продолжении: его
    хвост приклеивается к выгрузке тех же клиентов, а разошедшиеся с
    выгрузкой клиенты исключаются, как в churn_active90.
    """
    moment = _utc(cutoff(group, raw_dir))
    profile = read_profile(raw_dir / group / "profile.parquet")
    registered = lc.registrations(profile)

    tail: dict[str, pd.DataFrame] = {}
    diverged: set[str] = set()
    if group in FUTURE_LABEL_GROUPS:
        record = read_future(group, raw_dir, future_dir)
        followup = _utc(datetime.fromisoformat(record["period_end"]))
        diverged = set(record.get("diverged_clients", []))
        for block in client_blocks(future_dir / group / FUTURE_EVENTS, LIFECYCLE_PAYLOAD):
            tail.update(dict(tuple(block.groupby("client_id", sort=False))))
    else:
        followup = _utc(period_end(group, raw_dir))

    registered = {client: when for client, when in registered.items() if client not in diverged}

    parts: list[pd.DataFrame] = []
    seen: set[str] = set()
    empty: pd.DataFrame | None = None
    for block in client_blocks(raw_dir / group / "events.parquet", LIFECYCLE_PAYLOAD):
        members = set(block["client_id"])
        seen |= members
        empty = block.iloc[0:0]
        events = _with_tail(block, [tail[client] for client in members if client in tail])
        parts.append(lc.history(events, {c: registered[c] for c in members if c in registered}, followup))

    # Зарегистрированные в приложении без событий в выгрузке: стадии от
    # регистрации и, у train, по хвосту продолжения.
    silent = {client: when for client, when in registered.items() if client not in seen}
    if silent and empty is not None:
        events = _with_tail(empty, [tail[client] for client in silent if client in tail])
        parts.append(lc.history(events, silent, followup))

    history = (
        pd.concat(parts, ignore_index=True).sort_values(["client_id", "started_at"], kind="stable", ignore_index=True)
        if parts
        else pd.DataFrame(columns=list(lc.COLUMNS))
    )
    return GroupHistory(
        group=group,
        cutoff=moment,
        followup_end=followup,
        history=history,
        clients=len(profile),
        registered=len(registered),
        diverged=diverged,
    )


def _with_tail(block: pd.DataFrame, tails: list[pd.DataFrame]) -> pd.DataFrame:
    if not tails:
        return block
    return pd.concat([block, *tails], ignore_index=True).sort_values(
        ["client_id", "t"], kind="stable", ignore_index=True
    )


def _after(history: pd.DataFrame, moment: pd.Timestamp, clients: pd.Index) -> pd.DataFrame:
    """
    Переходы клиентов clients строго после moment, по времени.
    """
    later = history[(history["started_at"] > moment) & history["client_id"].isin(clients)]
    return later.sort_values(["client_id", "started_at"], kind="stable")


def window_end(cutoff: datetime, horizon: int | str) -> pd.Timestamp:
    """
    Конец окна метки: T + horizon дней или, при NEXT_MONTH, местная
    полночь после первого полного календарного месяца, который
    начинается не раньше T. У T на 1-е число это конец месяца самого T.
    """
    moment = _utc(cutoff)
    if horizon != NEXT_MONTH:
        return moment + pd.Timedelta(days=int(horizon))
    local = moment.tz_localize(None) + LOCAL_OFFSET
    start = local.normalize().replace(day=1)
    if local != start:
        start = start + pd.DateOffset(months=1)
    end = start + pd.DateOffset(months=1)
    return (end - LOCAL_OFFSET).tz_localize("UTC")


def _first(later: pd.DataFrame, stage: str) -> pd.DataFrame:
    return later[later["stage"] == stage].groupby("client_id").first()


def future_at_risk(
    history: pd.DataFrame, cutoff: datetime, followup_end: datetime, horizon: int | str, mode: str
) -> pd.DataFrame:
    """
    Строки active_to_at_risk на T в режиме mode: одна на клиента в
    активной стадии. target — 1 или 0 по режиму; NaN у цензурированной
    строки (окно не покрыто наблюдением).
    """
    if mode not in MODES_A:
        raise ValueError(f"режим {mode!r} не из {MODES_A}")

    moment, followup = _utc(cutoff), _utc(followup_end)
    end = window_end(moment, horizon)

    at = lc.stage_at(history, moment)
    population = at[at["stage"].isin(ACTIVE)]
    later = _after(history, moment, population.index)
    risk = _first(later, lc.AT_RISK).reindex(population.index)
    churn = _first(later, lc.CHURN).reindex(population.index)

    risk_at, churn_at = risk["started_at"], churn["started_at"]
    stage = population["stage"]

    # Первое ухудшение: At Risk или Churn, что раньше.
    risk_first = risk_at.notna() & (churn_at.isna() | (risk_at <= churn_at))
    worse = pd.Series(np.where(risk_first, lc.AT_RISK, np.where(churn_at.notna(), lc.CHURN, NONE)), index=stage.index)
    worse_at = risk_at.where(risk_first, churn_at)
    in_window = worse_at.notna() & (worse_at <= end)
    deterioration = worse.where(in_window, NONE)

    risk_in = risk_at.notna() & (risk_at <= end)
    churn_in = churn_at.notna() & (churn_at <= end)
    if mode == A1:
        positive = risk_in
    elif mode == A2:
        positive = risk_in | churn_in
    else:
        wanted = stage.map(A3_OUTCOMES)
        positive = (risk_in & wanted.map(lambda items: lc.AT_RISK in items)) | (
            churn_in & wanted.map(lambda items: lc.CHURN in items)
        )

    rows = pd.DataFrame(index=population.index)
    rows["T"] = moment
    rows["mode"] = mode
    rows["horizon"] = str(horizon)
    rows["horizon_days"] = (end - moment) / DAY
    rows["window_end"] = end
    rows["followup_end"] = followup
    rows["is_censored"] = followup < end
    rows["target"] = np.where(rows["is_censored"], np.nan, positive.astype(float))
    rows["current_stage"] = stage
    rows["stage_since"] = population["stage_since"]
    rows["deterioration"] = deterioration
    rows["transition_type"] = np.where(in_window, stage + "→" + worse, NONE)
    rows["first_at_risk_at"] = risk_at
    rows["days_to_at_risk"] = (risk_at - moment) / DAY
    rows["at_risk_reason"] = risk["reason"]
    rows["at_risk_previous_stage"] = risk["previous_stage"]
    rows["first_churn_at"] = churn_at
    rows["days_to_churn"] = (churn_at - moment) / DAY
    rows["churn_previous_stage"] = churn["previous_stage"]
    return rows.reset_index()[list(ROW_KEYS + META_A)]


def at_risk_outcome(
    history: pd.DataFrame, cutoff: datetime, followup_end: datetime, mode: str, horizon_days: int | None = None
) -> pd.DataFrame:
    """
    Строки at_risk_outcome на T в режиме mode: одна на клиента в At Risk.
    Исход — первый переход после T, не позже предела: конца наблюдения
    или T + H. B2 требует H, B3 — без H (до конца наблюдения).
    """
    if mode not in MODES_B:
        raise ValueError(f"режим {mode!r} не из {MODES_B}")
    if mode == B2 and horizon_days is None:
        raise ValueError(f"{B2}: нужен горизонт H")
    if mode == B3 and horizon_days is not None:
        raise ValueError(f"{B3}: время до исхода — до конца наблюдения, без горизонта")

    moment, followup = _utc(cutoff), _utc(followup_end)
    end = followup if horizon_days is None else moment + pd.Timedelta(days=horizon_days)
    limit = min(end, followup)

    at = lc.stage_at(history, moment)
    population = at[at["stage"] == lc.AT_RISK]
    first = _after(history, moment, population.index).groupby("client_id").first().reindex(population.index)

    exit_at, exit_stage = first["started_at"], first["stage"]
    resolved = exit_at.notna() & (exit_at <= limit)
    observed = end <= followup
    outcome = pd.Series(
        np.where(resolved, np.where(exit_stage == lc.CHURN, CHURN, RECOVERY), STILL_AT_RISK if observed else CENSORED),
        index=population.index,
    )
    # Без горизонта предел — конец наблюдения: не вышедший к нему клиент
    # цензурирован, «всё ещё At Risk» здесь не исход.
    if horizon_days is None:
        outcome = outcome.replace(STILL_AT_RISK, CENSORED)

    rows = pd.DataFrame(index=population.index)
    rows["T"] = moment
    rows["mode"] = mode
    rows["horizon"] = "first_outcome" if horizon_days is None else str(horizon_days)
    rows["horizon_days"] = (end - moment) / DAY
    rows["window_end"] = end
    rows["followup_end"] = followup
    rows["is_censored"] = outcome == CENSORED
    if mode == B1:
        rows["target"] = outcome.map({CHURN: 1.0, RECOVERY: 0.0})
    elif mode == B2:
        rows["target"] = outcome.map({name: float(code) for name, code in CLASSES_B.items()})
    else:
        rows["target"] = np.nan
    rows["current_stage"] = population["stage"]
    rows["previous_active_stage"] = population["previous_stage"]
    rows["at_risk_since"] = population["stage_since"]
    rows["at_risk_reason"] = population["reason"]
    rows["outcome"] = outcome
    rows["outcome_stage"] = exit_stage.where(resolved)
    rows["outcome_at"] = exit_at.where(resolved)
    rows["days_to_outcome"] = (rows["outcome_at"] - moment) / DAY
    # Время до первого исхода или до конца наблюдения (B3).
    rows["time_to_event"] = rows["days_to_outcome"].fillna((limit - moment) / DAY)
    rows["time_to_recovery"] = rows["days_to_outcome"].where(outcome == RECOVERY)
    rows["time_to_churn"] = rows["days_to_outcome"].where(outcome == CHURN)
    return rows.reset_index()[list(ROW_KEYS + META_B)]


def task_matrix(rows: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    X, y и метаданные строк задачи с известной меткой. X — признаки на T
    (features.compute, индекс client_id) и только они: колонки задачи и
    lifecycle в него не попадают.
    """
    leaked = META & set(features.columns)
    if leaked:
        raise ValueError(f"в признаках колонки задачи или lifecycle: {sorted(leaked)}")

    usable = rows[~rows["is_censored"] & rows["target"].notna()]
    missing = set(usable["client_id"]) - set(features.index)
    if missing:
        raise ValueError(f"у {len(missing)} строк задачи нет признаков на T")

    meta = usable.set_index("client_id")
    X = features.loc[meta.index]
    y = meta["target"].astype(int)
    return X, y, meta.drop(columns=["target"])


# ============================================================
# ДИАГНОСТИКА
# ============================================================


def _local(moment: pd.Timestamp):
    return (moment + LOCAL_OFFSET).date()


def _median(values: pd.Series) -> str:
    values = values.dropna()
    if values.empty:
        return "—"
    low, mid, high = values.quantile([0.25, 0.5, 0.75]).round(1).tolist()
    return f"{mid} ({low}–{high})"


def _label(horizon: int | str, state: GroupHistory) -> str:
    if horizon == NEXT_MONTH:
        return f"мес ({(window_end(state.cutoff, horizon) - state.cutoff) / DAY:.0f}д)"
    return str(horizon)


def describe_a(state: GroupHistory, horizons: list[int | str]) -> None:

    at = lc.stage_at(state.history, state.cutoff)
    print(f"\n# {TASK_A}: T {_local(state.cutoff)}, наблюдение до {_local(state.followup_end)}")
    print(f"  клиентов {state.clients}; без приложения к T {state.clients - len(at) - len(state.diverged)}; "
          f"разошлись в продолжении {len(state.diverged)}; уже At Risk {int((at['stage'] == lc.AT_RISK).sum())}; "
          f"уже Churn {int((at['stage'] == lc.CHURN).sum())}")

    rows = {
        (mode, horizon): future_at_risk(state.history, state.cutoff, state.followup_end, horizon, mode)
        for mode in MODES_A
        for horizon in horizons
    }

    print("\n## Сводка: режим × горизонт")
    summary = []
    for (mode, horizon), table in rows.items():
        eligible = table[~table["is_censored"]]
        summary.append({
            "режим": mode, "H": _label(horizon, state), "строк": len(eligible),
            "полож.": int(eligible["target"].sum()),
            "доля": round(float(eligible["target"].mean()), 4) if len(eligible) else np.nan,
            "цензур.": int(table["is_censored"].sum()),
        })
    print(pd.DataFrame(summary).to_string(index=False))

    # Тип перехода от режима не зависит: первое ухудшение в окне.
    print("\n## Тип перехода в окне (первое ухудшение), по горизонтам; цензурированные не считаются")
    types = pd.DataFrame({
        _label(horizon, state): rows[(A1, horizon)].loc[lambda t: ~t["is_censored"], "transition_type"].value_counts()
        for horizon in horizons
    }).fillna(0).astype(int)
    order = [f"{stage}→{worse}" for stage in lc.ACTIVE for worse in (lc.AT_RISK, lc.CHURN)] + [NONE]
    print(types.reindex([name for name in order if name in types.index]).to_string())

    for mode in MODES_A:
        print(f"\n## {mode}: положительные / строк по стадии на T")
        table = {}
        for horizon in horizons:
            eligible = rows[(mode, horizon)].loc[lambda t: ~t["is_censored"]]
            grouped = eligible.groupby("current_stage")["target"]
            table[_label(horizon, state)] = (grouped.sum().astype(int).astype(str) + " / " + grouped.size().astype(str))
        print(pd.DataFrame(table).reindex([stage for stage in lc.ACTIVE if stage in at["stage"].values]).to_string())


def describe_direct_churn(state: GroupHistory) -> None:
    """
    Клиенты в активной стадии на T, чьё первое ухудшение до конца
    наблюдения — Churn без At Risk. Почему правило At Risk не сработало.
    """
    rows = future_at_risk(state.history, state.cutoff, state.followup_end, int((state.followup_end - state.cutoff) / DAY), A2)
    direct = rows[rows["deterioration"] == lc.CHURN]
    print(f"\n# Уход в Churn, минуя At Risk, до конца наблюдения ({len(direct)} из {len(rows)} активных на T)")
    if direct.empty:
        return
    print("  стадия на T:", direct["current_stage"].value_counts().to_dict())
    print("  стадия прямо перед Churn:", direct["churn_previous_stage"].value_counts().to_dict())
    print(f"  дней от T до Churn: {_median(direct['days_to_churn'])}")
    # Churn — 60 суток без действия: на T клиент уже молчал около
    # 60 − (дней до Churn).
    silent = lc.RULES.churn_days - direct["days_to_churn"]
    print(f"  дней тишины уже на T (≈ 60 − дней до Churn): {_median(silent)}")
    for stage in (lc.ACTIVATED, lc.NEW, lc.GROWING, lc.CORE, lc.LOYAL):
        part = direct[direct["current_stage"] == stage]
        if part.empty:
            continue
        in_stage = (part["T"] - part["stage_since"]) / DAY
        print(f"  {stage}: {len(part)}; дней в стадии на T {_median(in_stage)}")
    activated = direct[direct["current_stage"] == lc.ACTIVATED]
    if len(activated):
        late = ((activated["T"] - activated["stage_since"]) / DAY > lc.RULES.activated_days).sum()
        print(f"  Activated дольше {lc.RULES.activated_days} дней на T: {int(late)} из {len(activated)} — "
              f"правило «14 дней от регистрации без действия» для них уже не действует, "
              f"а другого правила At Risk у Activated в Excel нет")


def describe_b(state: GroupHistory, horizons: list[int]) -> None:

    print(f"\n# {TASK_B}: At Risk на T")
    base = {None: at_risk_outcome(state.history, state.cutoff, state.followup_end, B3)}
    base.update({horizon: at_risk_outcome(state.history, state.cutoff, state.followup_end, B2, horizon) for horizon in horizons})

    print("\n## Исходы по горизонту")
    summary = []
    for horizon, table in base.items():
        counts = table["outcome"].value_counts()
        summary.append({
            "горизонт": "до конца набл." if horizon is None else f"H = {horizon}",
            "At Risk": len(table),
            RECOVERY: int(counts.get(RECOVERY, 0)),
            CHURN: int(counts.get(CHURN, 0)),
            STILL_AT_RISK: int(counts.get(STILL_AT_RISK, 0)),
            CENSORED: int(counts.get(CENSORED, 0)),
            "дней до recovery": _median(table.loc[table["outcome"] == RECOVERY, "days_to_outcome"]),
            "дней до churn": _median(table.loc[table["outcome"] == CHURN, "days_to_outcome"]),
        })
    print(pd.DataFrame(summary).to_string(index=False))

    print("\n## Режимы")
    for horizon in horizons:
        b1 = at_risk_outcome(state.history, state.cutoff, state.followup_end, B1, horizon)
        known = b1["target"].dropna()
        b2 = base[horizon]["outcome"].value_counts()
        print(f"  H = {horizon}: {B1} — {len(known)} строк, доля churn {known.mean():.3f}; "
              f"{B2} — " + ", ".join(f"{name} {int(b2.get(name, 0))}" for name in CLASSES_B))
    b3 = base[None]
    print(f"  {B3}: событий {int((~b3['is_censored']).sum())} (recovery {int((b3['outcome'] == RECOVERY).sum())}, "
          f"churn {int((b3['outcome'] == CHURN).sum())}), цензурировано {int(b3['is_censored'].sum())} "
          f"на {b3['horizon_days'].iloc[0]:.0f}-й день")

    print("\n## H = 60 по прежней стадии и причине At Risk" if 60 in horizons else "")
    if 60 in horizons:
        table = base[60]
        for column in ("previous_active_stage", "at_risk_reason"):
            print(pd.crosstab(table[column], table["outcome"]).to_string())


def _horizon(text: str) -> int | str:
    return text if text == NEXT_MONTH else int(text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Диагностика задач на стадиях CAPP; ничего не пишет")
    parser.add_argument("group", choices=GROUPS)
    parser.add_argument("--a-horizons", type=_horizon, nargs="+", default=[7, 14, 30, 31, 35, NEXT_MONTH],
                        help=f"дни или {NEXT_MONTH}")
    parser.add_argument("--b-horizons", type=int, nargs="+", default=[30, 45, 60])
    parser.add_argument("--final-test", action="store_true", help="test смотрится только в финальной оценке")
    args = parser.parse_args(argv)

    if args.group == FINAL_GROUP and not args.final_test:
        parser.error("test — только с --final-test")

    state = group_history(args.group)
    describe_a(state, args.a_horizons)
    describe_direct_churn(state)
    describe_b(state, args.b_horizons)


if __name__ == "__main__":
    main()
