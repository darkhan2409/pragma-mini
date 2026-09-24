"""
Деньги: внутренний журнал против выгруженной ленты.

    python audit/2026-09-24-fix/checks/money.py --name o1-observed --clients 64 \
        --seed 100 --start 2024-01-01 --end 2026-01-01 --compare p0-pilot

Внутреннее состояние симуляции наружу не выгружается, поэтому
harness перехватывает engine._finish: он ищется как глобал модуля
(engine.py:127), и подмена НЕ меняет ни решений, ни порядка, ни
случайности — обёртка только запоминает объект и зовёт настоящую
функцию.

Контроль невмешательства обязателен: RAW наблюдаемого прогона
сверяется побайтово с обычным (--compare). Без совпадения
результаты наблюдения не принимаются.

Что проверяется независимо от кода проекта:

  LEDGER-A  остаток счёта = начальный + приход − расход по журналу
  LEDGER-B  сумма каждой проводки целая и строго положительная
  LEDGER-C  каждая сторона — либо счёт этого клиента, либо счёт
            ДРУГОГО клиента сообщества, либо внешняя сторона
            объявленного вида
  P2P       перевод между клиентами отражён у обеих сторон: на
            проводку в журнале отправителя есть такая же в
            журнале получателя

История ошибок этой проверки:

    v1 считала стороной только счёт своего клиента и внешнюю
    сторону, поэтому 1414 проводок p2p попали в «неизвестные
    стороны». Это ошибка проверки, а не генератора: счёт
    контрагента — законная четвёртая категория. Добавлена
    категория и сверка обеих сторон.
  TAPE-A    цепочка balance_after в ленте согласована с amount и
            direction (считается по самой ленте)
  TAPE-B    событие не в статусе approved остаток не двигает
  SOLVENCY  сколько одобренных списаний не прошли бы, если бы
            остаток считался строго причинно — только по
            проводкам с временем не позже самой операции
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[1]
RUNS = AUDIT / "runs"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUDIT / "harness"))

from sources import SourcesChanged, verify  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402

# Объявленное окно расхождения решения и даты: ledger.py:69.
RECENT_WINDOW_SECONDS = 2 * 24 * 3600

# Внешние стороны проводки: объявлены в ledger.py:14-34.
EXTERNAL_PREFIXES = ("merchant:", "employer:", "external:", "loan:")
EXTERNAL_EXACT = ("government", "bank_pnl")


def sha256(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def external(side: str) -> bool:
    return side in EXTERNAL_EXACT or side.startswith(EXTERNAL_PREFIXES)


# ============================================================
# ЖУРНАЛ
# ============================================================


def check_ledger(states: list) -> dict:

    problems: dict[str, list] = defaultdict(list)

    accounts = 0
    postings = 0
    unknown_sides: dict[str, int] = defaultdict(int)

    # Счёт -> кому принадлежит. Нужен, чтобы отличить счёт
    # контрагента от неизвестной стороны.
    owner: dict[str, str] = {}

    for state in states:
        for account_id in state.ledger.accounts:
            owner[account_id] = state.client_id

    # Проводки, у которых сторона принадлежит другому клиенту.
    crossing: dict[str, list] = defaultdict(list)

    for state in states:

        ledger = state.ledger

        moved: dict[str, int] = defaultdict(int)

        for posting in ledger.postings:

            postings += 1

            if not isinstance(posting.amount, int):
                problems["LEDGER-B тип суммы"].append(
                    f"{state.client_id} {posting.posting_id}: {type(posting.amount).__name__}"
                )

            if posting.amount <= 0:
                problems["LEDGER-B неположительная сумма"].append(
                    f"{state.client_id} {posting.posting_id}: {posting.amount}"
                )

            for side, sign in ((posting.debit, -1), (posting.credit, +1)):

                if side in ledger.accounts:
                    moved[side] += sign * posting.amount
                elif side in owner:
                    # Счёт другого клиента: проводка обязана быть
                    # отражена и в его журнале.
                    crossing[owner[side]].append(
                        (posting.ts, posting.debit, posting.credit, posting.amount)
                    )
                elif not external(side):
                    unknown_sides[side] += 1
                    problems["LEDGER-C неизвестная сторона"].append(
                        f"{state.client_id} {posting.posting_id}: {side}"
                    )

        for account_id, account in ledger.accounts.items():

            accounts += 1

            expected = int(account.opening_balance) + moved.get(account_id, 0)

            if expected != int(account.balance):
                problems["LEDGER-A остаток не сходится"].append(
                    f"{state.client_id} {account_id}: журнал даёт {expected}, "
                    f"счёт показывает {account.balance}"
                )

    # P2P: у каждой проводки, назвавшей чужой счёт, обязана быть
    # такая же проводка в журнале владельца этого счёта.
    #
    # Время в ключ НЕ входит. Отправитель и получатель проводят
    # перевод не одной секундой: зачисление выпускается позже
    # списания, и сверка по точному ts объявляла односторонними
    # все до одной проводки. Здесь пара ищется по сторонам и
    # сумме, а расхождение времени измеряется и печатается
    # отдельно — оно обязано укладываться в объявленное окно
    # расхождения (ledger.RECENT_WINDOW).
    own: dict[str, dict] = {}

    for state in states:
        bucket: dict = defaultdict(list)
        for posting in state.ledger.postings:
            bucket[(posting.debit, posting.credit, posting.amount)].append(posting.ts)
        own[state.client_id] = bucket

    crossing_total = 0
    mirrored = 0

    # Расхождение времени зеркальных ног, в секундах.
    deltas: Counter = Counter()

    for client_id, items in crossing.items():

        bucket = own.get(client_id, {})

        for ts, debit, credit, amount in items:

            crossing_total += 1

            candidates = bucket.get((debit, credit, amount), [])

            if not candidates:
                if len(problems["P2P односторонняя проводка"]) < 10:
                    problems["P2P односторонняя проводка"].append(
                        f"{client_id}: {ts} {debit} -> {credit} {amount}"
                    )
                continue

            mirrored += 1

            nearest = min(candidates, key=lambda other: abs((other - ts).total_seconds()))

            deltas[int((nearest - ts).total_seconds())] += 1

    window = max((abs(value) for value in deltas), default=0)

    if window > RECENT_WINDOW_SECONDS:
        problems["P2P зеркало позже окна расхождения"].append(
            f"наибольшее расхождение {window} с при обещанных {RECENT_WINDOW_SECONDS}"
        )

    # Закон сохранения целиком: сумма по ВСЕМ сторонам, включая
    # внешние, обязана быть нулём. Проводки межклиентских
    # переводов считаются один раз — по журналу отправителя.
    trial: dict[str, int] = defaultdict(int)
    counted: set = set()

    for state in states:
        for posting in state.ledger.postings:
            mark = (posting.ts, posting.debit, posting.credit, posting.amount)
            if mark in counted:
                continue
            counted.add(mark)
            trial[posting.debit] -= posting.amount
            trial[posting.credit] += posting.amount

    balance = sum(trial.values())

    if balance != 0:
        problems["LEDGER-D общий баланс не ноль"].append(str(balance))

    return {
        "clients": len(states),
        "accounts": accounts,
        "postings": postings,
        "crossing_postings": crossing_total,
        "crossing_mirrored": mirrored,
        "mirror_time_deltas_seconds": dict(sorted(deltas.items())),
        "trial_balance": balance,
        "trial_sides": len(trial),
        "unknown_sides": dict(unknown_sides),
        "problems": {name: items[:10] for name, items in problems.items()},
        "problem_counts": {name: len(items) for name, items in problems.items()},
        "verdict": "PASS" if not problems else "FAIL",
    }


# ============================================================
# ЛЕНТА
# ============================================================


def check_tape(run: str) -> dict:
    """
    Цепочка balance_after по каждому счёту — только по выгрузке.
    """

    table = pq.read_table(RUNS / run / "events.parquet")

    clients = table.column("client_id").to_pylist()
    payloads = [json.loads(raw) for raw in table.column("payload").to_pylist()]

    last: dict[tuple, int] = {}

    breaks: list[str] = []
    checked = 0

    moved_by_declined = 0
    declined = 0

    for client_id, payload in zip(clients, payloads):

        account = payload.get("account_id")
        balance = payload.get("balance_after")

        if account is None or balance is None:
            continue

        key = (client_id, account)

        status = payload.get("status")
        amount = payload.get("amount")
        direction = payload.get("direction")

        if status is not None and status != "approved":

            declined += 1

            if key in last and balance != last[key]:
                moved_by_declined += 1
                if len(breaks) < 10:
                    breaks.append(
                        f"TAPE-B {client_id} {account}: status={status}, "
                        f"остаток {last[key]} -> {balance}"
                    )

            last[key] = balance
            continue

        if key in last and isinstance(amount, int) and direction in ("debit", "credit"):

            checked += 1

            step = -amount if direction == "debit" else amount

            if balance != last[key] + step:
                if len(breaks) < 20:
                    breaks.append(
                        f"TAPE-A {client_id} {account}: {last[key]} {direction} "
                        f"{amount} -> {balance}, ожидалось {last[key] + step}"
                    )

        last[key] = balance

    return {
        "run": run,
        "steps_checked": checked,
        "declined_events": declined,
        "declined_moving_balance": moved_by_declined,
        "breaks": breaks,
        "break_count": len(breaks),
        "verdict": "PASS" if not breaks else "FAIL",
    }


# ============================================================
# ХРОНОЛОГИЧЕСКАЯ ПЛАТЁЖЕСПОСОБНОСТЬ
# ============================================================


def check_solvency(states: list) -> dict:
    """
    Строго причинный остаток: только проводки со временем не
    позже самой операции.

    Генератор сознательно смотрит вперёд на два дня
    (ledger.RECENT_WINDOW), поэтому расхождение здесь не ошибка
    само по себе — измеряется его РАЗМЕР и то, укладывается ли
    он в объявленные двое суток.
    """

    short = 0
    total = 0
    worst_gap = 0
    worst_lookahead = timedelta(0)
    examples: list[str] = []

    for state in states:

        ledger = state.ledger

        by_account: dict[str, list] = defaultdict(list)

        for posting in ledger.postings:
            for side, sign in ((posting.debit, -1), (posting.credit, +1)):
                if side in ledger.accounts:
                    by_account[side].append((posting.ts, sign * posting.amount, posting))

        for account_id, moves in by_account.items():

            account = ledger.accounts[account_id]
            moves.sort(key=lambda item: item[0])

            running = int(account.opening_balance)

            for ts, delta, posting in moves:

                if delta < 0:

                    total += 1

                    limit = account.credit_limit if account.kind == "credit_card" else 0

                    if running + limit < -delta:

                        short += 1
                        gap = -delta - (running + limit)
                        worst_gap = max(worst_gap, gap)

                        # Насколько вперёд надо было заглянуть,
                        # чтобы покрытие нашлось.
                        ahead = running + limit
                        found = None

                        for other_ts, other_delta, _ in moves:
                            if other_ts <= ts:
                                continue
                            ahead += other_delta
                            if ahead >= -delta:
                                found = other_ts - ts
                                break

                        if found is not None:
                            worst_lookahead = max(worst_lookahead, found)

                        if len(examples) < 10:
                            examples.append(
                                f"{state.client_id} {account_id} {ts.isoformat()}: "
                                f"остаток {running}, лимит {limit}, списание {-delta}, "
                                f"покрытие через {found}"
                            )

                running += delta

    return {
        "debits_checked": total,
        "short_at_moment": short,
        "share": round(short / total, 6) if total else 0.0,
        "worst_shortfall": worst_gap,
        "worst_lookahead_seconds": worst_lookahead.total_seconds(),
        "within_recent_window": worst_lookahead <= timedelta(days=2),
        "examples": examples,
    }


# ============================================================
# ПРОГОН С НАБЛЮДЕНИЕМ
# ============================================================


def observed_run(name: str, clients: int, seed: int, start: str, end: str) -> tuple:

    out = (RUNS / name).resolve()

    if RUNS.resolve() not in out.parents:
        raise SystemExit(f"отказ: {out} вне {RUNS}")

    from src.generator import emit, engine

    captured: list = []

    original = engine._finish

    def watching(sim):
        # Состояния запоминаются ДО вызова настоящей функции:
        # она же и правит payload балансами.
        captured.extend(sim.clients[ordinal] for ordinal in sorted(sim.clients))
        return original(sim)

    engine._finish = watching

    try:
        counts = emit.generate_dataset(
            total_clients=clients,
            out_dir=out,
            seed=seed,
            world_seed=42,
            history_start=datetime.fromisoformat(start),
            history_end=datetime.fromisoformat(end),
            workers=1,
            quiet=True,
        )
    finally:
        engine._finish = original

    return captured, counts


def main() -> int:

    parser = argparse.ArgumentParser(prog="money")
    parser.add_argument("--name", required=True)
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--compare", default=None, help="прогон для контроля невмешательства")

    args = parser.parse_args()

    try:
        code_state = verify("до прогона")
    except SourcesChanged as error:
        print(error)
        return 2

    states, counts = observed_run(
        args.name, args.clients, args.seed, args.start, args.end
    )

    try:
        if verify("после прогона") != code_state:
            print("состояние кода изменилось во время прогона")
            return 2
    except SourcesChanged as error:
        print(error)
        return 2

    control = None

    if args.compare:
        control = {
            name: sha256(RUNS / args.name / name) == sha256(RUNS / args.compare / name)
            for name in ("events.parquet", "profile.parquet")
        }
        control["verdict"] = "PASS" if all(
            value for key, value in control.items() if key != "verdict"
        ) else "FAIL"

    report = {
        "run": args.name,
        "code_state": code_state,
        "counts": counts,
        "observation_control": control,
        "ledger": check_ledger(states),
        "tape": check_tape(args.name),
        "solvency": check_solvency(states),
    }

    destination = AUDIT / "evidence" / f"money-{args.name}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print(f"контроль невмешательства: {control}")
    print(f"журнал: {report['ledger']['verdict']} — счетов "
          f"{report['ledger']['accounts']}, проводок {report['ledger']['postings']}, "
          f"проблем {report['ledger']['problem_counts']}")
    print(f"лента: {report['tape']['verdict']} — сверено шагов "
          f"{report['tape']['steps_checked']}, разрывов {report['tape']['break_count']}, "
          f"отклонённых {report['tape']['declined_events']}, из них двигали остаток "
          f"{report['tape']['declined_moving_balance']}")
    print(f"платёжеспособность: списаний {report['solvency']['debits_checked']}, "
          f"не хватило в момент {report['solvency']['short_at_moment']} "
          f"({report['solvency']['share']:.4%}), заглядывание вперёд до "
          f"{report['solvency']['worst_lookahead_seconds'] / 3600:.1f} ч, "
          f"в двое суток укладывается: {report['solvency']['within_recent_window']}")
    print(f"-> {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
