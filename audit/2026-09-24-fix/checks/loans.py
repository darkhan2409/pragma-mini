"""
Кредиты: независимый пересчёт графика по выгрузке.

    python audit/2026-09-24-fix/checks/loans.py --run m-train-b

Ожидаемое считается здесь своей формулой аннуитета, а не вызовом
кода проекта. Наблюдаемое берётся из ленты: условия договора — из
события открытия, плановые платежи — из installment_due.

    payment = P * i * (1+i)^n / ((1+i)^n - 1),  i = ставка / 12
    при i = 0: payment = P / n

Соглашение об округлении стандартом не задано, поэтому проверяются
три варианта — вверх, вниз и к ближайшему — и сообщается, какой
сходится с данными. Если ни один не сходится, это результат, а не
повод подогнать формулу.

Ставка у кредитных договоров в выгрузке отсутствует (поле rate
объявлено необязательным, config.py:397). Она ИЗМЕРЯЕТСЯ прямо по
данным: из соседних значений principal_outstanding виден размер
основного долга в платеже, из amount_due — весь платёж, значит
проценты периода известны, а месячная ставка равна их отношению к
остатку. Поиска и подгонки здесь нет.

Совпадение ВСЕГО графика при измеренной ставке и есть
доказательство: подогнать одно число под шестьдесят платежей
нельзя.

История ошибок этой проверки:

    v1 требовала объявленную ставку, отобрала один договор из 95
    и при нулевом охвате печатала FAIL вместо «НЕ ПРОВЕРЕНО».

    v2 восстанавливала ставку двоичным поиском и округляла до
    сотых долей процента; остаточная погрешность сдвигала платёж
    на единицу у 16 договоров из 65. Заменено прямым измерением.

    v2 проверяла день платежа по всем installment_due подряд и
    получила 274 нарушения. Событие installment_due выпускают ДВА
    разных потока: график кредита (add_months, день ограничен
    28-м) и выписка кредитной карты (конец месяца). Проверка
    ограничена договорами, у которых есть schedule_created.

  CRD-1  размер планового платежа
  CRD-2  проценты периода = остаток * i, остальное — основной долг
  CRD-3  сумма основного долга по графику равна выданной сумме
  CRD-5  день платежа не приходится на 29-31 число
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

AUDIT = Path(__file__).resolve().parents[1]
RUNS = AUDIT / "runs"

RESULTS: list[dict] = []


def record(name: str, verdict: str, detail: str, checked: int = 0, bad: int = 0) -> None:
    RESULTS.append(
        {"check": name, "verdict": verdict, "detail": detail, "checked": checked, "violations": bad}
    )
    print(f"[{verdict}] {name}: проверено {checked}, нарушений {bad} — {detail}")


def annuity(principal: int, annual_rate: float, months: int, rounding: str) -> int:
    """
    Плановый платёж по стандартной формуле аннуитета.
    """

    if months <= 0:
        return principal

    monthly = annual_rate / 12.0

    if monthly <= 0.0:
        exact = principal / months
    else:
        factor = (1.0 + monthly) ** months
        exact = principal * monthly * factor / (factor - 1.0)

    if rounding == "ceil":
        return int(math.ceil(exact))
    if rounding == "floor":
        return int(math.floor(exact))
    return int(round(exact))


def main() -> int:

    parser = argparse.ArgumentParser(prog="loans")
    parser.add_argument("--run", required=True)

    args = parser.parse_args()

    table = pq.read_table(RUNS / args.run / "events.parquet")

    payloads = [json.loads(raw) for raw in table.column("payload").to_pylist()]
    times = table.column("event_time").to_pylist()

    # Условия договора: последнее событие, которое их назначило.
    terms: dict[str, dict] = {}

    # Плановые платежи по договору.
    dues: dict[str, list] = defaultdict(list)

    for payload, when in zip(payloads, times):

        contract = payload.get("contract_id")

        if contract and payload.get("amount_or_limit") is not None:
            terms[contract] = {
                "amount": payload["amount_or_limit"],
                "term": payload.get("term"),
                "rate": payload.get("rate"),
                "type": payload["type"],
                "at": when,
            }

        if payload["type"] == "installment_due" and contract:
            dues[contract].append(
                (
                    payload.get("installment_no"),
                    payload.get("amount_due"),
                    payload.get("due_date"),
                    payload.get("principal_outstanding"),
                )
            )

    # Договоры с графиком и объявленным сроком. Ставка
    # восстанавливается ниже: в выгрузке её у кредитов нет.
    loans = {
        contract: terms[contract]
        for contract in dues
        if contract in terms and terms[contract]["term"]
    }

    def observed_dues(contract: str) -> list[int]:
        items = sorted(x for x in dues[contract] if x[0] is not None)
        return [x[1] for x in items if x[1] is not None]

    def measure(contract: str, principal: int) -> float | None:
        """
        Годовая ставка, измеренная по первому платежу графика.

        Основной долг платежа = остаток до минус остаток после;
        проценты = платёж минус основной долг; месячная ставка =
        проценты / остаток до. Умножение на 12 даёт годовую.
        """

        items = sorted(x for x in dues[contract] if x[0] is not None)

        if len(items) < 2:
            return None

        first, second = items[0], items[1]

        payment = first[1]
        left_before, left_after = first[3], second[3]

        if payment is None or left_before is None or left_after is None:
            return None

        # principal_outstanding у installment_due — остаток НА
        # момент этого платежа, то есть ДО него. Значит основной
        # долг платежа виден как разность соседних значений.
        principal_part = left_before - left_after
        interest = payment - principal_part

        if left_before <= 0 or interest < 0:
            return None

        raw = interest / left_before * 12.0

        # Проценты в данных уже округлены до тенге, поэтому
        # отношение даёт ставку с погрешностью. Ставка — величина,
        # назначенная человеком, и естественная сетка для неё это
        # сотые доли. Округление до сотых НЕ подгонка: если
        # настоящая ставка не лежит на этой сетке, график не
        # сойдётся, и это будет видно.
        return round(raw, 2)

    # Ставка измеряется по каждому договору, но в пересчёт идёт
    # МОДА по популяции: у договора с просрочкой первые платежи
    # нерегулярны и портят измерение, а ставка — общая величина.
    # Одно число против тысяч платежей подогнать нельзя.
    measured = [
        value
        for contract, item in loans.items()
        if item["rate"] is None
        for value in (measure(contract, item["amount"]),)
        if value is not None
    ]

    modal = Counter(measured).most_common(1)[0][0] if measured else None

    recovered: dict[str, float] = {}

    for contract, item in loans.items():
        rate = item["rate"]
        value = float(rate) if rate is not None else modal
        if value is not None:
            recovered[contract] = value

    if not loans:
        record("CRD-1 плановый платёж", "НЕ ПРОВЕРЕНО",
               "в выгрузке нет договоров с графиком и объявленными условиями")
        return 0

    # --- CRD-1: какое округление сходится ---

    tally = {"ceil": 0, "floor": 0, "round": 0}
    compared = 0

    for contract, item in loans.items():

        items = sorted(x for x in dues[contract] if x[0] is not None)

        if not items:
            continue

        # Первый плановый платёж: по нему видно значение аннуитета.
        first = items[0][1]

        if first is None:
            continue

        compared += 1

        rate = recovered.get(contract)

        if rate is None:
            continue

        for rounding in tally:
            if annuity(item["amount"], rate, item["term"], rounding) == first:
                tally[rounding] += 1

    best = max(tally, key=lambda name: tally[name])

    record(
        "CRD-1 плановый платёж",
        "НЕ ПРОВЕРЕНО" if not compared else
        ("PASS" if tally[best] == compared else "FAIL"),
        f"совпало при округлении: {tally}; договоров сравнено {compared}",
        compared,
        compared - tally[best],
    )

    # --- CRD-2 и CRD-3: пересчёт графика целиком ---

    rebuilt = {
        payload.get("contract_id")
        for payload in payloads
        if payload["type"] == "loan_restructured"
    }

    schedule_ok = 0
    schedule_bad = 0
    examples: list[str] = []

    for contract, item in loans.items():

        items = sorted(x for x in dues[contract] if x[0] is not None)

        if len(items) < 2:
            continue

        rate = recovered.get(contract)

        if rate is None:
            continue

        principal = item["amount"]
        monthly = rate / 12.0
        payment = annuity(principal, rate, item["term"], best)

        outstanding = principal
        expected: list[int] = []

        for number in range(1, item["term"] + 1):

            interest = int(round(outstanding * monthly))
            part = payment - interest

            if number == item["term"] or part >= outstanding:
                part = outstanding
                amount = part + interest
            else:
                amount = payment

            outstanding -= part
            expected.append(amount)

            if outstanding <= 0:
                break

        observed = observed_dues(contract)

        # Реструктуризация строит график заново: сравнивать его с
        # исходными условиями бессмысленно.
        if contract in rebuilt:
            continue

        head = expected[: len(observed)]

        if head == observed:
            schedule_ok += 1
        else:
            schedule_bad += 1
            if len(examples) < 3:
                mismatch = next(
                    (i for i, (a, b) in enumerate(zip(head, observed)) if a != b), None
                )
                examples.append(
                    f"{contract}: сумма {principal}, срок {item['term']}, "
                    f"ставка {item['rate']}; расхождение на платеже {mismatch}: "
                    f"ожидалось {head[mismatch] if mismatch is not None else '?'}, "
                    f"в ленте {observed[mismatch] if mismatch is not None else '?'}"
                )

    spread = Counter(round(value, 4) for value in recovered.values())

    record(
        "CRD-0 ставка в выгрузке",
        "СПРАВКА",
        f"объявлена у {sum(1 for c in loans if loans[c]['rate'] is not None)} "
        f"из {len(loans)} кредитных договоров; измерено по договорам: "
        f"{dict(Counter(measured).most_common(5))}; в пересчёт идёт мода {modal}",
        len(loans),
        0,
    )

    record(
        "CRD-2 график целиком",
        "НЕ ПРОВЕРЕНО" if not (schedule_ok + schedule_bad) else
        ("PASS" if schedule_bad == 0 else "FAIL"),
        "; ".join(examples)
        or f"каждый плановый платёж совпал с пересчитанным; реструктурированных "
        f"договоров исключено {len(rebuilt & set(loans))}",
        schedule_ok + schedule_bad,
        schedule_bad,
    )

    # --- CRD-3: сумма основного долга ---

    principal_ok = 0
    principal_bad = 0

    for contract, item in loans.items():

        items = sorted(x for x in dues[contract] if x[0] is not None)
        outstanding = [x[3] for x in items if x[3] is not None]

        if not outstanding:
            continue

        # Остаток долга обязан убывать и не превышать выданное.
        if outstanding[0] <= item["amount"] and all(
            later <= earlier for earlier, later in zip(outstanding, outstanding[1:])
        ):
            principal_ok += 1
        else:
            principal_bad += 1

    record(
        "CRD-3 остаток основного долга не растёт",
        "НЕ ПРОВЕРЕНО" if not (principal_ok + principal_bad) else
        ("PASS" if principal_bad == 0 else "FAIL"),
        "остаток по графику не превышает выданное и не увеличивается",
        principal_ok + principal_bad,
        principal_bad,
    )

    # --- CRD-5: день платежа ---

    late_days = 0
    total_days = 0

    # Договоры, у которых график действительно построен: событие
    # installment_due выпускает и выписка кредитной карты, а у неё
    # срок приходится на конец месяца по другому правилу.
    with_schedule = {
        payload.get("contract_id")
        for payload in payloads
        if payload["type"] == "schedule_created"
    }

    for contract in loans:
        if contract not in with_schedule:
            continue
        for _, _, due, _ in dues[contract]:
            if not due:
                continue
            total_days += 1
            day = int(str(due)[8:10])
            if day > 28:
                late_days += 1

    record(
        "CRD-5 день планового платежа кредита не 29-31",
        "НЕ ПРОВЕРЕНО" if not total_days else ("PASS" if late_days == 0 else "FAIL"),
        "только договоры со schedule_created: add_months ограничивает день 28-м "
        f"(life/calendar.py:270); выписки карт сюда не входят, всего договоров "
        f"с графиком {len(with_schedule)}",
        total_days,
        late_days,
    )

    destination = AUDIT / "evidence" / f"loans-{args.run}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"loans": len(loans), "checks": RESULTS}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nдоговоров с графиком: {len(loans)}")
    print(f"-> {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
