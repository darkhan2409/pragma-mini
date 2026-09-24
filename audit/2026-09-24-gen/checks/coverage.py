"""
Таблица покрытия прогона: что встретилось в выгрузке и сколько.

    python audit/2026-09-24-gen/checks/coverage.py --run p0-pilot

Проверка описательная: она не выносит вердиктов, а показывает, по
каким правилам есть материал, а где охват нулевой и проверка будет
«НЕ ПРОВЕРЕНО».

Каждый показатель сопровождается ОПРЕДЕЛЕНИЕМ подсчёта и пометкой
источника:

    raw       считается только по events.parquet / profile.parquet
    catalog   нужен справочник продуктов, из RAW не выводится
    absent    в RAW этого нет вовсе, подменять похожим нельзя

История ошибок этой проверки:

    v1 считала доли присутствия полей одним проходом по растущему
    множеству имён, поэтому поле, впервые встреченное поздно,
    получало заниженную долю. Исправлено вторым проходом: сначала
    собирается объединение имён по типу, потом считается наличие.

    v1 определяла «перевод между своими» как перевод без
    counterparty. Определение неверно: counterparty есть у 100 %
    переводов. Заменено на парность по transfer_id, и результат
    перестал быть нулевым.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

AUDIT = Path(__file__).resolve().parents[1]
RUNS = AUDIT / "runs"


# Сценарии, ради которых пилот и делается. Значение — определение
# подсчёта, по которому событие относят к сценарию.
SCENARIOS: dict[str, tuple[str, object]] = {
    "просрочка зарегистрирована": (
        "события type=delinquency_registered",
        lambda t, p: t == "delinquency_registered",
    ),
    "просрочка погашена": (
        "события type=arrears_cleared",
        lambda t, p: t == "arrears_cleared",
    ),
    "платёж пропущен": (
        "события type=installment_missed",
        lambda t, p: t == "installment_missed",
    ),
    "частичная оплата": (
        "type=installment_paid и 0 < amount_paid < amount_due",
        lambda t, p: t == "installment_paid"
        and isinstance(p.get("amount_paid"), int)
        and isinstance(p.get("amount_due"), int)
        and 0 < p["amount_paid"] < p["amount_due"],
    ),
    "досрочное погашение": ("события type=early_repayment", lambda t, p: t == "early_repayment"),
    "реструктуризация": ("события type=loan_restructured", lambda t, p: t == "loan_restructured"),
    "кредит закрыт": ("события type=loan_closed", lambda t, p: t == "loan_closed"),
    "график без выдачи": (
        "type=schedule_created; сравнивается с числом loan_disbursement",
        lambda t, p: t == "schedule_created",
    ),
    "возврат": ("события type=refund", lambda t, p: t == "refund"),
    "отмена операции": ("события type=reversal", lambda t, p: t == "reversal"),
    "chargeback": ("события type=chargeback", lambda t, p: t == "chargeback"),
    "карта заблокирована": ("события type=card_blocked", lambda t, p: t == "card_blocked"),
    "карта разблокирована": ("события type=card_unblocked", lambda t, p: t == "card_unblocked"),
    "карта перевыпущена": ("события type=card_reissued", lambda t, p: t == "card_reissued"),
    "продукт закрыт": ("события type=product_closed", lambda t, p: t == "product_closed"),
    "продукт мигрировал": ("события type=product_migrated", lambda t, p: t == "product_migrated"),
    "условия изменены": (
        "события type=contract_terms_changed",
        lambda t, p: t == "contract_terms_changed",
    ),
    "тариф пересмотрен": ("события type=product_repriced", lambda t, p: t == "product_repriced"),
    "вклад продлён": ("события type=product_renewed", lambda t, p: t == "product_renewed"),
    "проценты вклада": ("события type=interest_credit", lambda t, p: t == "interest_credit"),
    "пополнение вклада": ("события type=deposit_topup", lambda t, p: t == "deposit_topup"),
    "снятие со вклада": ("события type=deposit_withdrawal", lambda t, p: t == "deposit_withdrawal"),
    "отказ по заявке": (
        "type=application_decision и decision=rejected",
        lambda t, p: t == "application_decision" and p.get("decision") == "rejected",
    ),
    "одобрение по заявке": (
        "type=application_decision и decision=approved",
        lambda t, p: t == "application_decision" and p.get("decision") == "approved",
    ),
    "операция отклонена": (
        "любое событие со status=declined",
        lambda t, p: p.get("status") == "declined",
    ),
    "нехватка средств": (
        "любое событие с decline_reason=insufficient_funds",
        lambda t, p: p.get("decline_reason") == "insufficient_funds",
    ),
    "мошенничество: алерт": ("события type=fraud_alert", lambda t, p: t == "fraud_alert"),
    "мошенничество: решение": ("события type=fraud_decision", lambda t, p: t == "fraud_decision"),
    "обращение в поддержку": ("события type=case_opened", lambda t, p: t == "case_opened"),
    "смена анкеты": ("события type=profile_change", lambda t, p: t == "profile_change"),
    "кешбэк": ("события type=cashback_credit", lambda t, p: t == "cashback_credit"),
    "комиссия": ("события type=fee_charge", lambda t, p: t == "fee_charge"),
    "зарплата": ("события type=salary_credit", lambda t, p: t == "salary_credit"),
    "безымянная точка": (
        "type=purchase, merchant_category есть, merchant_name пуст или отсутствует",
        lambda t, p: t == "purchase" and p.get("merchant_category") and not p.get("merchant_name"),
    ),
}


# Чего в RAW нет и чем это НЕЛЬЗЯ подменять.
ABSENT = {
    "версия продукта": (
        "в каталоге полей payload нет ни product_version, ни tariff_version "
        "(src/generator/config.py:377-397); события contract_terms_changed и "
        "product_repriced отмечают факт смены, но не называют версию"
    ),
    "семейство продукта": (
        "RAW несёт только product_id; семейство живёт в каталоге "
        "(src/generator/world/products.py) и из выгрузки не выводится"
    ),
    "жизненное состояние клиента": (
        "состояние (active, dormant, churned и прочие) наружу не выгружается вовсе: "
        "оно только модулирует интенсивности (src/generator/life/lifecycle.py)"
    ),
    "причина пропуска поля": (
        "причина объявлена в параметрах (source_unavailable, not_collected, unknown), "
        "но разбирается и выбрасывается (src/generator/observe/defects.py:91)"
    ),
    "источник пропуска поля": (
        "по RAW нельзя отличить поле, снятое слоем наблюдения, от поля, которое "
        "симуляция не заполняла: и то и другое выглядит как отсутствие ключа"
    ),
}


def sha256(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def quantile(values: list[int], share: float) -> int:

    if not values:
        return 0

    index = min(len(values) - 1, int(share * (len(values) - 1) + 0.5))

    return values[index]


def main() -> int:

    parser = argparse.ArgumentParser(prog="coverage")
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default=None)

    args = parser.parse_args()

    out = RUNS / args.run

    events_path = out / "events.parquet"
    profile_path = out / "profile.parquet"

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    events = pq.read_table(events_path)
    profile = pq.read_table(profile_path)

    integrity = {
        "events_sha256_ok": sha256(events_path) == manifest["events_sha256"],
        "profile_sha256_ok": sha256(profile_path) == manifest["profile_sha256"],
        "events_rows_ok": events.num_rows == manifest["events_rows"],
        "profile_rows_ok": profile.num_rows == manifest["profile_rows"],
        "envelope_columns": events.column_names,
        "definition": "sha256 и число строк посчитаны здесь и сверены с manifest.json",
    }

    client_ids = events.column("client_id").to_pylist()
    times = events.column("event_time").to_pylist()
    sources = events.column("source").to_pylist()
    payloads = [json.loads(raw) for raw in events.column("payload").to_pylist()]

    kinds = [payload["type"] for payload in payloads]

    # --- первый проход: объединение имён полей по типу события ---

    fields_of: defaultdict = defaultdict(set)

    for kind, payload in zip(kinds, payloads):
        fields_of[kind].update(payload)

    # --- второй проход: всё остальное ---

    by_type: Counter = Counter()
    by_source: Counter = Counter()
    clients_of_type: defaultdict = defaultdict(set)
    per_client: Counter = Counter()

    scenarios: Counter = Counter()
    scenario_clients: defaultdict = defaultdict(set)

    products: Counter = Counter()
    product_clients: defaultdict = defaultdict(set)
    channels: Counter = Counter()
    statuses: Counter = Counter()
    decline_reasons: Counter = Counter()
    transfer_ids: Counter = Counter()

    present: defaultdict = defaultdict(Counter)

    offsets: Counter = Counter()
    with_millis = 0
    earliest = None
    latest = None

    for client_id, text, source, payload, kind in zip(
        client_ids, times, sources, payloads, kinds
    ):

        by_type[kind] += 1
        by_source[source] += 1
        clients_of_type[kind].add(client_id)
        per_client[client_id] += 1

        moment = datetime.fromisoformat(text)

        offsets[text[-6:]] += 1

        if "." in text:
            with_millis += 1

        earliest = moment if earliest is None else min(earliest, moment)
        latest = moment if latest is None else max(latest, moment)

        if payload.get("product_id"):
            products[payload["product_id"]] += 1
            product_clients[payload["product_id"]].add(client_id)

        if payload.get("channel"):
            channels[payload["channel"]] += 1

        if payload.get("status"):
            statuses[payload["status"]] += 1

        if payload.get("decline_reason"):
            decline_reasons[payload["decline_reason"]] += 1

        if payload.get("transfer_id"):
            transfer_ids[payload["transfer_id"]] += 1

        for name, (_, check) in SCENARIOS.items():
            if check(kind, payload):
                scenarios[name] += 1
                scenario_clients[name].add(client_id)

        for field in fields_of[kind]:
            if field in payload:
                present[kind][field] += 1

    lengths = sorted(per_client.values())

    legs = Counter(transfer_ids.values())

    report = {
        "run": args.run,
        "code_state": json.loads(
            (out / "run_record.json").read_text(encoding="utf-8")
        ).get("code_state"),
        "manifest": manifest,
        "integrity": integrity,
        "clients": {
            "definition": "profile_rows — строки profile.parquet; with_events — "
            "различных client_id в events.parquet; длины — событий на client_id",
            "profile_rows": profile.num_rows,
            "with_events": len(per_client),
            "history_min": lengths[0] if lengths else 0,
            "history_p25": quantile(lengths, 0.25),
            "history_median": quantile(lengths, 0.50),
            "history_p75": quantile(lengths, 0.75),
            "history_max": lengths[-1] if lengths else 0,
            "history_total": sum(lengths),
        },
        "time": {
            "definition": "разбор строки event_time; смещение — последние 6 символов; "
            "миллисекунды — наличие точки в строке",
            "earliest": earliest.isoformat() if earliest else None,
            "latest": latest.isoformat() if latest else None,
            "offsets": dict(offsets),
            "with_milliseconds": with_millis,
        },
        "sources": {
            "definition": "строк events.parquet с этим значением колонки source",
            "counts": dict(by_source.most_common()),
        },
        "types": {
            "definition": "events — строк с этим payload.type; clients — различных "
            "client_id среди них",
            "counts": {
                kind: {"events": count, "clients": len(clients_of_type[kind])}
                for kind, count in by_type.most_common()
            },
        },
        "products": {
            "definition": "событий с этим payload.product_id и различных клиентов среди них",
            "counts": {
                code: {"events": count, "clients": len(product_clients[code])}
                for code, count in products.most_common()
            },
        },
        "channels": {
            "definition": "событий с этим payload.channel",
            "counts": dict(channels.most_common()),
        },
        "statuses": {
            "definition": "событий с этим payload.status",
            "counts": dict(statuses.most_common()),
        },
        "transfers": {
            "definition": "группировка событий по payload.transfer_id; пара — id, "
            "встретившийся ровно дважды (две ноги одного перевода)",
            "distinct_ids": len(transfer_ids),
            "legs_per_id": {str(count): number for count, number in sorted(legs.items())},
            "pairs": legs.get(2, 0),
        },
        "decline_reasons": {
            "definition": "событий с этим payload.decline_reason",
            "counts": dict(decline_reasons.most_common()),
        },
        "scenarios": {
            name: {
                "definition": definition,
                "events": scenarios.get(name, 0),
                "clients": len(scenario_clients.get(name, ())),
            }
            for name, (definition, _) in SCENARIOS.items()
        },
        "field_presence": {
            "definition": "доля событий этого типа, у которых ключ присутствует в "
            "payload; знаменатель — все события типа; поля собраны первым проходом",
            "counts": {
                kind: {
                    field: round(present[kind][field] / by_type[kind], 4)
                    for field in sorted(fields_of[kind])
                }
                for kind in sorted(fields_of)
            },
        },
        "not_in_raw": ABSENT,
    }

    destination = (
        Path(args.out) if args.out else AUDIT / "evidence" / f"coverage-{args.run}.json"
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"покрытие -> {destination}")
    print(f"целостность: {[k for k, v in integrity.items() if v is True]}")
    print(f"клиентов {profile.num_rows}, типов {len(by_type)}, источников {len(by_source)}")

    empty = [name for name, item in report["scenarios"].items() if item["events"] == 0]

    print(f"сценариев без материала: {len(empty)} — {', '.join(empty) if empty else 'нет'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
