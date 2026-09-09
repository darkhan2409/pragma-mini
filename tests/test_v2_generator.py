"""
Генератор V2: неизменность V1 и содержательные зависимости V2.

Первые тесты файла это эталон V1. Они сняты ДО первой правки
генератора и меняться не должны: git в проекте нет, и других
свидетелей прежнего поведения не существует.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.generator.config import FEATURE_END, HISTORY_START, LABEL_END
from src.generator.derive import derive_labels
from src.generator.emit import generate_dataset
from src.generator.history import STREAM_FIELDS, generate_client_history
from src.generator.persona import draw_persona
from src.generator.timeline import build_timeline
from src.generator.version import MANIFEST_KEY, manifest_version
from src.generator.v2.config import (
    AUTH_MAX_ATTEMPTS,
    BILL_KIND_OPERATION,
    BILL_MCC_GROUP,
    MAX_ATTEMPTS_PER_INTENT,
    MAX_SUPPORT_HOPS,
    V2_LATENT_NAMES,
    configuration,
)
from src.generator.v2.context import KIND_FAILURE, ClientContext, ContextView
from src.generator.v2.habits import client_habits, taste_at
from src.generator.v2.lifecycle import (
    FREE_PAYMENT_MCC,
    application_interest,
    derive_lifecycle_v2,
)
from src.generator.v2.outcomes import OperationContext, next_step_options, outcome_probabilities
from src.generator.v2.products import ProductStateV2
from src.generator.v2.scenarios import (
    BALANCE_CHECK,
    CARD_MANAGEMENT,
    PROFILE_SETTINGS,
    SCENARIO_STEPS,
    SUPPORT,
    catalog_screens,
    explore_target_weights,
    reachable_operations,
    reachable_screens,
    scenario_weights,
)
from src.generator.v2.sessions import run_session, session_starts_for_day
from src.generator.v2.state import (
    NEEDS_ACTIVE_CARD,
    StateView,
    operation_feasible,
    operation_offered,
)
from src.generator.v2.transactions import generate_transaction_history_v2
from src.generator.world import BROWSE_SCREENS, DOMAIN_OPERATIONS, REJECT_REASONS
from src.preprocessing.raw import RawDataset
from src.preprocessing.validate import validate_raw

from tests.test_raw_schema import LATENT_COLUMNS


ROOT = Path(__file__).resolve().parents[1]

SMOKE_DIR = ROOT / "data" / "raw" / "smoke"


# ============================================================
# ЭТАЛОН V1
# ============================================================
#
# sha256 от полной истории, ленты и метки одного клиента.
# Формула повторяет tests/test_determinism.py::HASH_SCRIPT.
# ============================================================

V1_DIGESTS = {
    0: "bfcef89615a6c203b53f1684908f49ba18f75b2b9bba9483198722ea5d364c83",
    3: "79e0c7666a87ced42b8d670ace66870c82207c166a7776480333624dbc0e0839",
    7: "4996bcd17b3928efa33a6483db87ac6d40601044c7bbc9f594464a8a16d097a8",
    11: "1cd336985b64724bcc2667f51fa91e6f462b436b93c6d120e5faece77652a568",
    673: "e4ea626d4b0b655f95a2ffae3a15643318639411da5530188845f42a3bf1b162",
}

# Побайтовые хэши data/raw/smoke: 100 клиентов, chunk 25, zstd.
V1_SMOKE_SHA256 = {
    "app_operations.parquet": "09f6691ce87ede4978a2980268d41547d8c30933d73dfc2f517dfb753c11db09",
    "app_screens.parquet": "420a57c4653cd7b084ad1cdf4144d1ced1d6eee3e5f99d0c45019a2a67f22061",
    "banners.parquet": "b4be39a80396aaa40e827ee7dce634d80231ce86c21ba719ebe459903c68188f",
    "communications.parquet": "966e2bac5d208aac8aa64e336d99e4ef98d6bfe4f127377a685938ce87f6e177",
    "labels.parquet": "2ca71ec2e5106ae9b72d054c6cdc737c305a7906aa3849cfad67ac87a6407af8",
    "manifest.json": "7e1dd79f3040f7a724d7eef47042e830ae176d93ba97561453b2e7824f91546c",
    "product_events.parquet": "0132588a9105c76c3d690e061ad749b6904132f78579abcafcb5aeee7873a215",
    "profile.parquet": "482b0fc841ce9ffcc7b5c42f79cf27bffc30ce8d46c280b5f97cc6b0089785a4",
    "source_coverage.parquet": "54d3565685392a11e323bd8d885d6271bfe8c5a4788a30b640d5b78250e268d7",
    "timeline.parquet": "e998fa964ebf696adb2c2ca25423c113b40fb213da2049d4d1e4c36231d195bd",
    "transactions.parquet": "b41a9f9f17e3166dd789d1ef5d072e835131ef68dd022fc8e58cc0eab5770b5c",
}


def client_digest(client_id: int, version: str | None = None) -> str:

    kwargs = {} if version is None else {"version": version}

    history = generate_client_history(
        client_id, start=HISTORY_START, end=LABEL_END, **kwargs
    )

    digest = hashlib.sha256()
    digest.update(repr(history).encode("utf-8"))
    digest.update(repr(build_timeline(history)).encode("utf-8"))
    digest.update(repr(derive_labels(history)).encode("utf-8"))

    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("client_id", sorted(V1_DIGESTS))
def test_v1_golden_digests_unchanged(client_id: int):
    """
    V1 не изменился. Литералы сняты до первой правки генератора.
    """

    assert client_digest(client_id) == V1_DIGESTS[client_id]


@pytest.mark.skipif(not SMOKE_DIR.exists(), reason="нет data/raw/smoke")
def test_v1_smoke_files_byte_identical(prep_raw_dir: Path):
    """
    Побайтовая воспроизводимость, а не только равенство данных:
    те же настройки записи должны давать те же файлы.
    """

    produced = {path.name: path for path in prep_raw_dir.iterdir() if path.is_file()}

    assert set(produced) == set(V1_SMOKE_SHA256)

    mismatched = {
        name: (V1_SMOKE_SHA256[name], file_sha256(path))
        for name, path in sorted(produced.items())
        if file_sha256(path) != V1_SMOKE_SHA256[name]
    }

    assert not mismatched, f"файлы V1 изменились: {sorted(mismatched)}"


# ============================================================
# ВЕРСИЯ ПО УМОЛЧАНИЮ
# ============================================================


def test_default_version_is_v1(raw_dir: Path, v2_raw_dir: Path):
    """
    Умолчание не изменилось, и манифест v1 остался прежним:
    ключ версии появляется только у v2.
    """

    assert generate_client_history(7) == generate_client_history(7, version="v1")

    v1_manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    v2_manifest = json.loads((v2_raw_dir / "manifest.json").read_text(encoding="utf-8"))

    assert MANIFEST_KEY not in v1_manifest
    assert manifest_version(v1_manifest) == "v1"
    assert v2_manifest[MANIFEST_KEY] == "v2.1"


# ============================================================
# ВОСПРОИЗВОДИМОСТЬ V2
# ============================================================


def test_v2_worker_count_independence(
    v2_raw_dir: Path, tmp_path_factory: pytest.TempPathFactory
):
    """
    Побайтово одинаковый датасет при любом числе воркеров.
    """

    other = tmp_path_factory.mktemp("raw_v2_workers")

    generate_dataset(
        total_clients=24,
        chunk_clients=8,
        out_dir=other,
        workers=2,
        version="v2.1",
    )

    produced = sorted(path.name for path in v2_raw_dir.iterdir() if path.is_file())

    assert produced

    mismatched = [
        name
        for name in produced
        if file_sha256(v2_raw_dir / name) != file_sha256(other / name)
    ]

    assert not mismatched


V2_HASH_SCRIPT = """
import hashlib, os, sys
sys.path.insert(0, os.getcwd())

from src.generator.config import HISTORY_START, LABEL_END
from src.generator.derive import derive_labels
from src.generator.history import generate_client_history
from src.generator.timeline import build_timeline

def digest(client_id):
    history = generate_client_history(
        client_id, start=HISTORY_START, end=LABEL_END, version="v2.1"
    )
    h = hashlib.sha256()
    h.update(repr(history).encode("utf-8"))
    h.update(repr(build_timeline(history)).encode("utf-8"))
    h.update(repr(derive_labels(history)).encode("utf-8"))
    return h.hexdigest()
"""


def test_v2_deterministic_across_processes():
    """
    Другой PYTHONHASHSEED ловит зависимость от обхода set/dict.
    """

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "12345"

    result = subprocess.run(
        [sys.executable, "-c", V2_HASH_SCRIPT + "\nprint(digest(3))"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == client_digest(3, version="v2.1")


def test_v2_raw_passes_validate_raw_without_latent(
    v2_raw_dir: Path, v2_raw_tables: dict
):
    """
    RAW версии v2 проходит те же 17 проверок, что и v1,
    и не содержит ни одного скрытого имени.
    """

    report = validate_raw(RawDataset(v2_raw_dir))

    assert report["status"] == "ok"

    forbidden = LATENT_COLUMNS | V2_LATENT_NAMES

    for name, table in v2_raw_tables.items():
        assert not (forbidden & set(table.columns)), name

    keys: set[str] = set()

    for payload in v2_raw_tables["timeline"].payload:
        keys |= set(json.loads(payload))

    assert not (forbidden & keys)


def test_v2_feature_prefix_stable_when_horizon_extends():
    """
    Продление горизонта не меняет прошлое НИ В ОДНОМ потоке.

    Это главная проверка причинности: контекст сессий и счета
    зависят только от того, что случилось раньше.
    """

    for client_id in (3, 673):

        full = generate_client_history(
            client_id, start=HISTORY_START, end=LABEL_END, version="v2.1"
        )
        short = generate_client_history(
            client_id, start=HISTORY_START, end=FEATURE_END, version="v2.1"
        )

        for field in STREAM_FIELDS.values():
            assert getattr(full.before(FEATURE_END), field) == getattr(short, field), (
                f"{client_id} / {field}"
            )


# ============================================================
# ПРИЧИННОСТЬ ПО ВРЕМЕНИ СОБЫТИЯ
# ============================================================


def test_v2_context_respects_event_timestamps():
    """
    Сессии перекрываются. Сбой в 09:50 из сессии, начатой в 09:00,
    не виден сессии, начатой в 09:15, и не виден её шагу в 09:40,
    но виден шагу в 09:56.
    """

    context = ClientContext()

    context.record(datetime(2025, 6, 1, 9, 5), KIND_FAILURE, "pay_utility")
    context.record(datetime(2025, 6, 1, 9, 50), KIND_FAILURE, "transfer_card")

    early = context.view(datetime(2025, 6, 1, 9, 15))
    middle = context.view(datetime(2025, 6, 1, 9, 40))
    late = context.view(datetime(2025, 6, 1, 9, 56))

    assert early.recent_failures == ("pay_utility",)
    assert middle.recent_failures == ("pay_utility",)
    assert late.recent_failures == ("pay_utility", "transfer_card")

    # На реальном клиенте: ни одна запись, увиденная сессией,
    # не может быть моложе её начала.
    lifecycle = derive_lifecycle_v2(3, HISTORY_START, FEATURE_END)

    journal = ClientContext()

    for event in lifecycle.app_operations:
        if event.status == "failed":
            journal.record(event.ts, KIND_FAILURE, event.operation)

    for day in (datetime(2025, 6, 1), datetime(2025, 9, 1)):
        for _, started_at in session_starts_for_day(3, day):
            seen = journal.view(started_at)
            assert all(
                record.ts < started_at
                for record in journal.records
                if record.key in seen.recent_failures
                and (started_at - record.ts).days <= 3
            )


# ============================================================
# СОСТОЯНИЕ МЕНЯЕТСЯ ВМЕСТЕ С ОПЕРАЦИЕЙ
# ============================================================


# ============================================================
# СЧЕТА ЗАМЕНЯЮТ СЛУЧАЙНЫЕ ПЛАТЕЖИ
# ============================================================


def test_v2_bills_replace_random_payments():
    """
    Платёж по счёту не добавляется к случайной покупке, а заменяет
    её: MCC счетов не встречаются вне реестра оплат.
    """

    for client_id in (0, 3, 11, 23):

        lifecycle = derive_lifecycle_v2(client_id, HISTORY_START, FEATURE_END)

        transactions = generate_transaction_history_v2(
            client_id, HISTORY_START, FEATURE_END, lifecycle
        )

        billed = [
            event
            for event in transactions
            if event.mcc in BILL_MCC_GROUP and not event.is_subscription
        ]

        ledger = [
            payment
            for payment in lifecycle.payments
            if HISTORY_START <= payment.ts < FEATURE_END
        ]

        assert abs(len(billed) - len(ledger)) <= 1, client_id
        assert set(event.mcc for event in billed) <= set(BILL_MCC_GROUP)

        for payment in ledger:

            if payment.paid_via != "app":
                continue

            operation = BILL_KIND_OPERATION.get(payment.kind, "pay_tax")

            assert any(
                event.ts == payment.ts
                and event.status == "success"
                and event.operation in FREE_PAYMENT_MCC
                for event in lifecycle.app_operations
            ), (client_id, payment)

            delays = [
                (event.ts - payment.ts).total_seconds()
                for event in billed
                if event.amount == payment.amount
                and event.mcc == payment.mcc
                and event.ts >= payment.ts
            ]

            assert any(5 <= delay <= 60 for delay in delays), (client_id, payment)


def test_v2_transactions_habitual_and_novel():
    """
    У клиента есть свои точки и свои суммы, но история не
    превращается в копирование одного шаблона.
    """

    for client_id in (3, 11):

        events = [
            event
            for event in generate_transaction_history_v2(
                client_id, HISTORY_START, FEATURE_END
            )
            if event.direction == "debit" and not event.is_subscription
        ]

        combos = Counter(
            (event.mcc, event.merchant_city, event.is_online) for event in events
        )

        # Привычные точки: у v1 те же десять сочетаний покрывают
        # 0.31-0.36 покупок, у v2 заметно больше.
        top10 = sum(count for _, count in combos.most_common(10)) / len(events)

        assert top10 > 0.42, (client_id, top10)

        # Но история не сводится к одному шаблону: новые покупки
        # дают длинный хвост сочетаний, встреченных один раз.
        assert sum(1 for count in combos.values() if count == 1) >= 15, client_id

        # Привычная сумма повторяется: в v1 максимум 3-4 раза.
        amounts = Counter(event.amount for event in events)

        assert amounts.most_common(1)[0][1] >= 8, client_id


def test_v2_subscription_lifecycle():
    """
    Подписка начинается, иногда прекращается и изредка меняет
    сумму. Город у неё один: это одна и та же площадка.
    """

    started_late = stopped = changed = 0

    for client_id in range(24):

        habits = client_habits(client_id)

        for subscription in habits.subscriptions:

            if subscription.start_month > 0:
                started_late += 1

            if subscription.end_month is not None:
                stopped += 1

            if subscription.change_month is not None:
                changed += 1

        events = [
            event
            for event in generate_transaction_history_v2(
                client_id, HISTORY_START, FEATURE_END
            )
            if event.is_subscription
        ]

        cities: dict[str, set] = {}

        for event in events:
            cities.setdefault(event.mcc, set()).add(event.merchant_city)

        for mcc, values in cities.items():
            assert len(values) == 1, (client_id, mcc)

    assert started_late and stopped and changed


# ============================================================
# ЗАВИСИМОСТИ ОТ ОТНОСЯЩЕГОСЯ СОСТОЯНИЯ
# ============================================================


def transfer_context(**kwargs) -> OperationContext:
    return OperationContext(operation="transfer_card", domain="transfers", **kwargs)


def test_v2_probabilities_depend_on_relevant_state_only():
    """
    Меняется то, что относится к делу, и не меняется то, что нет.
    """

    plain = outcome_probabilities(transfer_context())

    assert abs(sum(plain) - 1.0) < 1e-9
    assert all(0.02 <= value <= 0.98 for value in plain)

    assert outcome_probabilities(transfer_context(attempt=2))[1] < plain[1]
    assert outcome_probabilities(transfer_context(outage=True))[1] > plain[1]
    assert outcome_probabilities(transfer_context(stress=0.9))[1] > plain[1]
    assert outcome_probabilities(transfer_context(card_blocked=True))[1] > plain[1]

    # Стресс не мешает смотреть карту, а утилизация лимита
    # относится только к смене лимита.
    read_only = OperationContext(operation="card_view", domain="cards")

    assert outcome_probabilities(read_only) == outcome_probabilities(
        OperationContext(operation="card_view", domain="cards", stress=0.9)
    )

    assert outcome_probabilities(transfer_context(utilization=0.9)) == plain

    limit = OperationContext(operation="limit_change", domain="cards")

    assert (
        outcome_probabilities(
            OperationContext(operation="limit_change", domain="cards", utilization=0.9)
        )[1]
        > outcome_probabilities(limit)[1]
    )


def context_view(**kwargs) -> ContextView:

    defaults = dict(
        recent_failures=(),
        recent_offers=frozenset(),
        recent_reminder=False,
        recent_rejections=frozenset(),
        views={},
        unfinished="",
        due_bills=(),
        card_blocked=False,
        credit_need=0.5,
    )

    defaults.update(kwargs)

    return ContextView(**defaults)


def test_v2_scenario_weights_depend_on_context():
    """
    Выбор сценария следует за доступностью и за недавним прошлым.
    """

    app = client_habits(3).app

    adopted = ("home", "profile", "cards", "payments", "support", "deposits")

    with_card = scenario_weights(app, context_view(), frozenset({"debit_card"}), adopted)
    without = scenario_weights(app, context_view(), frozenset(), adopted)

    assert with_card[CARD_MANAGEMENT] > 0.0
    assert without[CARD_MANAGEMENT] == 0.0

    blocked = scenario_weights(
        app, context_view(card_blocked=True), frozenset({"debit_card"}), adopted
    )

    assert blocked[CARD_MANAGEMENT] > with_card[CARD_MANAGEMENT]

    failed = scenario_weights(
        app,
        context_view(recent_failures=("pay_utility",)),
        frozenset({"debit_card"}),
        adopted,
    )

    assert failed[SUPPORT] > with_card[SUPPORT]
    assert failed[BALANCE_CHECK] == with_card[BALANCE_CHECK]

    offered = explore_target_weights(
        app, context_view(recent_offers=frozenset({"deposit"})), frozenset(), adopted
    )
    neutral = explore_target_weights(app, context_view(), frozenset(), adopted)

    assert offered["deposit"] > neutral["deposit"]


def test_v2_interest_and_habit_drift():
    """
    Интерес растёт с глубиной и повторами и падает после отказа.
    Вкус смещается постепенно, а не одним скачком.
    """

    shallow = application_interest("deposit", "root", 0, False, False, 0.5)
    deep = application_interest("deposit", "terms", 0, False, False, 0.5)
    repeated = application_interest("deposit", "root", 3, False, False, 0.5)
    rejected = application_interest("deposit", "root", 0, True, False, 0.5)
    offered = application_interest("deposit", "root", 0, False, True, 0.5)

    assert deep > shallow
    assert repeated > shallow
    assert rejected < shallow
    assert offered > shallow

    client_id = next(
        cid for cid in range(50) if len(client_habits(cid).drift_points) >= 1
    )

    habits = client_habits(client_id)
    point = habits.drift_points[0]

    before = taste_at(habits, point.day - 1)
    inside = taste_at(habits, point.day + point.blend_days // 2)
    after = taste_at(habits, point.day + point.blend_days + 1)

    assert before.mix == 0.0
    assert 0.0 < inside.mix < 1.0
    assert before.taste != after.taste


# ============================================================
# V2.1: РЕГРЕССИИ НА ИСПРАВЛЕННЫЕ ДЕФЕКТЫ
# ============================================================
#
# Все проверки идут по ВНУТРЕННЕМУ результату жизненного цикла,
# то есть ДО наблюдательного шума: шум обнуляет часть статусов
# и городов и скрыл бы нарушение причинности.
# ============================================================


V21_CLIENTS = (0, 3, 11, 17, 23)


def lifecycles(clients=V21_CLIENTS):
    return [
        (client_id, derive_lifecycle_v2(client_id, HISTORY_START, FEATURE_END))
        for client_id in clients
    ]


def test_v21_protected_actions_require_authorization():
    """
    Экран и защищённое действие невозможны до успешного входа,
    а сессия без входа не оставляет ничего, кроме самих попыток.
    """

    checked = failed_sessions = resumed = 0

    for _, lifecycle in lifecycles():

        for run in lifecycle.runs:

            if not run.authorized:
                failed_sessions += 1
                assert run.screens == 0
                assert run.actions == 0
                continue

            if not run.auth_required:
                resumed += 1
                continue

            checked += 1

            assert run.auth_success_ts is not None

            if run.first_screen_ts is not None:
                assert run.first_screen_ts > run.auth_success_ts

            if run.first_action_ts is not None:
                assert run.first_action_ts > run.auth_success_ts

    # Обе ветки должны встречаться: и вход, и продолжение сессии.
    assert checked and resumed and failed_sessions


def test_v21_auth_attempts_are_bounded():
    """
    Попыток входа не больше предела, а после провала клиент
    либо восстанавливает доступ, либо уходит.
    """

    for _, lifecycle in lifecycles():

        for run in lifecycle.runs:
            assert run.auth_failures <= AUTH_MAX_ATTEMPTS

        by_session: dict[str, int] = {}

        for run in lifecycle.runs:
            by_session[run.session_id] = run.auth_failures

        assert all(value <= AUTH_MAX_ATTEMPTS for value in by_session.values())


def test_v21_successful_payment_has_exactly_one_effect():
    """
    Каждая успешная pay_* исполняет ровно одно намерение и даёт
    ровно одно списание. Неуспешная не даёт ничего.
    """

    seen_bill = seen_free = 0

    for client_id, lifecycle in lifecycles():

        successes = [
            event
            for event in lifecycle.app_operations
            if event.operation in FREE_PAYMENT_MCC and event.status == "success"
        ]

        by_key = [
            payment
            for payment in lifecycle.payments
            if payment.paid_via == "app"
        ]

        assert len(successes) == len(by_key), client_id

        keys = [payment.op_key for payment in by_key]

        assert len(keys) == len(set(keys)), client_id

        for payment in by_key:
            if payment.bill_key is None:
                seen_free += 1
                assert payment.kind == "free"
            else:
                seen_bill += 1

        # Счёт гасится ровно один раз.
        paid = [
            payment.bill_key
            for payment in lifecycle.payments
            if payment.bill_key is not None
        ]

        assert len(paid) == len(set(paid)), client_id

        # Списание есть на каждую запись и только на неё.
        transactions = generate_transaction_history_v2(
            client_id, HISTORY_START, FEATURE_END, lifecycle
        )

        billed = [
            event
            for event in transactions
            if event.mcc in BILL_MCC_GROUP and not event.is_subscription
        ]

        inside = [
            payment
            for payment in lifecycle.payments
            if HISTORY_START <= payment.ts < FEATURE_END
        ]

        assert abs(len(billed) - len(inside)) <= 1, client_id

    assert seen_bill and seen_free


def test_v21_failed_payment_creates_no_charge():
    """
    Неуспешная оплата не создаёт ни записи, ни списания.
    """

    for _, lifecycle in lifecycles():

        moments = {
            payment.ts
            for payment in lifecycle.payments
            if payment.paid_via == "app"
        }

        for event in lifecycle.app_operations:

            if event.operation not in FREE_PAYMENT_MCC:
                continue

            if event.status == "success":
                continue

            assert event.ts not in moments or any(
                other.ts == event.ts and other.status == "success"
                for other in lifecycle.app_operations
                if other.operation in FREE_PAYMENT_MCC
            )


def test_v21_state_changes_take_effect_at_their_moment():
    """
    Договор не появляется раньше решения, а блокировка карты
    действует ровно с момента успешной операции.
    """

    for _, lifecycle in lifecycles():

        state = lifecycle.state

        approved = [
            screen
            for screen in lifecycle.app_screens
            if screen.funnel_stage == "approved"
        ]

        for screen in approved:

            later = [
                holding
                for holding in state.holdings
                if holding.product_type == screen.product
                and holding.opened_at > screen.ts
            ]

            earlier_same_day = [
                event
                for event in lifecycle.product_events
                if event.product_type == screen.product
                and screen.ts - timedelta(days=2) < event.ts < screen.ts
            ]

            assert later or not earlier_same_day

        blocks = state.blocked_intervals()

        for event in lifecycle.app_operations:

            if event.status != "success":
                continue

            if event.operation == "card_block":
                assert any(start == event.ts for start, _ in blocks)

            if event.operation == "card_unblock":
                assert any(end == event.ts for _, end in blocks)


def test_v21_overlapping_sessions_share_one_timeline():
    """
    Перекрывающиеся сессии исполняются в общем порядке событий.

    Проверяется по последствию: успешная блокировка карты в одной
    сессии делает невозможным успешное списание в другой, если
    оно произошло позже по времени.
    """

    overlaps = 0

    for _, lifecycle in lifecycles():

        spans: dict[str, list] = {}

        for screen in lifecycle.app_screens:
            span = spans.setdefault(screen.session_id, [screen.ts, screen.ts])
            span[0] = min(span[0], screen.ts)
            span[1] = max(span[1], screen.ts)

        items = sorted(spans.values())

        for index in range(1, len(items)):
            if items[index][0] < items[index - 1][1]:
                overlaps += 1

        state = lifecycle.state

        for event in lifecycle.app_operations:

            if event.status != "success":
                continue

            if event.operation in NEEDS_ACTIVE_CARD:
                assert not state.card_blocked_at(event.ts)

    assert overlaps, "в выборке нет перекрывающихся сессий"


def test_v21_hard_constraints_precede_the_draw():
    """
    Невозможное действие не может закончиться успехом:
    пол вероятности к успеху не применяется.
    """

    blocked = OperationContext(
        operation="pay_utility", domain="payments", card_blocked=True, feasible=False
    )

    probabilities = outcome_probabilities(blocked)

    assert probabilities[0] == 0.0
    assert abs(sum(probabilities) - 1.0) < 1e-9
    assert all(value >= 0.02 for value in probabilities[1:])

    view = StateView(ts=HISTORY_START, owned=frozenset({"debit_card"}), accounts=1)

    # Перевод между своими счетами требует второго счёта.
    assert not operation_offered("transfer_own", view)

    both = StateView(
        ts=HISTORY_START,
        owned=frozenset({"debit_card", "deposit"}),
        accounts=2,
        authorized=True,
    )

    assert operation_offered("transfer_own", both)

    # Закрыть можно только существующий открытый договор.
    assert not operation_offered("deposit_close", both)

    closable = StateView(
        ts=HISTORY_START,
        owned=frozenset({"debit_card", "deposit"}),
        closable=frozenset({"deposit"}),
        accounts=2,
        authorized=True,
    )

    assert operation_offered("deposit_close", closable)

    # Списание по заблокированной карте не проходит.
    frozen = StateView(
        ts=HISTORY_START,
        owned=frozenset({"debit_card"}),
        accounts=1,
        card_blocked=True,
        authorized=True,
    )

    assert not operation_feasible("pay_utility", frozen)
    assert operation_feasible("card_unblock", frozen)

    # Без авторизации защищённое действие невозможно.
    anonymous = StateView(ts=HISTORY_START, owned=frozenset({"debit_card"}), accounts=1)

    assert not operation_feasible("card_view", anonymous)
    assert operation_feasible("login", anonymous)


def test_v21_reject_reason_matches_the_cause():
    """
    Причина отказа соответствует состоянию, которое к нему привело.
    """

    seen = set()

    for client_id, lifecycle in lifecycles(range(24)):

        persona = draw_persona(client_id)

        for screen in lifecycle.app_screens:

            if screen.funnel_stage != "rejected":
                continue

            reason = screen.reject_reason
            seen.add(reason)

            owned = lifecycle.state.owned_at(screen.ts)

            if reason == "age_limit":
                assert persona.age < 21 or persona.age > 70

            if reason == "income_not_confirmed":
                assert (
                    persona.income_type in ("unemployed", "student")
                    or persona.declared_income < 90_000
                )

            if reason == "existing_debt":
                assert screen.product in ("cash_loan", "credit_card")
                assert "cash_loan" in owned

    assert seen

    # Сам набор причин не расширился.
    assert seen <= set(REJECT_REASONS)


def test_v21_repeat_generation_is_byte_identical(
    v2_raw_dir: Path, tmp_path_factory: pytest.TempPathFactory
):
    """
    Повторная генерация даёт те же файлы: версия, seed и
    конфигурация записаны в манифест.
    """

    again = tmp_path_factory.mktemp("raw_v21_again")

    generate_dataset(
        total_clients=24,
        chunk_clients=8,
        out_dir=again,
        workers=1,
        version="v2.1",
    )

    for path in sorted(v2_raw_dir.iterdir()):
        if path.is_file():
            assert file_sha256(path) == file_sha256(again / path.name), path.name

    manifest = json.loads((again / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["seed"] == 42
    assert manifest[MANIFEST_KEY] == "v2.1"
    assert manifest["generation_config"] == configuration()
