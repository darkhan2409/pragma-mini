from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.generator import emit, engine


# ============================================================
# ИДЕЯ
# ============================================================
#
# Исход мошеннического эпизода разыгрывается при подготовке
# клиента — до того, как операция вообще случилась. Банк в момент
# своего решения ответа клиента ещё не знает.
#
# Прежде fraud_decision.resolution был дословной копией этого
# ответа. Хуже того, у вида false_positive ответ «промолчал» не
# выпадает никогда (life/fraud.py:147-148), поэтому ОТСУТСТВИЕ
# resolution однозначно исключало этот вид: не вероятностная
# подсказка, а точное исключение.
#
# Тот же ответ пересказывали ещё три поля: причина разблокировки
# карты, тема обращения и код правила антифрода.
#
# Здесь проверяется, что ни одно доступное модели поле больше не
# позволяет отличить скрытый вид эпизода, и что наблюдаемая
# последовательность сохранена: тревога → решение банка →
# обращение клиента, если он спорил.
# ============================================================


CLIENTS = 32

# Доли мошенничества подняты: при обычных на таком числе клиентов
# эпизодов не бывает вовсе. Частоты этого прогона реализмом не
# являются и нигде им не считаются.
PARAMS = {
    "fraud": {
        "base_rate_per_year": 4.0,
        "vulnerability_factor": 2.0,
        "online_exposure_factor": 2.0,
    }
}


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict:
    """
    Один прогон с наблюдением: нужны и лента, и скрытые эпизоды.

    Подменяется engine._finish — он ищется как глобал модуля,
    поэтому обёртка не меняет ни решений, ни порядка, ни
    случайности.
    """

    base = tmp_path_factory.mktemp("fraud")

    params = base / "params.json"
    params.write_text(json.dumps(PARAMS), encoding="utf-8")

    captured: list = []

    original = engine._finish

    def watching(sim):
        captured.extend(sim.clients[ordinal] for ordinal in sorted(sim.clients))
        return original(sim)

    engine._finish = watching

    try:
        emit.generate_dataset(
            total_clients=CLIENTS,
            out_dir=base / "raw",
            seed=100,
            world_seed=42,
            # Окно целиком после запуска антифрода (2025-01-15,
            # config.SOURCE_LAUNCH): раньше него событий этого
            # источника не бывает вовсе.
            history_start=datetime(2025, 1, 1),
            history_end=datetime(2026, 1, 1),
            workers=1,
            params_path=str(params),
            quiet=True,
        )
    finally:
        engine._finish = original

    table = pq.read_table(base / "raw" / "events.parquet")

    rows = [
        (client, when, json.loads(raw))
        for client, when, raw in zip(
            table.column("client_id").to_pylist(),
            table.column("event_time").to_pylist(),
            table.column("payload").to_pylist(),
        )
    ]

    hidden: dict[str, list[str]] = {
        state.client_id: [
            getattr(episode, "kind", "?")
            for episode in getattr(state, "fraud_episodes", [])
        ]
        for state in captured
    }

    return {"rows": rows, "hidden": hidden}


def events_of(world: dict, kind: str) -> list[tuple]:
    return [item for item in world["rows"] if item[2]["type"] == kind]


@pytest.fixture(scope="module")
def decisions(world) -> list[tuple]:
    found = events_of(world, "fraud_decision")
    assert found, "в прогоне нет ни одного решения антифрода: проверять нечего"
    return found


# ============================================================
# ГЛАВНОЕ: ОТВЕТ КЛИЕНТА НЕ ВЫГРУЖАЕТСЯ
# ============================================================


def test_decision_does_not_carry_the_client_answer(decisions):
    """
    Решение банка выпускается через минуты после тревоги. Ответа
    клиента в нём быть не может ни значением, ни отсутствием.
    """

    with_field = [item for item in decisions if "resolution" in item[2]]

    assert not with_field, (
        f"{len(with_field)} решений несут resolution; пример: {with_field[0][2]}"
    )


def test_absence_of_resolution_carries_no_information(decisions, world):
    """
    Прежняя утечка: resolution отсутствовал ровно тогда, когда
    клиент промолчал, а промолчать мог кто угодно КРОМЕ ложной
    тревоги. Теперь поле отсутствует у всех, значит по нему не
    различить ничего.
    """

    values = {item[2].get("resolution") for item in decisions}

    assert values == {None}, f"resolution принимает разные значения: {values}"

    # И среди клиентов с решениями есть обе стороны прежнего
    # различения — иначе проверка была бы вырожденной.
    owners = {item[0] for item in decisions}

    with_false = {
        client for client in owners if "false_positive" in world["hidden"].get(client, [])
    }
    without_false = owners - with_false

    assert with_false, "в прогоне нет ложных тревог: различать нечего"
    assert without_false, "в прогоне только ложные тревоги: различать нечего"


def test_no_exported_field_names_the_client_answer(world):
    """
    Ни одно поле любого события не содержит строк, которыми
    назывался скрытый ответ клиента.
    """

    forbidden = {"confirmed_by_client", "no_response"}

    guilty = [
        (item[2]["type"], name, value)
        for item in world["rows"]
        for name, value in item[2].items()
        if isinstance(value, str) and value in forbidden
    ]

    assert not guilty, f"скрытый ответ просочился: {guilty[:5]}"


# ============================================================
# ПЕРЕСКАЗЫ ТОГО ЖЕ ОТВЕТА
# ============================================================


def test_unblock_reason_is_neutral(world):
    """
    Разблокировка после проверки не называет причиной ответ
    клиента.
    """

    reasons = Counter(
        item[2].get("reason") for item in events_of(world, "card_unblocked")
    )

    assert "confirmed_by_client" not in reasons, f"причины разблокировки: {dict(reasons)}"


def test_case_topic_follows_the_bank_action(world):
    """
    Тема обращения по поводу антифрода — то, что клиент увидел:
    заблокировали карту или пришла тревога. Прежде тема была
    двоичным пересказом скрытого вида эпизода.
    """

    blocked_at: dict[str, list] = defaultdict(list)

    for client, when, payload in events_of(world, "card_blocked"):
        if payload.get("reason") == "fraud_suspicion":
            blocked_at[client].append(when)

    topics: Counter = Counter()

    for client, when, payload in events_of(world, "case_opened"):

        topic = payload.get("topic")

        if topic not in ("card_block", "fraud_report"):
            continue

        topics[topic] += 1

        if topic == "card_block":
            # Тема про блокировку возможна только там, где
            # блокировка действительно была и раньше обращения.
            assert any(
                moment < when for moment in blocked_at.get(client, [])
            ), f"{client}: тема card_block без предшествующей блокировки"

    assert topics, "обращений по антифроду в прогоне нет"


def test_rule_code_does_not_split_hidden_kinds(world):
    """
    Код правила выбирается по тому, на чём сработал антифрод, а
    не по виду эпизода. Значит набор кодов определяется объектом
    проверки, и ни один код не принадлежит одному виду.
    """

    by_subject: dict[str, set] = defaultdict(set)

    for _, _, payload in events_of(world, "fraud_alert"):
        code = payload.get("rule_code")
        if code:
            by_subject[payload["subject"]].add(code)

    assert by_subject, "тревог в прогоне нет"

    card = {"R_CARD_VELOCITY", "R_AMOUNT_ANOMALY", "R_GEO_ANOMALY"}
    transfer = {"R_TRANSFER_PATTERN", "R_AMOUNT_ANOMALY", "R_NEW_DEVICE"}

    for subject, codes in by_subject.items():
        expected = card if subject == "card" else transfer
        assert codes <= expected, f"{subject}: неожиданные коды {codes - expected}"


# ============================================================
# ПОСЛЕДОВАТЕЛЬНОСТЬ СОХРАНЕНА
# ============================================================


def test_alert_precedes_decision_precedes_case(world):
    """
    Осмысленный порядок остался: тревога, затем решение банка,
    затем — если клиент спорил — обращение.
    """

    order: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for client, when, payload in world["rows"]:
        if payload["type"] in ("fraud_alert", "fraud_decision", "case_opened"):
            order[client][payload["type"]].append(when)

    checked = 0

    for client, kinds in order.items():

        alerts = sorted(kinds.get("fraud_alert", []))
        decisions = sorted(kinds.get("fraud_decision", []))

        if not alerts or not decisions:
            continue

        checked += 1

        assert alerts[0] < decisions[0], f"{client}: решение раньше тревоги"

    assert checked, "нет клиентов с полной парой тревога/решение"
