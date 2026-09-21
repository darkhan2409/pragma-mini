from __future__ import annotations

from collections import Counter, defaultdict

from ..config import CLIENT_ACTION_EVENT_TYPES


# ============================================================
# ИСТОРИИ КЛИЕНТОВ
# ============================================================
#
# Эталонная история читается целиком: скрытая причина рядом с
# наблюдаемым следствием. Именно так видно, что данные не
# набор независимых строк.
# ============================================================


WANTED = (
    ("заёмщик с просрочкой и восстановлением", lambda profile: profile["delinquencies"] > 0 and profile["cleared"] > 0),
    ("клиент, вернувшийся после паузы", lambda profile: profile["pauses"] > 0 and profile["returned"]),
    ("клиент со сменой работы", lambda profile: "job_change" in profile["life_events"]),
    ("клиент с переездом", lambda profile: "move" in profile["life_events"]),
    ("жертва мошенничества", lambda profile: profile["fraud"] > 0),
    ("вкладчик", lambda profile: profile["deposits"] > 0),
    ("клиент, зарегистрировавшийся в окне", lambda profile: profile["registered_in_window"]),
    ("высокоактивный цифровой клиент", lambda profile: profile["events"] > 2500 and profile["sessions"] > 40),
    ("спящий клиент с начислениями банка", lambda profile: profile["client_events"] < 40 and profile["bank_events"] > 5),
    ("клиент, мигрировавший на новый продукт", lambda profile: profile["migrations"] > 0),
)

KIND_TITLES = {
    "life_event": "жизненное событие",
    "stress_start": "начало стресса",
    "stress_end": "конец стресса",
    "pause_start": "начало паузы",
    "pause_end": "возвращение",
    "fraud_episode": "мошеннический эпизод",
    "state_transition": "смена состояния",
    "trait_shift": "сдвиг характеристик",
}

EVENT_TITLES = {
    "product_opened": "открыт договор",
    "product_migrated": "переход на другой продукт",
    "product_closed": "договор закрыт",
    "loan_disbursement": "выдача кредита",
    "delinquency_registered": "просрочка",
    "arrears_cleared": "просрочка погашена",
    "loan_closed": "кредит закрыт",
    "card_blocked": "карта заблокирована",
    "card_unblocked": "карта разблокирована",
    "card_reissued": "карта перевыпущена",
    "chargeback": "возврат по оспариванию",
    "case_opened": "обращение в поддержку",
    "application_submitted": "заявка",
    "application_decision": "решение по заявке",
    "profile_change": "изменение профиля",
    "salary_credit": "зарплата",
}


def _profiles(data: dict) -> dict:

    profiles: dict[str, dict] = {}

    counts: dict[str, Counter] = defaultdict(Counter)
    initiators: dict[str, Counter] = defaultdict(Counter)
    sessions: dict[str, set] = defaultdict(set)

    for row in data["events"]:
        counts[row["client_id"]][row["event_type"]] += 1
        if row["event_type"] in CLIENT_ACTION_EVENT_TYPES:
            initiators[row["client_id"]]["client"] += 1
        else:
            initiators[row["client_id"]]["bank"] += 1

        if row["event_type"] == "app_screen" and row["payload"].get("session_id"):
            sessions[row["client_id"]].add(row["payload"]["session_id"])

    truth_by_client: dict[str, list] = defaultdict(list)

    for row in data["truth_events"]:
        truth_by_client[row["client_id"]].append(row)

    for client in data["truth_clients"]:

        client_id = client["client_id"]

        events = truth_by_client.get(client_id, [])

        life = {row["key"] for row in events if row["kind"] == "life_event"}

        profiles[client_id] = {
            "client": client,
            "events": sum(counts[client_id].values()),
            "client_events": initiators[client_id].get("client", 0),
            "bank_events": initiators[client_id].get("bank_employee", 0)
            + initiators[client_id].get("system", 0),
            "sessions": len(sessions[client_id]),
            "delinquencies": counts[client_id].get("delinquency_registered", 0),
            "cleared": counts[client_id].get("arrears_cleared", 0),
            "deposits": counts[client_id].get("deposit_topup", 0),
            "fraud": counts[client_id].get("fraud_alert", 0),
            "migrations": counts[client_id].get("product_migrated", 0),
            "pauses": sum(1 for row in events if row["kind"] == "pause_start"),
            "returned": any(row["kind"] == "pause_end" for row in events),
            "life_events": life,
            "registered_in_window": client["registered_in_window"],
            "truth": events,
        }

    return profiles


def build_stories(data: dict, limit: int = 6) -> list:

    profiles = _profiles(data)

    by_client: dict[str, list] = defaultdict(list)

    for row in data["events"]:
        by_client[row["client_id"]].append(row)

    stories = []

    used: set[str] = set()

    for title, rule in WANTED:

        if len(stories) >= limit:
            break

        for client_id, profile in sorted(profiles.items()):

            if client_id in used:
                continue

            try:
                if not rule(profile):
                    continue
            except Exception:
                continue

            used.add(client_id)

            stories.append(_story(title, client_id, profile, by_client[client_id]))

            break

    return stories


def _story(title: str, client_id: str, profile: dict, events: list) -> dict:

    client = profile["client"]

    summary = (
        f"Клиент {client_id}: {client['life_stage']}, {client['settlement']} "
        f"({client['settlement_type']}), роль банка {client['hcb_role']}, "
        f"режим активности {client['activity_mode']}, состояние на конец окна {client['final_state']}. "
        f"Событий {profile['events']}, из них клиентских {profile['client_events']}, "
        f"сессий приложения {profile['sessions']}."
    )

    lines = []

    milestones = []

    for row in profile["truth"]:
        if row["kind"] in KIND_TITLES:
            milestones.append(
                (row["ts"], f"скрытое: {KIND_TITLES[row['kind']]} — {row['key']}")
            )

    for row in events:
        if row["event_type"] in EVENT_TITLES:
            payload = row["payload"]
            detail = EVENT_TITLES[row["event_type"]]
            extra = (
                payload.get("product_code")
                or payload.get("reason")
                or payload.get("decision")
                or payload.get("field_name")
                or ""
            )
            milestones.append((row["event_time"], f"наблюдаемое: {detail} {extra}".strip()))

    milestones.sort(key=lambda item: item[0])

    for moment, text in milestones[:40]:
        lines.append(f"{moment:%Y-%m-%d} {text}")

    return {
        "title": f"{title} ({client_id})",
        "client_id": client_id,
        "summary": summary,
        "timeline": lines,
    }


__all__ = ["build_stories"]
