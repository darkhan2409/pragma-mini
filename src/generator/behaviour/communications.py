from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .. import params as params_module
from ..life.persona import Persona
from ..rng import COMPONENT_CONTENT, COMPONENT_TIME, NS_COMM, day_rng, event_rng, stable_hash
from ..world.dictionaries import (
    CAMPAIGNS,
    CAMPAIGNS_BY_FAMILY,
    CAMPAIGN_BY_CODE,
    DELIVERY_RATE,
)


# ============================================================
# КОММУНИКАЦИИ
# ============================================================
#
# Причинная цепочка:
#
#   кампания -> контакт -> доставлено или нет -> открыто
#   -> клик -> действие или его отсутствие
#
# Клик и сам факт интереса остаются внутри симуляции: банк
# видит отправку и доставку, а реакция проявляется только
# последующими действиями клиента.
#
# Учитываются усталость от коммуникаций, согласие клиента,
# предпочитаемый канал и актуальность предложения.
# ============================================================


SEND_HOURS = tuple(range(9, 21))

SEND_WEIGHTS = (
    0.06, 0.09, 0.11, 0.11, 0.10, 0.09,
    0.09, 0.08, 0.08, 0.07, 0.07, 0.05,
)


@dataclass(frozen=True)
class Contact:
    ts: datetime
    campaign_code: str
    purpose: str
    channel: str
    template: str
    product_family: str | None
    delivered: bool
    clicked: bool
    offer_id: str | None


def daily_rate(persona: Persona, ts: datetime, consented: bool, fatigue: int,
               state_factor: float = 1.0) -> float:
    """
    Банк пишет клиенту заметно реже, чем клиент заходит в
    приложение. Усталость от коммуникаций гасит частоту.
    """

    if not consented:
        return 0.0

    # Частота вынесена в параметры, но значение пришпилено к
    # ИЗМЕРЕННОМУ якорю банка: около 3.8 отправок на клиента в
    # месяц (calibration.communications_per_client_month, отчёт
    # банка, уверенность high).
    #
    # Поднимать её ради объёма нельзя: это единственная из пяти
    # частот, у которой есть настоящее измерение, а не гипотеза.
    # Клиент и так получит больше сообщений — просто потому, что
    # у него станет больше договоров и платежей, на которые банк
    # отвечает сервисными уведомлениями.
    rate = params_module.active().activity.communications_base_per_month / 30.0

    # Молчащему клиенту банк пишет заметно реже: остаются
    # только сервисные сообщения и возврат в игру.
    rate *= max(0.05, state_factor)

    rate *= 0.75 + 0.5 * persona.trait("digital_affinity", ts)

    if ts.weekday() >= 5:
        rate *= 0.80

    if ts.month == 12:
        rate *= 1.15

    rate *= max(0.45, 1.0 - 0.05 * fatigue)

    return float(rate)


def _collection_template(dpd: int, days_to_due: int | None) -> str:
    """
    Какой текст уместен при этом состоянии долга.
    """

    if dpd >= 60:
        return "PR_RESTRUCTURE"

    if dpd >= 30:
        return "PR_OVERDUE_HARD"

    if dpd > 0:
        return "PR_OVERDUE_SOFT"

    if days_to_due is not None and days_to_due <= 0:
        return "PR_DUE_TODAY"

    return "PR_DUE_SOON"


def _campaign_weights(
    persona: Persona,
    ts: datetime,
    owned_families: frozenset,
    candidate_families: frozenset,
    dpd: int,
    in_pause: bool,
    stress: float,
    pending_notice: bool,
    fraud_alert: bool,
    days_to_due: int | None = None,
) -> dict:

    weights: dict[str, float] = {}

    for campaign in CAMPAIGNS:

        weight = 1.0

        if campaign.purpose == "offer":

            if campaign.family is None:
                weight = 1.2
            elif campaign.family in candidate_families:
                weight = 2.2
            elif campaign.family in owned_families:
                weight = 0.12
            else:
                weight = 0.20

            if campaign.family in ("cash_loan", "refinance", "credit_card"):
                weight *= 0.4 + 2.0 * persona.trait("credit_appetite", ts)
                weight *= 1.0 + 1.3 * stress

            if campaign.family in ("deposit", "deposit_certificate", "bonds"):
                weight *= 0.4 + 1.8 * persona.trait("savings_propensity", ts)

        elif campaign.purpose == "collection":
            # Напоминание о платеже приходит ДО срока. Раньше
            # вес был нулевым, пока клиент не просрочил, и
            # «платёж скоро» уходило уже после дефолта.
            reminder = params_module.active().activity.due_reminder_days
            if dpd > 0:
                weight = 3.0 + 0.08 * min(90, dpd)
            elif days_to_due is not None and 0 <= days_to_due <= reminder:
                weight = 2.4
            else:
                weight = 0.0

        elif campaign.purpose == "winback":
            weight = 2.2 if in_pause else 0.0

        if in_pause and campaign.purpose == "offer":
            # Клиенту, который замолчал, предложения почти не шлют.
            weight *= 0.15

        elif campaign.purpose == "security":
            weight = 2.6 if fraud_alert else 0.25

        elif campaign.purpose == "service":
            weight = 2.4 if pending_notice else 0.9

        elif campaign.purpose == "survey":
            weight = 0.3 * (0.4 + 1.4 * persona.trait("digital_affinity", ts))

        if weight > 0.0:
            weights[campaign.code] = weight

    return weights


def _channel(persona: Persona, campaign, ts: datetime, app_adopted: bool, rng) -> str:

    preferences = persona.traits.channel_preferences

    weights = []

    for channel in campaign.channels:

        weight = 1.0

        if channel == "push":
            weight = (0.5 + 1.6 * persona.trait("digital_affinity", ts))
            if not app_adopted:
                weight *= 0.12
        elif channel == "sms":
            weight = 1.2 - 0.4 * persona.trait("digital_affinity", ts)
        elif channel == "call":
            weight = 0.7 * (1.0 + preferences.get("call_center", 0.05) * 3.0)
        else:
            weight = 0.4

        weights.append(max(0.01, weight))

    return str(rng.choice(list(campaign.channels), p=weights))


def _delivered(channel: str, persona: Persona, app_adopted: bool, rng) -> bool:

    if channel == "push" and not app_adopted:
        return False

    return rng.random() < DELIVERY_RATE.get(channel, 0.5)


def _clicked(persona: Persona, campaign, channel: str, ts: datetime, stress: float, rng) -> bool:
    """
    Скрытая реакция на доставленное сообщение.
    """

    probability = 0.04 + 0.16 * persona.trait("digital_affinity", ts)

    if channel == "call":
        probability += 0.10

    if campaign.family in ("cash_loan", "refinance", "credit_card"):
        probability += 0.20 * persona.trait("credit_appetite", ts) + 0.12 * stress
    elif campaign.family in ("deposit", "deposit_certificate", "bonds"):
        probability += 0.14 * persona.trait("savings_propensity", ts)

    if campaign.purpose == "collection":
        probability += 0.18

    return rng.random() < min(0.80, probability)


# Шаблоны с условием на аудиторию.
#
# Кампания выбирается по семейству продукта, а текст внутри неё —
# по весу позиции, и этого мало: «поднимем лимит вашей карты»
# уходило человеку без кредитной карты, а пенсионный вклад —
# тридцатилетнему. Здесь перечислено, кому такой текст уместен;
# всё остальное внутри кампании пишется кому угодно.
#
# Проверка положительная: шаблона нет в таблице — значит условий
# у него нет. Так новый текст не становится молча запрещённым.
TEMPLATE_AUDIENCE: dict[str, str] = {
    # Повышение лимита обсуждают с тем, у кого лимит есть.
    "CC_LIMIT_UP": "credit_card",
    # Пенсионный вклад и пенсионная карта — пенсионерам.
    "DEP_PENSION": "pensioner",
    # Рефинансирование чужого долга предлагают тому, у кого есть
    # действующий кредит.
    "REFIN_BASE": "loan",
    "REFIN_LOWER_PAYMENT": "loan",
    # Досрочное закрытие и продление вклада — владельцам вклада.
    "CERT_FLEX": "deposit",
    # Кешбэк по категории имеет смысл при действующей карте.
    "CB_CATEGORY_FOOD": "card",
    "CB_CATEGORY_FUEL": "card",
    "CB_PARTNER": "card",
}


def _audience_allows(rule: str, persona: Persona, ts: datetime, owned_families: frozenset) -> bool:
    """
    Подходит ли клиент под условие шаблона.
    """

    if rule == "pensioner":
        return persona.is_pensioner_at(ts)

    if rule == "credit_card":
        return "credit_card" in owned_families

    if rule == "deposit":
        return bool({"deposit", "deposit_certificate"} & owned_families)

    if rule == "card":
        return bool({"credit_card", "debit_card"} & owned_families)

    if rule == "loan":
        return bool({"cash_loan", "installment", "refinance"} & owned_families)

    return True


def _template_for(campaign, persona, ts, owned_families, rng) -> str | None:
    """
    Текст кампании, уместный этому клиенту.

    Если ни один текст кампании ему не подходит, письма не будет
    вовсе: лучше промолчать, чем отправить предложение, которое
    человеку не о чем читать.
    """

    allowed = [
        name
        for name in campaign.templates
        if _audience_allows(TEMPLATE_AUDIENCE.get(name, ""), persona, ts, owned_families)
    ]

    if not allowed:
        return None

    weights = [1.0 / ((position + 1) ** 1.4) for position in range(len(allowed))]

    total = sum(weights)

    return str(rng.choice(list(allowed), p=[value / total for value in weights]))


def contacts_for_day(
    persona: Persona,
    day: datetime,
    consented: bool,
    app_adopted: bool,
    fatigue: int,
    state_factor: float,
    owned_families: frozenset,
    candidate_families: frozenset,
    dpd: int,
    in_pause: bool,
    stress: float,
    pending_notice: bool,
    fraud_alert: bool,
    days_to_due: int | None = None,
) -> tuple:
    """
    Отправки банка за день.
    """

    rate = daily_rate(persona, day, consented, fatigue, state_factor)

    if rate <= 0.0:
        return ()

    ordinal = day.toordinal()

    count = day_rng(NS_COMM, persona.client_ordinal, ordinal).poisson(rate)

    if count <= 0:
        return ()

    weights = _campaign_weights(
        persona, day, owned_families, candidate_families, dpd, in_pause,
        stress, pending_notice, fraud_alert, days_to_due,
    )

    if not weights:
        return ()

    result: list[Contact] = []

    for index in range(count):

        rng = event_rng(NS_COMM, persona.client_ordinal, ordinal, index, COMPONENT_CONTENT)
        time_rng = event_rng(NS_COMM, persona.client_ordinal, ordinal, index, COMPONENT_TIME)

        code = rng.weighted(weights)

        campaign = CAMPAIGN_BY_CODE[code]

        channel = _channel(persona, campaign, day, app_adopted, rng)

        if campaign.purpose == "collection":
            # Текст напоминания следует состоянию долга, а не
            # позиции в списке. Иначе «платёж скоро» уходит на
            # девяностом дне просрочки.
            template = _collection_template(dpd, days_to_due)
        else:
            template = _template_for(campaign, persona, day, owned_families, rng)

            # Подходящего текста нет — банк не пишет вовсе.
            if template is None:
                continue

        hour = int(time_rng.choice(list(SEND_HOURS), p=list(SEND_WEIGHTS)))

        ts = day.replace(
            hour=hour,
            minute=int(time_rng.integers(0, 60)),
            second=0,
            microsecond=0,
        )

        delivered = _delivered(channel, persona, app_adopted, rng)

        clicked = delivered and _clicked(persona, campaign, channel, day, stress, rng)

        offer_id = (
            f"off_{stable_hash('offer', persona.client_id, ordinal, index) % 10 ** 12:012d}"
            if campaign.purpose == "offer" and campaign.family
            else None
        )

        result.append(
            Contact(
                ts=ts,
                campaign_code=code,
                purpose=campaign.purpose,
                channel=channel,
                template=template,
                product_family=campaign.family,
                delivered=delivered,
                clicked=clicked,
                offer_id=offer_id,
            )
        )

    result.sort(key=lambda item: item.ts)

    return tuple(result)


def campaign_for_family(family: str) -> str:

    pool = CAMPAIGNS_BY_FAMILY.get(family)

    if pool:
        return pool[0].code

    return "SERVICE"


__all__ = ["Contact", "campaign_for_family", "contacts_for_day", "daily_rate"]
