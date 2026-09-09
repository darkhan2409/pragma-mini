from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..world import BROWSE_SCREENS, SCREEN_HOME, SCREEN_OFFERS
from .config import (
    OWNED_BANNER_FACTOR,
    SUPPORT_ADOPTED_FACTOR,
    SUPPORT_FOREIGN_FACTOR,
    UNFINISHED_FACTOR,
)

if TYPE_CHECKING:  # pragma: no cover
    from .context import ContextView
    from .habits import AppHabits


# ============================================================
# ИДЕЯ
# ============================================================
#
# В v1 экран внутри домена выбирался равномерно, а операция
# не зависела от экрана: последовательность не значила ничего.
#
# В v2 у сессии есть намерение. Сценарий это маленький автомат
# над УЖЕ СУЩЕСТВУЮЩИМИ экранами: ни одного нового имени.
# Переходы описаны данными, поэтому по ним можно статически
# проверить, что весь каталог достижим.
#
# Пустая строка в переходах значит конец сессии.
# ============================================================


BALANCE_CHECK = "balance_check"
TRANSFER = "transfer"
PAYMENT = "payment"
CARD_MANAGEMENT = "card_management"
PRODUCT_EXPLORE = "product_explore"
SUPPORT = "support"
MARKET = "market"
PROFILE_SETTINGS = "profile_settings"

SCENARIOS: tuple[str, ...] = (
    BALANCE_CHECK,
    TRANSFER,
    PAYMENT,
    CARD_MANAGEMENT,
    PRODUCT_EXPLORE,
    SUPPORT,
    MARKET,
    PROFILE_SETTINGS,
)


@dataclass(frozen=True)
class Step:
    """
    Шаг сценария: экран, иногда операция, куда идти дальше.

    on_success / on_cancel / retry_to описывают, что делает
    клиент после исхода операции.
    """

    screen: str | None = None
    domain: str = "home"
    operation: str | None = None
    nexts: tuple[tuple[str, float], ...] = ()
    on_success: str = ""
    on_cancel: str = ""
    retry_to: str = ""
    depth: str | None = None
    tag: str = ""


Scenario = dict[str, Step]


# ------------------------------------------------------------
# ПРОВЕРКА БАЛАНСА
# ------------------------------------------------------------

BALANCE_STEPS: Scenario = {
    "start": Step(SCREEN_HOME, "home", nexts=(("balance", 0.6), ("history", 0.4))),
    "balance": Step(
        "s_010_balance", "home",
        nexts=(("history", 0.35), ("card_peek", 0.15), ("", 0.50)),
    ),
    "history": Step(
        "s_011_history", "home",
        nexts=(("balance", 0.20), ("", 0.80)),
    ),
    "card_peek": Step(
        "s_101_card_detail", "cards", operation="card_view",
        on_success="", on_cancel="", retry_to="card_peek",
        nexts=(("", 1.0),),
    ),
}


# ------------------------------------------------------------
# ПЕРЕВОД
# ------------------------------------------------------------

TRANSFER_STEPS: Scenario = {
    "start": Step(SCREEN_HOME, "home", nexts=(("root", 1.0),)),
    "root": Step(
        "s_200_transfers", "transfers",
        nexts=(
            ("phone", 0.40),
            ("card", 0.22),
            ("own", 0.16),
            ("template", 0.14),
            ("abroad", 0.05),
            ("", 0.03),
        ),
    ),
    "phone": Step("s_201_transfer_phone", "transfers", nexts=(("confirm_phone", 1.0),), tag="form"),
    "template": Step("s_201_transfer_phone", "transfers", nexts=(("confirm_template", 1.0),), tag="form"),
    "card": Step("s_202_transfer_card", "transfers", nexts=(("confirm_card", 1.0),), tag="form"),
    "own": Step("s_202_transfer_card", "transfers", nexts=(("confirm_own", 1.0),), tag="form"),
    "abroad": Step("s_202_transfer_card", "transfers", nexts=(("confirm_abroad", 1.0),), tag="form"),
    "confirm_phone": Step(
        "s_203_transfer_confirm", "transfers", operation="transfer_phone",
        on_success="result", on_cancel="root", retry_to="phone", tag="confirm",
    ),
    "confirm_template": Step(
        "s_203_transfer_confirm", "transfers", operation="transfer_template",
        on_success="result", on_cancel="root", retry_to="template", tag="confirm",
    ),
    "confirm_card": Step(
        "s_203_transfer_confirm", "transfers", operation="transfer_card",
        on_success="result", on_cancel="root", retry_to="card", tag="confirm",
    ),
    "confirm_own": Step(
        "s_203_transfer_confirm", "transfers", operation="transfer_own",
        on_success="result", on_cancel="root", retry_to="own", tag="confirm",
    ),
    "confirm_abroad": Step(
        "s_203_transfer_confirm", "transfers", operation="transfer_abroad",
        on_success="result", on_cancel="root", retry_to="abroad", tag="confirm",
    ),
    "result": Step("s_011_history", "home", nexts=(("", 1.0),)),
}


# ------------------------------------------------------------
# ОПЛАТА
# ------------------------------------------------------------

PAYMENT_STEPS: Scenario = {
    "start": Step(SCREEN_HOME, "home", nexts=(("root", 1.0),)),
    "root": Step(
        "s_300_payments", "payments",
        nexts=(
            ("utility", 0.34),
            ("mobile", 0.26),
            ("internet", 0.12),
            ("fine", 0.10),
            ("tax", 0.06),
            ("qr", 0.09),
            ("", 0.03),
        ),
    ),
    "utility": Step("s_301_payment_utility", "payments", nexts=(("confirm_utility", 1.0),), tag="form"),
    "internet": Step("s_301_payment_utility", "payments", nexts=(("confirm_internet", 1.0),), tag="form"),
    "mobile": Step("s_302_payment_mobile", "payments", nexts=(("confirm_mobile", 1.0),), tag="form"),
    "fine": Step("s_303_payment_fine", "payments", nexts=(("confirm_fine", 1.0),), tag="form"),
    "tax": Step("s_303_payment_fine", "payments", nexts=(("confirm_tax", 1.0),), tag="form"),
    "qr": Step("s_300_payments", "payments", nexts=(("confirm_qr", 1.0),), tag="form"),
    "confirm_utility": Step(
        "s_301_payment_utility", "payments", operation="pay_utility",
        on_success="result", on_cancel="root", retry_to="utility", tag="confirm",
    ),
    "confirm_internet": Step(
        "s_301_payment_utility", "payments", operation="pay_internet",
        on_success="result", on_cancel="root", retry_to="internet", tag="confirm",
    ),
    "confirm_mobile": Step(
        "s_302_payment_mobile", "payments", operation="pay_mobile",
        on_success="result", on_cancel="root", retry_to="mobile", tag="confirm",
    ),
    "confirm_fine": Step(
        "s_303_payment_fine", "payments", operation="pay_fine",
        on_success="result", on_cancel="root", retry_to="fine", tag="confirm",
    ),
    "confirm_tax": Step(
        "s_303_payment_fine", "payments", operation="pay_tax",
        on_success="result", on_cancel="root", retry_to="tax", tag="confirm",
    ),
    "confirm_qr": Step(
        "s_300_payments", "payments", operation="pay_qr",
        on_success="result", on_cancel="root", retry_to="qr", tag="confirm",
    ),
    "result": Step("s_011_history", "home", nexts=(("", 1.0),)),
}

# Шаг оплаты -> вид счёта, который он гасит. Ключи есть и у
# формы, и у подтверждения: намерение определяется на шаге
# подтверждения, а ветка выбирается на форме.
# "qr" сюда не входит: это покупка, а не счёт.
BRANCH_BILL_KIND: dict[str, str] = {
    "utility": "utility",
    "internet": "internet",
    "mobile": "mobile",
    "fine": "fine",
    "tax": "tax",
    "confirm_utility": "utility",
    "confirm_internet": "internet",
    "confirm_mobile": "mobile",
    "confirm_fine": "fine",
    "confirm_tax": "tax",
}

# Какая ветка оплаты соответствует счёту.
BILL_KIND_BRANCH: dict[str, str] = {
    "utility": "utility",
    "mobile": "mobile",
    "internet": "internet",
    "fine": "fine",
    "tax": "tax",
    "service": "tax",
}


# ------------------------------------------------------------
# УПРАВЛЕНИЕ КАРТОЙ
# ------------------------------------------------------------

CARD_STEPS: Scenario = {
    "start": Step(SCREEN_HOME, "home", nexts=(("root", 1.0),)),
    "root": Step(
        "s_100_cards", "cards",
        nexts=(
            ("detail", 0.40),
            ("limits", 0.16),
            ("pin", 0.12),
            ("block", 0.10),
            ("unblock", 0.10),
            ("order", 0.08),
            ("", 0.04),
        ),
    ),
    "detail": Step(
        "s_101_card_detail", "cards", operation="card_view",
        on_success="", on_cancel="root", retry_to="detail", nexts=(("", 1.0),),
    ),
    "limits": Step(
        "s_102_card_limits", "cards", operation="limit_change",
        on_success="detail_result", on_cancel="root", retry_to="limits", tag="confirm",
    ),
    "pin": Step(
        "s_103_card_pin", "cards", operation="pin_reset",
        on_success="detail_result", on_cancel="root", retry_to="pin", tag="confirm",
    ),
    "block": Step(
        "s_101_card_detail", "cards", operation="card_block",
        on_success="detail_result", on_cancel="root", retry_to="block", tag="confirm",
    ),
    "unblock": Step(
        "s_101_card_detail", "cards", operation="card_unblock",
        on_success="detail_result", on_cancel="root", retry_to="unblock", tag="confirm",
    ),
    "order": Step(
        "s_100_cards", "cards", operation="card_order",
        on_success="detail_result", on_cancel="root", retry_to="order", tag="confirm",
    ),
    "detail_result": Step("s_101_card_detail", "cards", nexts=(("", 1.0),)),
}


# ------------------------------------------------------------
# ИЗУЧЕНИЕ ПРОДУКТА
# ------------------------------------------------------------
#
# Глубина изучения это то, что потом превращается в заявку:
# корень раздела значит меньше, чем условия договора.
# ------------------------------------------------------------

EXPLORE_STEPS: dict[str, Scenario] = {
    "cash_loan": {
        "start": Step(SCREEN_HOME, "home", nexts=(("offers", 0.35), ("root", 0.65))),
        "offers": Step(SCREEN_OFFERS, "home", nexts=(("root", 1.0),)),
        "root": Step(
            "s_400_loans", "loans", depth="root",
            nexts=(("calc", 0.45), ("view", 0.20), ("schedule", 0.10), ("", 0.25)),
        ),
        "view": Step(
            "s_400_loans", "loans", operation="loan_view", depth="root",
            on_success="calc", on_cancel="", retry_to="view", nexts=(("calc", 1.0),),
        ),
        "schedule": Step(
            "s_400_loans", "loans", operation="loan_schedule", depth="root",
            on_success="statement", on_cancel="", retry_to="schedule", nexts=(("statement", 1.0),),
        ),
        "statement": Step(
            "s_400_loans", "loans", operation="loan_statement", depth="root",
            on_success="early", on_cancel="", retry_to="statement", nexts=(("early", 1.0),),
        ),
        "early": Step(
            "s_402_loan_terms", "loans", operation="loan_early_repay", depth="terms",
            on_success="", on_cancel="", retry_to="early", nexts=(("", 1.0),),
        ),
        "calc": Step(
            "s_401_loan_calc", "loans", operation="loan_calc", depth="calc",
            on_success="terms", on_cancel="", retry_to="calc",
            nexts=(("terms", 0.55), ("", 0.45)),
        ),
        "terms": Step("s_402_loan_terms", "loans", depth="terms", nexts=(("", 1.0),)),
    },
    "deposit": {
        "start": Step(SCREEN_HOME, "home", nexts=(("offers", 0.30), ("root", 0.70))),
        "offers": Step(SCREEN_OFFERS, "home", nexts=(("root", 1.0),)),
        "root": Step(
            "s_500_deposits", "deposits", depth="root",
            nexts=(("calc", 0.40), ("view", 0.25), ("topup", 0.15), ("close", 0.05), ("", 0.15)),
        ),
        "view": Step(
            "s_500_deposits", "deposits", operation="deposit_view", depth="root",
            on_success="calc", on_cancel="", retry_to="view", nexts=(("calc", 1.0),),
        ),
        "topup": Step(
            "s_500_deposits", "deposits", operation="deposit_topup", depth="root",
            on_success="", on_cancel="", retry_to="topup", nexts=(("", 1.0),), tag="confirm",
        ),
        "close": Step(
            "s_502_deposit_terms", "deposits", operation="deposit_close", depth="terms",
            on_success="", on_cancel="", retry_to="close", nexts=(("", 1.0),), tag="confirm",
        ),
        "calc": Step(
            "s_501_deposit_calc", "deposits", depth="calc",
            nexts=(("terms", 0.60), ("", 0.40)),
        ),
        "terms": Step(
            "s_502_deposit_terms", "deposits", depth="terms",
            nexts=(("open", 0.35), ("", 0.65)),
        ),
        "open": Step(
            "s_502_deposit_terms", "deposits", operation="deposit_open", depth="terms",
            on_success="", on_cancel="terms", retry_to="open", nexts=(("", 1.0),), tag="confirm",
        ),
    },
    "insurance": {
        "start": Step(SCREEN_HOME, "home", nexts=(("offers", 0.30), ("root", 0.70))),
        "offers": Step(SCREEN_OFFERS, "home", nexts=(("root", 1.0),)),
        "root": Step(
            "s_600_insurance", "insurance", depth="root",
            nexts=(("terms", 0.55), ("", 0.45)),
        ),
        "terms": Step("s_601_insurance_terms", "insurance", depth="terms", nexts=(("", 1.0),)),
    },
    "credit_card": {
        "start": Step(SCREEN_HOME, "home", nexts=(("offers", 0.35), ("root", 0.65))),
        "offers": Step(SCREEN_OFFERS, "home", nexts=(("root", 1.0),)),
        "root": Step(
            "s_100_cards", "cards", depth="root",
            nexts=(("detail", 0.55), ("", 0.45)),
        ),
        "detail": Step(
            "s_101_card_detail", "cards", depth="calc",
            nexts=(("limits", 0.55), ("", 0.45)),
        ),
        "limits": Step("s_102_card_limits", "cards", depth="terms", nexts=(("", 1.0),)),
    },
}

EXPLORE_PRODUCTS: tuple[str, ...] = tuple(EXPLORE_STEPS)

# Псевдонимов операций нет: каждая операция сценария существует
# в каталоге v1 под своим именем.
DEPOSIT_CALC_ALIAS: dict[str, str] = {}


# ------------------------------------------------------------
# ПОДДЕРЖКА
# ------------------------------------------------------------

SUPPORT_STEPS: Scenario = {
    "start": Step(SCREEN_HOME, "home", nexts=(("root", 1.0),)),
    "root": Step(
        "s_900_support", "support",
        nexts=(("chat", 0.55), ("faq", 0.35), ("callback", 0.10)),
    ),
    "faq": Step("s_902_faq", "support", nexts=(("chat", 0.35), ("", 0.65))),
    "chat": Step(
        "s_901_chat", "support", operation="chat_open",
        on_success="chat_done", on_cancel="", retry_to="chat", nexts=(("chat_done", 1.0),),
    ),
    "chat_done": Step(
        "s_901_chat", "support",
        nexts=(("complaint", 0.20), ("", 0.80)),
    ),
    "complaint": Step(
        "s_901_chat", "support", operation="complaint",
        on_success="", on_cancel="", retry_to="complaint", nexts=(("", 1.0),),
    ),
    "callback": Step(
        "s_900_support", "support", operation="callback_request",
        on_success="", on_cancel="", retry_to="callback", nexts=(("", 1.0),),
    ),
}


# ------------------------------------------------------------
# МАРКЕТ
# ------------------------------------------------------------

MARKET_STEPS: Scenario = {
    "start": Step(SCREEN_HOME, "home", nexts=(("root", 1.0),)),
    "root": Step(
        "s_700_market", "market", operation="market_browse",
        on_success="item", on_cancel="", retry_to="root", nexts=(("item", 1.0),),
    ),
    "item": Step(
        "s_701_market_item", "market",
        nexts=(("cart", 0.55), ("returns", 0.05), ("", 0.40)),
    ),
    "cart": Step(
        "s_702_market_cart", "market", operation="market_order",
        on_success="", on_cancel="item", retry_to="cart", nexts=(("", 1.0),), tag="confirm",
    ),
    "returns": Step(
        "s_702_market_cart", "market", operation="market_return",
        on_success="", on_cancel="item", retry_to="returns", nexts=(("", 1.0),),
    ),
}


# ------------------------------------------------------------
# ПРОФИЛЬ И НАСТРОЙКИ
# ------------------------------------------------------------
#
# Отдельный сценарий, а не случайная вставка в чужую сессию:
# в середину оплаты или заявки настройки не вклиниваются.
# ------------------------------------------------------------

PROFILE_STEPS: Scenario = {
    "start": Step(SCREEN_HOME, "home", nexts=(("root", 1.0),)),
    "root": Step(
        "s_800_profile", "profile",
        nexts=(("settings", 0.55), ("documents", 0.30), ("", 0.15)),
    ),
    "documents": Step("s_802_documents", "profile", nexts=(("settings", 0.30), ("", 0.70))),
    "settings": Step(
        "s_801_settings", "profile",
        nexts=(("pin", 0.30), ("device", 0.25), ("signout", 0.15), ("", 0.30)),
    ),
    "pin": Step(
        "s_801_settings", "auth", operation="pin_change",
        on_success="", on_cancel="settings", retry_to="pin", nexts=(("", 1.0),), tag="confirm",
    ),
    "device": Step(
        "s_801_settings", "auth", operation="device_bind",
        on_success="", on_cancel="settings", retry_to="device", nexts=(("", 1.0),), tag="confirm",
    ),
    "signout": Step(
        "s_800_profile", "auth", operation="logout",
        on_success="", on_cancel="", retry_to="signout", nexts=(("", 1.0),),
    ),
}


SCENARIO_STEPS: dict[str, Scenario] = {
    BALANCE_CHECK: BALANCE_STEPS,
    TRANSFER: TRANSFER_STEPS,
    PAYMENT: PAYMENT_STEPS,
    CARD_MANAGEMENT: CARD_STEPS,
    SUPPORT: SUPPORT_STEPS,
    MARKET: MARKET_STEPS,
    PROFILE_SETTINGS: PROFILE_STEPS,
}


# Куда клиент попадает, продолжая сессию вторым намерением:
# домашний экран он уже видел.
SCENARIO_ENTRY: dict[str, str] = {
    BALANCE_CHECK: "balance",
    TRANSFER: "root",
    PAYMENT: "root",
    CARD_MANAGEMENT: "root",
    PRODUCT_EXPLORE: "root",
    SUPPORT: "root",
    MARKET: "root",
    PROFILE_SETTINGS: "root",
}


def steps_for(scenario: str, target: str | None) -> Scenario:

    if scenario == PRODUCT_EXPLORE:
        return EXPLORE_STEPS[target or "cash_loan"]

    return SCENARIO_STEPS[scenario]


# ============================================================
# ВЕСА СЦЕНАРИЕВ
# ============================================================


def scenario_weights(
    app: "AppHabits",
    view: "ContextView",
    owned: frozenset[str],
    adopted: tuple[str, ...],
) -> dict[str, float]:
    """
    Чего клиент хочет от этой сессии.

    Зависит от устойчивых склонностей, доступных продуктов
    и того, что происходило РАНЬШЕ момента запроса.
    """

    bias = dict(zip(SCENARIOS, app.scenario_bias))

    weights = {scenario: 0.0 for scenario in SCENARIOS}

    weights[BALANCE_CHECK] = 3.0 * bias[BALANCE_CHECK]
    weights[PROFILE_SETTINGS] = 0.30 * bias[PROFILE_SETTINGS]

    if "transfers" in adopted:
        weights[TRANSFER] = 2.0 * bias[TRANSFER]

    if "payments" in adopted:

        # Без счёта в раздел оплаты заходят заметно реже: там
        # нечего делать, кроме разового платежа.
        due = len(view.due_bills)

        weights[PAYMENT] = 1.8 * bias[PAYMENT] * (0.15 + 2.0 * min(due, 3))

    if "cards" in adopted and ({"debit_card", "credit_card"} & owned):
        weights[CARD_MANAGEMENT] = 1.0 * bias[CARD_MANAGEMENT]
        if view.card_blocked:
            weights[CARD_MANAGEMENT] *= 4.0

    if "market" in adopted:
        weights[MARKET] = 1.0 * bias[MARKET]

    support_factor = (
        SUPPORT_ADOPTED_FACTOR if "support" in adopted else SUPPORT_FOREIGN_FACTOR
    )

    weights[SUPPORT] = (
        0.25
        * bias[SUPPORT]
        * app.support_bias
        * support_factor
        * (1.0 + 4.0 * min(len(view.recent_failures), 3))
    )

    explore = explore_target_weights(app, view, owned, adopted)

    if explore:
        weights[PRODUCT_EXPLORE] = sum(explore.values()) * bias[PRODUCT_EXPLORE]

    if view.unfinished in weights and weights[view.unfinished] > 0.0:
        weights[view.unfinished] *= UNFINISHED_FACTOR

    return weights


EXPLORE_DOMAIN: dict[str, str] = {
    "cash_loan": "loans",
    "deposit": "deposits",
    "insurance": "insurance",
    "credit_card": "cards",
}


def explore_target_weights(
    app: "AppHabits",
    view: "ContextView",
    owned: frozenset[str],
    adopted: tuple[str, ...],
) -> dict[str, float]:
    """
    Какой продукт клиент пойдёт изучать.
    """

    result: dict[str, float] = {}

    base = {
        "cash_loan": 0.5 + 2.0 * view.credit_need,
        "deposit": 0.4 + 1.0 * (1.0 - view.credit_need),
        "insurance": 0.25,
        "credit_card": 0.4 + 1.2 * view.credit_need,
    }

    for product, weight in base.items():

        domain = EXPLORE_DOMAIN[product]

        if domain not in adopted:
            continue

        value = weight * app.explore_bias

        if product in owned:
            # Свой продукт тоже смотрят, но реже и не за офферами.
            value *= 0.35
        elif product in view.recent_offers:
            value *= 1.8

        if product in view.recent_rejections:
            value *= 0.4

        if value > 0.0:
            result[product] = value

    return result


# ============================================================
# ДОСТИЖИМОСТЬ КАТАЛОГА
# ============================================================
#
# Тест проверяет по этим функциям, что v2 не сузил словарь:
# каждый экран и каждая операция v1 остаются достижимы.
# ============================================================


def _walk(scenario: Scenario) -> tuple[set[str], set[str]]:

    screens: set[str] = set()
    operations: set[str] = set()

    for step in scenario.values():

        if step.screen is not None:
            screens.add(step.screen)

        if step.operation is not None:
            operations.add(DEPOSIT_CALC_ALIAS.get(step.operation, step.operation))

    return screens, operations


def reachable_screens() -> frozenset[str]:

    screens: set[str] = set()

    for scenario in SCENARIO_STEPS.values():
        screens |= _walk(scenario)[0]

    for scenario in EXPLORE_STEPS.values():
        screens |= _walk(scenario)[0]

    return frozenset(screens)


def reachable_operations() -> frozenset[str]:

    operations: set[str] = {"login", "biometry_login"}

    for scenario in SCENARIO_STEPS.values():
        operations |= _walk(scenario)[1]

    for scenario in EXPLORE_STEPS.values():
        operations |= _walk(scenario)[1]

    return frozenset(operations)


def catalog_screens() -> frozenset[str]:

    screens = {SCREEN_OFFERS}

    for names in BROWSE_SCREENS.values():
        screens |= set(names)

    return frozenset(screens)


__all__ = [
    "BILL_KIND_BRANCH",
    "BRANCH_BILL_KIND",
    "CARD_MANAGEMENT",
    "EXPLORE_DOMAIN",
    "EXPLORE_PRODUCTS",
    "PAYMENT",
    "PRODUCT_EXPLORE",
    "PROFILE_SETTINGS",
    "SCENARIOS",
    "SCENARIO_ENTRY",
    "SCENARIO_STEPS",
    "SUPPORT_STEPS",
    "SUPPORT",
    "Step",
    "TRANSFER",
    "BALANCE_CHECK",
    "MARKET",
    "catalog_screens",
    "explore_target_weights",
    "reachable_operations",
    "reachable_screens",
    "scenario_weights",
    "steps_for",
]
