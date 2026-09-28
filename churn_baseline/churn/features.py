from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .config import LOCAL_OFFSET, WINDOWS


# ============================================================
# ПРИЗНАКИ КЛИЕНТА НА CUTOFF T
# ============================================================
#
# Всё считается только по событиям строго раньше T: блок событий
# обрезается в самом начале, и ни одна функция ниже не видит момента T
# и позже. Окно «за w дней» — события с T - w <= t < T.
#
# Имя признака и его описание рождаются в одном месте (Table.put),
# поэтому список признаков в отчёте не расходится с кодом.
# ============================================================


DAY = pd.Timedelta(days=1)

LOGIN_OPERATIONS = ("login", "biometry_login")
INCOME_TYPES = ("salary_credit", "pension_credit", "other_income_credit")
WINBACK_PREFIX = "WB_"

# Виды денежных операций клиента: (имя, условие).
FLOWS: tuple[tuple[str, str], ...] = (
    ("purchase", "покупки клиента (purchase, reason=purchase)"),
    ("transfer", "исходящие переводы клиента (transfer_out, p2p_out)"),
    ("cash_out", "снятия наличных (cash_withdrawal)"),
    ("cash_in", "взносы наличных (cash_deposit)"),
)


class Table:
    """
    Признаки одного блока клиентов: колонки и их описания.
    """

    def __init__(self, clients: pd.Index) -> None:
        self.clients = clients
        self.columns: dict[str, np.ndarray] = {}
        self.descriptions: dict[str, str] = {}

    def put(self, name: str, values, description: str) -> None:
        if name in self.columns:
            raise ValueError(f"признак {name} уже есть")
        self.columns[name] = np.asarray(values, dtype=float)
        self.descriptions[name] = description

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.columns, index=self.clients)


class Past:
    """
    События блока строго раньше T и счётчики по клиентам.
    """

    def __init__(self, events: pd.DataFrame, clients: pd.Index, cutoff: datetime, action: np.ndarray) -> None:
        keep = (events["t"] < cutoff).to_numpy()
        self.events = events[keep].reset_index(drop=True)
        self.action = action[keep]
        self.n = len(clients)
        self.code = clients.get_indexer(self.events["client_id"])
        # Давность события от T в сутках: > 0 у всех событий прошлого.
        self.age = ((cutoff - self.events["t"]) / DAY).to_numpy(dtype=float)
        local = self.events["t"] + LOCAL_OFFSET
        self.day = ((local - pd.Timestamp("1970-01-01", tz="UTC")) // DAY).to_numpy(dtype=np.int64)
        self.cutoff_day = int((pd.Timestamp(cutoff) + LOCAL_OFFSET - pd.Timestamp("1970-01-01", tz="UTC")) // DAY)

    def col(self, name: str) -> pd.Series:
        return self.events[name]

    def within(self, mask: np.ndarray, days: float | None) -> np.ndarray:
        return mask if days is None else mask & (self.age <= days)

    def count(self, mask: np.ndarray, days: float | None = None) -> np.ndarray:
        mask = self.within(mask, days)
        return np.bincount(self.code[mask], minlength=self.n).astype(float)

    def total(self, mask: np.ndarray, values: pd.Series, days: float | None = None) -> np.ndarray:
        mask = self.within(mask, days) & values.notna().to_numpy()
        weights = values.to_numpy(dtype=float, na_value=np.nan)[mask]
        return np.bincount(self.code[mask], weights=weights, minlength=self.n)

    def since_last(self, mask: np.ndarray) -> np.ndarray:
        return self._reduce(mask, self.age, "min")

    def since_first(self, mask: np.ndarray) -> np.ndarray:
        return self._reduce(mask, self.age, "max")

    def maximum(self, mask: np.ndarray, values: pd.Series, days: float | None = None) -> np.ndarray:
        mask = self.within(mask, days) & values.notna().to_numpy()
        return self._reduce(mask, values.to_numpy(dtype=float, na_value=np.nan), "max")

    def nunique(self, mask: np.ndarray, values: pd.Series, days: float | None = None) -> np.ndarray:
        mask = self.within(mask, days) & values.notna().to_numpy()
        frame = pd.DataFrame({"code": self.code[mask], "value": values.to_numpy()[mask]})
        counts = frame.drop_duplicates().groupby("code").size()
        return counts.reindex(range(self.n), fill_value=0).to_numpy(dtype=float)

    def last_value(self, mask: np.ndarray, values: pd.Series, before_days: float = 0.0) -> np.ndarray:
        """
        Последнее значение по времени среди событий старше before_days суток.
        События внутри клиента уже упорядочены по времени.
        """
        mask = mask & (self.age > before_days) & values.notna().to_numpy()
        frame = pd.DataFrame({"code": self.code[mask], "value": values.to_numpy(dtype=float, na_value=np.nan)[mask]})
        last = frame.groupby("code")["value"].last()
        return last.reindex(range(self.n)).to_numpy(dtype=float)

    def _reduce(self, mask: np.ndarray, values: np.ndarray, how: str) -> np.ndarray:
        series = pd.Series(values[mask]).groupby(self.code[mask]).agg(how)
        return series.reindex(range(self.n)).to_numpy(dtype=float)


def _share(part: np.ndarray, whole: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(whole > 0, part / np.where(whole > 0, whole, 1.0), np.nan)


def compute(events: pd.DataFrame, cutoff: datetime, action: np.ndarray) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Признаки клиентов блока на T по событиям строго раньше T.

    events — события целых клиентов (client_blocks), action — маска
    действий клиента той же длины. Строка результата на каждого клиента
    блока, индекс client_id.
    """
    clients = pd.Index(pd.unique(events["client_id"]), name="client_id")
    past = Past(events, clients, cutoff, action)
    table = Table(clients)

    _activity(table, past)
    _app(table, past)
    _flows(table, past)
    _money(table, past)
    _products(table, past)
    _bank(table, past)
    return table.frame(), table.descriptions


def _activity(table: Table, past: Past) -> None:
    act = past.action
    table.put("act_days_since_last", past.since_last(act), "дней от последнего действия клиента до T")
    table.put("act_days_since_first", past.since_first(act), "дней от первого действия клиента в выгрузке до T")
    table.put("act_count_all", past.count(act), "действий клиента за всю историю до T")
    for days in WINDOWS:
        table.put(f"act_count_{days}", past.count(act, days), f"действий клиента за {days} дней до T")

    frame = pd.DataFrame({"code": past.code[act], "day": past.day[act]}).drop_duplicates()
    for days in WINDOWS:
        recent = frame[frame["day"] >= past.cutoff_day - days]
        counts = recent.groupby("code").size().reindex(range(past.n), fill_value=0)
        table.put(f"act_days_{days}", counts.to_numpy(), f"дней с действиями клиента за {days} дней до T")

    table.put(
        "act_trend_30_90",
        _share(table.columns["act_count_30"], table.columns["act_count_90"]),
        "доля действий за 30 дней среди действий за 90 дней",
    )

    recent = frame[frame["day"] >= past.cutoff_day - 90].sort_values(["code", "day"])
    gaps = recent.groupby("code")["day"].diff().dropna()
    gap_codes = recent.loc[gaps.index, "code"]
    table.put(
        "act_gap_mean_90",
        gaps.groupby(gap_codes).mean().reindex(range(past.n)).to_numpy(),
        "средний разрыв в днях между соседними днями с действиями за 90 дней",
    )
    table.put(
        "act_gap_max_90",
        gaps.groupby(gap_codes).max().reindex(range(past.n)).to_numpy(),
        "наибольший разрыв в днях между соседними днями с действиями за 90 дней",
    )


def _app(table: Table, past: Past) -> None:
    kind = past.col("type")
    operation = past.col("operation")
    login = ((kind == "app_operation") & operation.isin(LOGIN_OPERATIONS)).to_numpy()
    failed = login & (past.col("status") == "failed").to_numpy()
    in_app = kind.isin(["app_operation", "app_screen"]).to_numpy()
    screen = (kind == "app_screen").to_numpy()
    other_ops = ((kind == "app_operation") & ~operation.isin(LOGIN_OPERATIONS)).to_numpy()
    click = (kind == "banner_clicked").to_numpy()

    for days in (7, 30, 90):
        table.put(f"app_login_count_{days}", past.count(login, days), f"входов в приложение за {days} дней")
    table.put("app_login_failed_30", past.count(failed, 30), "неудачных входов в приложение за 30 дней")
    for days in (7, 30, 90):
        table.put(
            f"app_sessions_{days}",
            past.nunique(in_app, past.col("session_id"), days),
            f"различных сессий приложения за {days} дней",
        )
    table.put("app_screen_count_30", past.count(screen, 30), "просмотров экранов приложения за 30 дней")
    table.put("app_operation_count_30", past.count(other_ops, 30), "операций в приложении кроме входа за 30 дней")
    table.put("banner_click_count_90", past.count(click, 90), "кликов по баннерам за 90 дней")
    table.put("app_days_since_last", past.since_last(in_app), "дней от последнего события приложения до T")


def _flow_masks(past: Past) -> dict[str, np.ndarray]:
    kind = past.col("type")
    return {
        "purchase": ((kind == "purchase") & (past.col("reason") == "purchase")).to_numpy(),
        "transfer": kind.isin(["transfer_out", "p2p_out"]).to_numpy(),
        "cash_out": (kind == "cash_withdrawal").to_numpy(),
        "cash_in": (kind == "cash_deposit").to_numpy(),
    }


def _flows(table: Table, past: Past) -> None:
    amount = past.col("amount")
    masks = _flow_masks(past)
    for name, title in FLOWS:
        mask = masks[name]
        for days in (7, 30, 90):
            table.put(f"{name}_count_{days}", past.count(mask, days), f"{title}: число за {days} дней")
        for days in (30, 90):
            table.put(f"{name}_sum_{days}", past.total(mask, amount, days), f"{title}: сумма KZT за {days} дней")
        table.put(f"{name}_days_since_last", past.since_last(mask), f"{title}: дней от последней до T")

    purchase = masks["purchase"]
    count_90 = table.columns["purchase_count_90"]
    table.put("purchase_mean_90", _share(table.columns["purchase_sum_90"], count_90), "средняя покупка KZT за 90 дней")
    declined = purchase & (past.col("status") == "declined").to_numpy()
    table.put("purchase_declined_share_90", _share(past.count(declined, 90), count_90), "доля отклонённых покупок за 90 дней")
    online = purchase & past.col("is_online").fillna(False).to_numpy(dtype=bool)
    table.put("purchase_online_share_90", _share(past.count(online, 90), count_90), "доля онлайн-покупок за 90 дней")
    table.put("purchase_mcc_nunique_90", past.nunique(purchase, past.col("mcc"), 90), "различных MCC покупок за 90 дней")
    table.put(
        "purchase_merchant_nunique_90",
        past.nunique(purchase, past.col("merchant_name"), 90),
        "различных названий мерчантов покупок за 90 дней",
    )

    kind = past.col("type")
    channel = past.col("channel")
    system = (channel == "system").to_numpy()
    bill = (kind == "bill_payment").to_numpy()
    loan = (kind == "loan_payment").to_numpy()
    subscription = ((kind == "purchase") & (past.col("reason") == "subscription")).to_numpy()
    table.put("bill_manual_count_90", past.count(bill & ~system, 90), "оплат счетов самим клиентом за 90 дней")
    table.put("bill_auto_count_90", past.count(bill & system, 90), "автоплатежей по счетам за 90 дней")
    table.put("loan_pay_manual_count_90", past.count(loan & ~system, 90), "ручных платежей по кредиту за 90 дней")
    table.put("loan_pay_auto_count_90", past.count(loan & system, 90), "автосписаний по кредиту за 90 дней")
    table.put("subscription_count_90", past.count(subscription, 90), "списаний подписок за 90 дней")


def _money(table: Table, past: Past) -> None:
    kind = past.col("type")
    amount = past.col("amount")
    salary = (kind == "salary_credit").to_numpy()
    income = kind.isin(INCOME_TYPES).to_numpy()
    incoming = kind.isin(["transfer_in", "p2p_in"]).to_numpy()

    table.put("salary_count_90", past.count(salary, 90), "зачислений зарплаты за 90 дней")
    table.put("salary_sum_90", past.total(salary, amount, 90), "сумма зарплаты KZT за 90 дней")
    table.put("salary_days_since_last", past.since_last(salary), "дней от последней зарплаты до T")
    table.put("pension_count_90", past.count((kind == "pension_credit").to_numpy(), 90), "зачислений пенсии за 90 дней")
    table.put("income_sum_90", past.total(income, amount, 90), "сумма зарплаты, пенсии и прочих доходов KZT за 90 дней")
    table.put("transfer_in_count_90", past.count(incoming, 90), "входящих переводов за 90 дней")
    table.put("transfer_in_sum_90", past.total(incoming, amount, 90), "сумма входящих переводов KZT за 90 дней")

    balance = past.col("balance_after")
    has_balance = balance.notna().to_numpy()
    last = past.last_value(has_balance, balance)
    earlier = past.last_value(has_balance, balance, before_days=30)
    table.put("balance_last", last, "последний остаток balance_after до T (по любому счёту)")
    table.put("balance_30d_ago", earlier, "последний остаток balance_after раньше T - 30 дней")
    table.put("balance_change_30", last - earlier, "изменение остатка за последние 30 дней")

    table.put("fee_count_90", past.count((kind == "fee_charge").to_numpy(), 90), "списаний комиссий за 90 дней")
    short = (past.col("decline_reason") == "insufficient_funds").to_numpy()
    table.put("insufficient_funds_90", past.count(short, 90), "отказов по недостатку средств за 90 дней")


def _products(table: Table, past: Past) -> None:
    kind = past.col("type")
    source = past.col("source")
    every = np.ones(len(past.events), dtype=bool)
    loans = (source == "loans").to_numpy()

    table.put("contracts_nunique_90", past.nunique(every, past.col("contract_id"), 90), "различных договоров с событиями за 90 дней")
    table.put("cards_nunique_90", past.nunique(every, past.col("card_id"), 90), "различных карт с событиями за 90 дней")
    table.put("products_nunique_90", past.nunique(every, past.col("product_id"), 90), "различных продуктов с событиями за 90 дней")
    table.put("loan_contracts_90", past.nunique(loans, past.col("contract_id"), 90), "различных кредитов с событиями за 90 дней")
    table.put(
        "deposit_contracts_90",
        past.nunique((kind == "interest_credit").to_numpy(), past.col("contract_id"), 90),
        "различных договоров с начислением процентов (вклады) за 90 дней",
    )
    table.put(
        "insurance_premium_count",
        past.count(((kind == "purchase") & (past.col("reason") == "insurance_premium")).to_numpy()),
        "списаний страховых премий за всю историю до T",
    )

    submitted = (kind == "application_submitted").to_numpy()
    rejected = ((kind == "application_decision") & (past.col("decision") == "rejected")).to_numpy()
    opened = kind.isin(["product_opened", "account_opened"]).to_numpy()
    table.put("application_count_90", past.count(submitted, 90), "поданных заявок за 90 дней")
    table.put("application_rejected_90", past.count(rejected, 90), "отказов по заявкам за 90 дней")
    table.put("product_opened_90", past.count(opened, 90), "открытий продуктов и счетов за 90 дней")
    table.put("product_closed_90", past.count((kind == "product_closed").to_numpy(), 90), "закрытий продуктов за 90 дней")
    table.put("card_blocked_90", past.count((kind == "card_blocked").to_numpy(), 90), "блокировок карт за 90 дней")

    table.put("installment_missed_90", past.count((kind == "installment_missed").to_numpy(), 90), "пропущенных платежей по кредиту за 90 дней")
    table.put("delinquency_90", past.count((kind == "delinquency_registered").to_numpy(), 90), "регистраций просрочки за 90 дней")
    table.put("loan_dpd_max_90", past.maximum(loans, past.col("days_past_due"), 90), "наибольшие дни просрочки за 90 дней")


def _bank(table: Table, past: Past) -> None:
    kind = past.col("type")
    comm = (kind == "communication_sent").to_numpy()
    channel = past.col("channel")
    delivered = comm & past.col("delivered").fillna(False).to_numpy(dtype=bool)
    winback = comm & past.col("template").fillna("").str.startswith(WINBACK_PREFIX).to_numpy(dtype=bool)

    for days in (30, 90):
        table.put(f"comm_count_{days}", past.count(comm, days), f"коммуникаций банка за {days} дней")
    for name in ("push", "sms", "call", "email"):
        table.put(f"comm_{name}_90", past.count(comm & (channel == name).to_numpy(), 90), f"коммуникаций банка ({name}) за 90 дней")
    table.put(
        "comm_delivered_share_90",
        _share(past.count(delivered, 90), table.columns["comm_count_90"]),
        "доля доставленных коммуникаций за 90 дней",
    )
    table.put("comm_winback_90", past.count(winback, 90), "winback-коммуникаций (шаблоны WB_*) за 90 дней")

    consent = (
        (kind == "profile_change")
        & (past.col("field_name") == "consent_marketing")
        & (past.col("new_value") == "false")
    ).to_numpy()
    table.put("consent_withdrawn", (past.count(consent) > 0).astype(float), "клиент отозвал согласие на маркетинг раньше T")

    table.put("support_case_count_90", past.count((kind == "case_opened").to_numpy(), 90), "обращений в поддержку за 90 дней")
    table.put("fraud_alert_count_90", past.count((kind == "fraud_alert").to_numpy(), 90), "антифрод-алертов за 90 дней")
    every = np.ones(len(past.events), dtype=bool)
    table.put("events_count_90", past.count(every, 90), "всех событий ленты за 90 дней")
    table.put("bank_events_count_30", past.count(~past.action, 30), "событий не от клиента за 30 дней")
