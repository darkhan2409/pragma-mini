from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ELIGIBLE_FROM, LOCAL_OFFSET


# ============================================================
# ПРИЗНАКИ ОПЕРАЦИИ НА МОМЕНТ РЕШЕНИЯ
# ============================================================
#
# Строка — операция клиента: покупка (reason=purchase) или transfer_out.
# Момент решения — время операции. Модели доступны:
#
#   сама операция  то, что известно при авторизации: тип, канал, сумма,
#                  валюта, мерчант, MCC, страна, время суток;
#   история        события того же клиента СТРОГО РАНЬШЕ операции: окно
#                  «за w» — t_i - w <= t < t_i. Событие в тот же момент в
#                  историю не входит: порядок внутри момента банк не знает;
#   анкета         на момент операции: откат по profile_change.
#
# Не доступны и не используются:
#   исход самой операции   status, decline_reason, balance_after;
#   всё, что после неё     алерты, решения антифрода, блокировки,
#                          обращения, chargeback, перевыпуск;
#   следы генератора       есть ли у строки transfer_id, card_id,
#                          account_id, session_id; порядок ключей payload;
#                          секунды и минуты времени (шаги мошенника стоят
#                          на целых секундах от целого часа); «круглость»
#                          суммы (суммы мошенника округлены до 100);
#   идентификаторы         counterparty как значение — только новизна
#                          получателя.
#
# Имена признаков и описания рождаются в одном месте (Table.put).
# ============================================================


SPAN = 10**13  # ключ события: номер клиента * SPAN + время в мс
EPOCH = pd.Timestamp(0, tz="UTC")
HOUR = 3_600_000
DAY = 24 * HOUR

LOGIN_OPERATIONS = ("login", "biometry_login")
INCOME_TYPES = ("salary_credit", "pension_credit", "other_income_credit")
# Операции, чей balance_after описывает повседневный счёт клиента.
EVERYDAY_TYPES = (
    "purchase",
    "transfer_out",
    "p2p_out",
    "cash_withdrawal",
    "cash_deposit",
    "bill_payment",
    "salary_credit",
    "pension_credit",
    "other_income_credit",
    "transfer_in",
    "p2p_in",
)

CATEGORICAL: tuple[str, ...] = (
    "channel",
    "merchant_country",
    "mcc",
    "merchant_category",
    "merchant_name",
    "gender",
    "region",
    "income_type",
)

PROFILE_FIELDS: tuple[str, ...] = ("region", "city", "income_type", "declared_income")
MILESTONES: tuple[str, ...] = ("bank_registered", "app_registered")

MISSING = "NA"


def eligible(events: pd.DataFrame) -> np.ndarray:
    """
    Строки датасета: покупки клиента и исходящие переводы не раньше
    ELIGIBLE_FROM. Подписки и страховые списания автоматические, мошенник
    их не делает.
    """
    kind = events["type"]
    operation = ((kind == "purchase") & (events["reason"] == "purchase")) | (kind == "transfer_out")
    return (operation & (events["t"] >= ELIGIBLE_FROM)).to_numpy(dtype=bool)


def transactions(events: pd.DataFrame) -> np.ndarray:
    kind = events["type"]
    return (((kind == "purchase") & (events["reason"] == "purchase")) | (kind == "transfer_out")).to_numpy(dtype=bool)


class Table:
    def __init__(self, rows: int) -> None:
        self.rows = rows
        self.columns: dict[str, np.ndarray] = {}
        self.descriptions: dict[str, str] = {}

    def put(self, name: str, values, description: str, categorical: bool = False) -> None:
        if name in self.columns:
            raise ValueError(f"признак {name} уже есть")
        if categorical:
            series = pd.Series(values, dtype=object)
            self.columns[name] = series.where(series.notna(), MISSING).astype(str).to_numpy()
        else:
            self.columns[name] = np.asarray(values, dtype=np.float32)
        self.descriptions[name] = description


class Stream:
    """
    События одного вида по возрастанию ключа (клиент, время) и накопленные
    суммы значений. Все запросы — строго раньше момента строки.
    """

    def __init__(self, keys: np.ndarray, values: np.ndarray | None = None) -> None:
        self.keys = keys
        self.values = values
        self.cumulative = None if values is None else np.r_[0.0, np.cumsum(values)]

    def _bounds(self, at: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
        return np.searchsorted(self.keys, at - window, "left"), np.searchsorted(self.keys, at, "left")

    def count(self, at: np.ndarray, window: int) -> np.ndarray:
        low, high = self._bounds(at, window)
        return (high - low).astype(float)

    def total(self, at: np.ndarray, window: int) -> np.ndarray:
        low, high = self._bounds(at, window)
        return self.cumulative[high] - self.cumulative[low]

    def previous(self, at: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Номер последнего события строго раньше at у того же клиента и маска,
        есть ли оно.
        """
        index = np.searchsorted(self.keys, at, "left") - 1
        safe = np.maximum(index, 0)
        valid = (index >= 0) & (self.keys[safe] // SPAN == at // SPAN) if len(self.keys) else np.zeros(len(at), bool)
        return safe, valid

    def hours_since(self, at: np.ndarray) -> np.ndarray:
        index, valid = self.previous(at)
        if not len(self.keys):
            return np.full(len(at), np.nan)
        return np.where(valid, (at - self.keys[index]) / HOUR, np.nan)

    def last_value(self, at: np.ndarray) -> np.ndarray:
        index, valid = self.previous(at)
        if not len(self.keys):
            return np.full(len(at), np.nan)
        return np.where(valid, self.values[index], np.nan)


def _stream(key: np.ndarray, mask: np.ndarray, values: np.ndarray | None = None) -> Stream:
    return Stream(key[mask], None if values is None else values[mask])


def _grouped_since(code: np.ndarray, t_ms: np.ndarray, values: pd.Series, history: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """
    Часов с прошлого события того же клиента с тем же значением (MCC,
    мерчант, страна, получатель) строго раньше строки. Пусто — значение у
    клиента впервые или пустое.
    """
    present = values.notna().to_numpy()
    wanted = np.zeros(len(values), dtype=bool)
    wanted[rows] = True
    subset = np.flatnonzero((history & present) | wanted)
    labels = pd.Series(code[subset].astype(str), dtype=object) + "\x1f" + values.iloc[subset].astype(object).astype(str).to_numpy()
    group, _ = pd.factorize(labels)
    # Ключ группы * SPAN + время помещается в int64, пока групп меньше 9e5.
    if len(group) and group.max() >= 900_000:
        raise OverflowError("слишком много пар (клиент, значение) в блоке")
    keys = group.astype(np.int64) * SPAN + t_ms[subset]
    in_history = (history & present)[subset]
    stream = Stream(np.sort(keys[in_history], kind="stable"))
    position = np.searchsorted(subset, rows)
    result = stream.hours_since(keys[position])
    result[~present[rows]] = np.nan
    return result


def _share(part: np.ndarray, whole: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(whole > 0, part / np.where(whole > 0, whole, 1.0), np.nan)


def compute(events: pd.DataFrame, rows: np.ndarray, profile: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """
    Признаки строк rows (позиции в events) блока целых клиентов.

    events упорядочены по клиенту и времени (raw.client_blocks); profile —
    анкета тех же клиентов из выгрузки, индекс client_id.
    """
    clients = pd.Index(pd.unique(events["client_id"]))
    code = clients.get_indexer(events["client_id"]).astype(np.int64)
    t_ms = milliseconds(events["t"])
    key = code * SPAN + t_ms
    at = key[rows]
    table = Table(len(rows))

    _operation(table, events, rows)
    _history(table, events, rows, key, at)
    _novelty(table, events, rows, code, t_ms)
    _app(table, events, key, at)
    _money(table, events, rows, key, at)
    _profile(table, events, rows, t_ms[rows], profile)
    return table.columns, table.descriptions


def milliseconds(moments: pd.Series) -> np.ndarray:
    return ((moments - EPOCH) // pd.Timedelta(milliseconds=1)).to_numpy(dtype=np.int64)


def _operation(table: Table, events: pd.DataFrame, rows: np.ndarray) -> None:
    row = events.iloc[rows]
    kind = row["type"].to_numpy()
    amount = row["amount"].to_numpy(dtype=float, na_value=np.nan)
    country = row["merchant_country"]
    original = row["original_currency"]
    local = row["t"] + LOCAL_OFFSET

    table.put("is_transfer", kind == "transfer_out", "исходящий перевод (иначе покупка)")
    table.put("own_transfer", (row["reason"] == "own_transfer").to_numpy(), "перевод между своими счетами")
    table.put("channel", row["channel"].to_numpy(), "канал операции (pos, ecom, qr, app)", categorical=True)
    table.put("is_online", row["is_online"].fillna(False).to_numpy(dtype=bool), "онлайн-операция")
    table.put("amount", amount, "сумма операции, KZT")
    table.put("log_amount", np.log1p(amount), "log(1 + сумма)")
    table.put(
        "foreign_currency",
        (original.notna() & (original != "KZT")).to_numpy(),
        "исходная валюта операции не тенге",
    )
    table.put("merchant_country", country.to_numpy(), "страна мерчанта", categorical=True)
    table.put("is_foreign", (country.notna() & (country != "KZ")).to_numpy(), "страна мерчанта не Казахстан")
    table.put("mcc", row["mcc"].to_numpy(), "MCC", categorical=True)
    table.put("merchant_category", row["merchant_category"].to_numpy(), "категория мерчанта", categorical=True)
    table.put("merchant_name", row["merchant_name"].to_numpy(), "название мерчанта", categorical=True)
    table.put("hour", local.dt.hour.to_numpy(), "час операции по местному времени")
    table.put("weekday", local.dt.weekday.to_numpy(), "день недели (0 — понедельник)")
    table.put("day_of_month", local.dt.day.to_numpy(), "число месяца")


def _history(table: Table, events: pd.DataFrame, rows: np.ndarray, key: np.ndarray, at: np.ndarray) -> None:
    txn = transactions(events)
    amount = events["amount"].to_numpy(dtype=float, na_value=np.nan)
    amount = np.where(np.isnan(amount), 0.0, amount)
    log_amount = np.log1p(amount)
    status = events["status"]
    country = events["merchant_country"]

    all_txn = _stream(key, txn, amount)
    declined = _stream(key, txn & (status == "declined").to_numpy())
    online = _stream(key, txn & events["is_online"].fillna(False).to_numpy(dtype=bool))
    foreign = _stream(key, txn & (country.notna() & (country != "KZ")).to_numpy())
    logs = _stream(key, txn, log_amount)
    squares = _stream(key, txn, log_amount**2)

    table.put("txn_hours_since_prev", all_txn.hours_since(at), "часов с прошлой операции клиента")
    for name, window in (("1h", HOUR), ("1d", DAY), ("7d", 7 * DAY), ("30d", 30 * DAY), ("90d", 90 * DAY)):
        table.put(f"txn_count_{name}", all_txn.count(at, window), f"операций клиента за {name} до строки")
    for name, window in (("1h", HOUR), ("1d", DAY), ("7d", 7 * DAY), ("30d", 30 * DAY), ("90d", 90 * DAY)):
        table.put(f"txn_sum_{name}", all_txn.total(at, window), f"сумма операций клиента за {name} до строки, KZT")
    for name, window in (("1h", HOUR), ("1d", DAY)):
        table.put(f"declined_count_{name}", declined.count(at, window), f"отклонённых операций за {name} до строки")
    table.put("online_count_1d", online.count(at, DAY), "онлайн-операций за 1d до строки")
    table.put("foreign_count_1d", foreign.count(at, DAY), "операций за рубежом за 1d до строки")
    table.put(
        "foreign_share_90d",
        _share(foreign.count(at, 90 * DAY), table.columns["txn_count_90d"].astype(float)),
        "доля операций за рубежом за 90d",
    )

    row_amount = amount[rows]
    count_90 = table.columns["txn_count_90d"].astype(float)
    mean_90 = _share(table.columns["txn_sum_90d"].astype(float), count_90)
    table.put("amount_to_mean_90d", _share(row_amount, mean_90), "сумма / средняя операция за 90d")

    log_sum = logs.total(at, 90 * DAY)
    square_sum = squares.total(at, 90 * DAY)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = log_sum / count_90
        spread = np.sqrt(np.maximum(square_sum / count_90 - mean**2, 0.0))
        z = np.where((count_90 >= 2) & (spread > 0), (np.log1p(row_amount) - mean) / np.where(spread > 0, spread, 1.0), np.nan)
    table.put("log_amount_z_90d", z, "z-оценка log-суммы среди операций за 90d")

    # Наибольшая прошлая операция клиента за всю историю до строки.
    code = key[txn] // SPAN
    running = pd.Series(amount[txn]).groupby(code).cummax().to_numpy()
    largest = Stream(key[txn], running).last_value(at)
    table.put("amount_to_max_prev", _share(row_amount, largest), "сумма / наибольшая прошлая операция")


def _novelty(table: Table, events: pd.DataFrame, rows: np.ndarray, code: np.ndarray, t_ms: np.ndarray) -> None:
    kind = events["type"]
    purchase = ((kind == "purchase") & (events["reason"] == "purchase")).to_numpy()
    outgoing = kind.isin(["transfer_out", "p2p_out"]).to_numpy()

    for name, column, history, title in (
        ("mcc", "mcc", purchase, "этим MCC"),
        ("merchant", "merchant_name", purchase, "у этого мерчанта"),
        ("country", "merchant_country", purchase, "в этой стране"),
        ("counterparty", "counterparty", outgoing, "этому получателю"),
    ):
        hours = _grouped_since(code, t_ms, events[column], history, rows)
        table.put(f"{name}_hours_since_prev", hours, f"часов с прошлой операции клиента {title}; пусто — впервые")
        present = events[column].notna().to_numpy()[rows]
        table.put(f"{name}_is_new", present & np.isnan(hours), f"клиент впервые платит {title}")


def _app(table: Table, events: pd.DataFrame, key: np.ndarray, at: np.ndarray) -> None:
    kind = events["type"]
    login = ((kind == "app_operation") & events["operation"].isin(LOGIN_OPERATIONS)).to_numpy()
    failed = login & (events["status"] == "failed").to_numpy()
    new_device = login & events["device_new"].fillna(False).to_numpy(dtype=bool)
    alert = (kind == "fraud_alert").to_numpy()
    blocked = ((kind == "card_blocked") & (events["reason"] == "fraud_suspicion")).to_numpy()

    logins = _stream(key, login)
    table.put("login_hours_since_prev", logins.hours_since(at), "часов с прошлого входа в приложение")
    for name, window in (("1h", HOUR), ("1d", DAY)):
        table.put(f"login_count_{name}", logins.count(at, window), f"входов в приложение за {name} до строки")
        table.put(f"login_failed_{name}", _stream(key, failed).count(at, window), f"неудачных входов за {name} до строки")
        table.put(f"new_device_login_{name}", _stream(key, new_device).count(at, window), f"входов с нового устройства за {name} до строки")
    table.put("new_device_hours_since_prev", _stream(key, new_device).hours_since(at), "часов с прошлого входа с нового устройства")

    for name, window in (("1d", DAY), ("30d", 30 * DAY)):
        table.put(f"fraud_alert_{name}", _stream(key, alert).count(at, window), f"антифрод-алертов банка за {name} до строки")
    table.put("fraud_block_30d", _stream(key, blocked).count(at, 30 * DAY), "блокировок карты антифродом за 30d до строки")


def _money(table: Table, events: pd.DataFrame, rows: np.ndarray, key: np.ndarray, at: np.ndarray) -> None:
    kind = events["type"]
    amount = events["amount"].to_numpy(dtype=float, na_value=np.nan)
    amount = np.where(np.isnan(amount), 0.0, amount)
    income = kind.isin(INCOME_TYPES).to_numpy()
    table.put("income_sum_90d", _stream(key, income, amount).total(at, 90 * DAY), "доходы (зарплата, пенсия, прочие) за 90d, KZT")

    balance = events["balance_after"].to_numpy(dtype=float, na_value=np.nan)
    known = kind.isin(EVERYDAY_TYPES).to_numpy() & ~np.isnan(balance)
    before = _stream(key, known, balance).last_value(at)
    table.put("balance_before", before, "последний известный остаток повседневного счёта до строки")
    row_amount = events["amount"].to_numpy(dtype=float, na_value=np.nan)[rows]
    table.put("amount_to_balance", _share(row_amount, np.where(before > 0, before, np.nan)), "сумма / остаток до строки")


def _profile(table: Table, events: pd.DataFrame, rows: np.ndarray, row_ms: np.ndarray, profile: pd.DataFrame) -> None:
    """
    profile — анкета из выгрузки с колонками gender, birth_date, полями
    PROFILE_FIELDS на as_of и моментами вех в мс (<веха>_ms).
    """
    row = events.iloc[rows][["client_id", "t"]].reset_index(drop=True)
    row["position"] = np.arange(len(row))
    ordered = row.sort_values("t", kind="stable")

    changes = events[(events["type"] == "profile_change") & events["field_name"].isin(PROFILE_FIELDS)]
    values: dict[str, np.ndarray] = {}
    for field in PROFILE_FIELDS:
        later = changes[changes["field_name"] == field][["client_id", "t", "old_value"]].sort_values("t", kind="stable")
        later = later.rename(columns={"t": "changed_at"})
        # Первое изменение в момент строки или позже несёт значение на момент строки.
        merged = pd.merge_asof(
            ordered, later, left_on="t", right_on="changed_at", by="client_id",
            direction="forward", allow_exact_matches=True,
        ).sort_values("position")
        snapshot = profile[field].reindex(merged["client_id"]).to_numpy()
        values[field] = np.where(merged["changed_at"].notna().to_numpy(), merged["old_value"].to_numpy(), snapshot)

    local_day = (row["t"] + LOCAL_OFFSET).dt.date
    born = profile["birth_date"].reindex(row["client_id"]).to_numpy()
    age = [day.year - b.year - ((day.month, day.day) < (b.month, b.day)) for day, b in zip(local_day, born)]

    table.put("age", age, "полных лет на местную дату операции")
    table.put("gender", profile["gender"].reindex(row["client_id"]).to_numpy(), "пол", categorical=True)
    table.put("region", values["region"], "регион на момент операции", categorical=True)
    table.put("income_type", values["income_type"], "вид дохода на момент операции", categorical=True)
    table.put(
        "declared_income",
        pd.to_numeric(pd.Series(values["declared_income"], dtype=object), errors="raise").to_numpy(dtype=float),
        "заявленный доход на момент операции",
    )

    city = pd.Series(values["city"], dtype=object)
    merchant_city = pd.Series(events["merchant_city"].to_numpy()[rows], dtype=object)
    other_city = merchant_city.notna() & city.notna() & (merchant_city.astype(str) != city.astype(str))
    table.put("merchant_city_other", other_city.to_numpy(), "город мерчанта не совпадает с городом клиента")

    for milestone in MILESTONES:
        moments = profile[f"{milestone}_ms"].reindex(row["client_id"]).to_numpy(dtype=float)
        days = (row_ms - moments) / DAY
        table.put(
            f"days_since_{milestone}",
            np.where(days > 0, days, np.nan),
            f"дней от вехи {milestone} до операции; пусто — вехи строго раньше ещё не было",
        )
