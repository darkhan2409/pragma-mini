from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime

from .app import AppOperationEvent, AppScreenEvent, BannerEvent
from .chains import derive_lifecycle
from .communications import CommunicationEvent
from .config import HISTORY_START, LABEL_END
from .coverage import first_seen
from .noise import apply_noise
from .products import ProductEvent
from .profile import ProfileSnapshot, profile_snapshots
from .transactions import TransactionEvent, generate_transaction_history
from .version import DEFAULT_VERSION, V2_1, check_version, revision_for


# ============================================================
# ИДЕЯ
# ============================================================
#
# История клиента это семь потоков на общем горизонте:
#
#     profile          состояние на конец каждого месяца
#     transactions     покупки, зарплата, подписки
#     product_events   открытия договоров
#     communications   отправки банка
#     app_screens      экраны приложения и воронка заявки
#     app_operations   операции по доменам
#     banners          показы и клики
#
# После генерации применяется фильтр покрытия: событие остаётся
# только если источник уже существовал И клиент в нём уже виден.
#
# Реестр договоров старше окна: его события могут быть раньше
# HISTORY_START. Остальные потоки живут внутри [start, end).
# ============================================================


# Имя RAW-таблицы -> поле ClientHistory.
EVENT_TABLES: dict[str, str] = {
    "transactions": "transactions",
    "product_events": "product_events",
    "communications": "communications",
    "app_screens": "app_screens",
    "app_operations": "app_operations",
    "banners": "banners",
}

# Все потоки, включая профиль.
STREAM_FIELDS: dict[str, str] = {"profile": "profile", **EVENT_TABLES}


@dataclass(frozen=True)
class ClientHistory:

    client_id: int

    start: datetime
    end: datetime

    profile: list[ProfileSnapshot]
    transactions: list[TransactionEvent]
    product_events: list[ProductEvent]
    communications: list[CommunicationEvent]
    app_screens: list[AppScreenEvent]
    app_operations: list[AppOperationEvent]
    banners: list[BannerEvent]

    # Правилами какой версии порождена история. Нужно ленте:
    # набор полей payload зависит от ревизии схемы, а ревизия
    # от версии. repr=False и compare=False, потому что золотые
    # дайджесты V1 это sha256(repr(history)).
    version: str = field(default=DEFAULT_VERSION, repr=False, compare=False)

    @property
    def revision(self) -> int:
        """
        Ревизия схемы RAW, которой соответствует эта история.
        """

        return revision_for(self.version)

    # --------------------------------------------------------

    def events(self, source: str) -> list:
        return getattr(self, STREAM_FIELDS[source])

    def before(self, ts: datetime) -> ClientHistory:
        """
        Копия истории только с событиями строго раньше ts.
        """

        if ts >= self.end:
            return self

        return replace(
            self,
            end=ts,
            **{
                field: [event for event in getattr(self, field) if event.ts < ts]
                for field in STREAM_FIELDS.values()
            },
        )

    def since(self, ts: datetime) -> ClientHistory:
        """
        Копия истории только с событиями не раньше ts.
        """

        if ts <= self.start:
            return self

        return replace(
            self,
            start=ts,
            **{
                field: [event for event in getattr(self, field) if event.ts >= ts]
                for field in STREAM_FIELDS.values()
            },
        )


# ============================================================
# ФИЛЬТР ПОКРЫТИЯ
# ============================================================


def apply_coverage(client_id: int, source: str, events: list) -> list:
    """
    Оставляет события, которые реально попали бы в хранилище.
    """

    start = first_seen(client_id, source)

    if start is None:
        return []

    return [event for event in events if event.ts >= start]


# ============================================================
# НАБЛЮДАЕМОЕ ПРЕДСТАВЛЕНИЕ
# ============================================================


def observed(history: ClientHistory) -> ClientHistory:
    """
    История, какой её видит хранилище: с наблюдательным шумом.

    ЕДИНСТВЕННЫЙ источник и для типизированных таблиц, и для
    единой ленты. Если применять шум только к таблицам, одно
    и то же событие будет выглядеть в них по-разному.
    """

    return replace(
        history,
        **{
            field: [
                apply_noise(source, event, history.client_id)
                for event in getattr(history, field)
            ]
            for source, field in STREAM_FIELDS.items()
        },
    )


# ============================================================
# ГЕНЕРАЦИЯ
# ============================================================


def generate_client_history(
    client_id: int,
    total_clients: int = 0,
    start: datetime = HISTORY_START,
    end: datetime = LABEL_END,
    version: str = DEFAULT_VERSION,
) -> ClientHistory:
    """
    Полная история одного клиента.

    total_clients не используется: клиенты независимы.
    Параметр оставлен для совместимости вызовов.

    version выбирает правила поведения. Схемы, горизонт,
    покрытие, шум, лента и метка общие для всех версий.
    """

    if end <= start:
        raise ValueError("end must be after start")

    check_version(version)

    if version == V2_1:
        # Импорт внутри функции: подпакет v2 обращается
        # к chains и app, и модульный импорт дал бы цикл.
        from .v2.lifecycle import derive_lifecycle_v2
        from .v2.transactions import generate_transaction_history_v2

        lifecycle = derive_lifecycle_v2(client_id, start, end)
        transactions = generate_transaction_history_v2(
            client_id, start, end, lifecycle
        )

    else:
        # Коммуникации, приложение и договоры строятся вместе:
        # они зависят от владения продуктами на каждый день.
        lifecycle = derive_lifecycle(client_id, start, end)

        transactions = generate_transaction_history(client_id, start, end)

    profile = profile_snapshots(client_id, start, end, lifecycle.state)

    streams = {
        "profile": profile,
        "transactions": transactions,
        "product_events": lifecycle.product_events,
        "communications": lifecycle.communications,
        "app_screens": lifecycle.app_screens,
        "app_operations": lifecycle.app_operations,
        "banners": lifecycle.banners,
    }

    visible = {
        source: apply_coverage(client_id, source, events)
        for source, events in streams.items()
    }

    return ClientHistory(
        client_id=client_id,
        start=start,
        end=end,
        profile=visible["profile"],
        transactions=visible["transactions"],
        product_events=visible["product_events"],
        communications=visible["communications"],
        app_screens=visible["app_screens"],
        app_operations=visible["app_operations"],
        banners=visible["banners"],
        version=version,
    )
