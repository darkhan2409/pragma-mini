from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .. import params as params_module
from .. import config
from ..config import REGISTRY_START
from ..rng import state_cache
from . import hcb_timeline
from .hcb_timeline import Period, ProductRecord, Timeline, Version


# ============================================================
# ВРЕМЕННОЙ КАТАЛОГ ПРОДУКТОВ
# ============================================================
#
# Продукт живёт во времени: он объявляется, проходит пилот,
# продаётся, приостанавливается, закрывается для новых клиентов
# и наконец перестаёт обслуживаться.
#
# ЗАКРЫТИЕ ПРОДАЖ НЕ ЗАКРЫВАЕТ ДОГОВОРЫ. Договор запоминает
# версию условий и тарифа на дату подписания, и новая версия
# касается его только по applies_to.
#
# Статус выводится ТОЛЬКО из дат: наличие или отсутствие
# договоров в конкретной синтетической выборке на него не
# влияет.
# ============================================================


STATUS_PLANNED = "planned"
STATUS_PILOT = "pilot"
STATUS_ACTIVE = "active"
STATUS_SUSPENDED = "temporarily_suspended"
STATUS_CLOSED_TO_NEW = "closed_to_new_clients"
STATUS_SERVICING = "servicing_existing_contracts"
STATUS_ARCHIVED = "archived"

STATUSES = (
    STATUS_PLANNED,
    STATUS_PILOT,
    STATUS_ACTIVE,
    STATUS_SUSPENDED,
    STATUS_CLOSED_TO_NEW,
    STATUS_SERVICING,
    STATUS_ARCHIVED,
)

SELLABLE_STATUSES = frozenset({STATUS_PILOT, STATUS_ACTIVE})

# Статусы, при которых действующий договор ещё обслуживается.
SERVICED_STATUSES = frozenset(
    {STATUS_PILOT, STATUS_ACTIVE, STATUS_SUSPENDED, STATUS_CLOSED_TO_NEW, STATUS_SERVICING}
)

FAR_FUTURE = datetime(9999, 1, 1)


# ============================================================
# ДАТА ДЛЯ СИМУЛЯЦИИ
# ============================================================


def simulation_date(period: Period, policy: str, unknown_start: datetime) -> datetime:
    """
    Конкретный день, которым симуляция пользуется вместо периода.

    Период НЕ переписывается: он остаётся в каталоге как есть,
    а выбранный день лежит рядом отдельной колонкой.
    """

    if period.precision == "unknown" or period.start is None:
        return unknown_start

    start = datetime.combine(period.start, datetime.min.time())

    if period.precision == "day" or period.end is None:
        return start

    end = datetime.combine(period.end, datetime.min.time())

    if policy == "period_end":
        return end

    if policy == "period_middle":
        return start + (end - start) / 2

    return start


# ============================================================
# СТРОКА КАТАЛОГА
# ============================================================


@dataclass(frozen=True)
class CatalogRow:
    product_id: str
    product_code: str
    product_family: str
    product_name: str
    group: str | None
    product_version: int
    tariff_version: int
    status: str
    valid_from: datetime
    valid_to: datetime
    announced: Period | None
    sales_start: Period | None
    sales_end: Period | None
    service_end: Period | None
    announced_at: datetime | None
    sales_start_at: datetime | None
    sales_end_at: datetime | None
    service_end_at: datetime | None
    eligibility: dict
    channels: tuple
    terms: dict
    applies_to: str
    notice_days: int | None
    allow_multiple: bool
    max_active_holdings: int
    compatibility_rules: dict
    replacement_rules: dict
    predecessor: str | None
    successor: str | None
    migration_policy: str
    source_url: str | None
    evidence_at: date | None
    confidence: str
    unresolved_source: bool
    is_synthetic: bool
    note: str


@dataclass(frozen=True)
class ProductView:
    """
    Продукт целиком: запись хронологии плюс разложенные во
    времени версии и статусы.
    """

    record: ProductRecord
    rows: tuple
    version_starts: tuple
    status_spans: tuple

    @property
    def code(self) -> str:
        return self.record.product_code

    @property
    def family(self) -> str:
        return self.record.family

    def version_at(self, ts: datetime) -> Version:
        """
        Версия условий на дату. Договор фиксирует её у себя и
        живёт на ней, пока обслуживается.
        """

        chosen = self.version_starts[0][1]

        for start, version in self.version_starts:
            if start <= ts:
                chosen = version
            else:
                break

        return chosen

    def later_versions(self, after: datetime) -> tuple:
        return tuple(
            version for start, version in self.version_starts if start > after
        )

    def version_start(self, version: Version) -> datetime:
        for start, item in self.version_starts:
            if item is version:
                return start
        return self.version_starts[0][0]

    def status_at(self, ts: datetime) -> str:
        status = STATUS_PLANNED

        for start, value in self.status_spans:
            if start <= ts:
                status = value
            else:
                break

        return status

    def sellable_at(self, ts: datetime) -> bool:
        return self.status_at(ts) in SELLABLE_STATUSES

    def serviced_at(self, ts: datetime) -> bool:
        return self.status_at(ts) in SERVICED_STATUSES


# ============================================================
# СБОРКА
# ============================================================


_KIND_ORDER = {
    "announced": 0,
    "pilot_start": 1,
    "sales_start": 2,
    "sales_resumed": 3,
    "sales_suspended": 4,
    "sales_end": 5,
    "service_end": 6,
}


def _status_spans(
    record: ProductRecord,
    policy: str,
    unknown_start: datetime,
    closed_to_servicing_days: int,
    first_version_start: datetime,
) -> tuple:
    """
    Переходы статуса во времени, выведенные только из дат.
    """

    transitions: list[tuple[datetime, int, str]] = []

    for milestone in record.milestones:

        kind = milestone.kind

        if kind == "feature_added":
            continue

        moment = simulation_date(milestone.period, policy, unknown_start)

        if kind == "announced":
            transitions.append((moment, _KIND_ORDER[kind], STATUS_PLANNED))
        elif kind == "pilot_start":
            transitions.append((moment, _KIND_ORDER[kind], STATUS_PILOT))
        elif kind in ("sales_start", "sales_resumed"):
            transitions.append((moment, _KIND_ORDER[kind], STATUS_ACTIVE))
        elif kind == "sales_suspended":
            transitions.append((moment, _KIND_ORDER[kind], STATUS_SUSPENDED))
        elif kind == "sales_end":
            transitions.append((moment, _KIND_ORDER[kind], STATUS_CLOSED_TO_NEW))
            transitions.append(
                (
                    moment + timedelta(days=closed_to_servicing_days),
                    _KIND_ORDER[kind],
                    STATUS_SERVICING,
                )
            )
        elif kind == "service_end":
            transitions.append((moment, _KIND_ORDER[kind], STATUS_ARCHIVED))

    if not transitions:
        transitions.append((first_version_start, _KIND_ORDER["sales_start"], STATUS_ACTIVE))

    transitions.sort(key=lambda item: (item[0], item[1]))

    spans: list[tuple[datetime, str]] = []

    for moment, _, status in transitions:
        if spans and spans[-1][0] == moment:
            spans[-1] = (moment, status)
        else:
            spans.append((moment, status))

    # Архив это терминальное состояние: производный переход в
    # servicing_existing_contracts, посчитанный от закрытия
    # продаж, не должен воскрешать уже снятый продукт.
    for index, (_, status) in enumerate(spans):
        if status == STATUS_ARCHIVED:
            return tuple(spans[: index + 1])

    return tuple(spans)


def _holding_rules(record: ProductRecord) -> dict:

    rules = dict(record.holding_rules or {})

    return {
        "allow_multiple": bool(rules.get("allow_multiple", False)),
        "max_active_holdings": int(rules.get("max_active_holdings", 1)),
        "compatibility_rules": dict(rules.get("compatibility_rules") or {}),
        "replacement_rules": dict(rules.get("replacement_rules") or {}),
    }


def _version_starts(
    record: ProductRecord,
    policy: str,
    unknown_start: datetime,
) -> tuple:
    """
    Дни, с которых версии условий действуют в симуляции.

    Версии перечислены в хронологии по порядку. Если у более
    поздней версии известен только факт «действует к такой-то
    дате», её началом нельзя брать начало реестра: тогда
    свежий тариф оказался бы старше предыдущего. Границы такой
    версии это отрезок от предыдущей версии до даты
    свидетельства, и день выбирается внутри него.
    """

    starts: list[tuple[datetime, Version]] = []

    previous: datetime | None = None

    for version in record.versions:

        period = version.effective

        start = simulation_date(period, policy, unknown_start)

        if previous is not None and start <= previous:

            end = (
                datetime.combine(period.end, datetime.min.time())
                if period.end is not None
                else None
            )

            if end is not None and end > previous:
                bounded = simulation_date(
                    Period(start=previous.date(), end=period.end, precision="interval"),
                    policy,
                    unknown_start,
                )
                start = bounded if bounded > previous else previous + (end - previous) / 2
            else:
                start = previous + timedelta(days=1)

        starts.append((start, version))

        previous = start

    starts.sort(key=lambda item: (item[0], item[1].product_version, item[1].tariff_version))

    return tuple(starts)


def _build_view(
    record: ProductRecord,
    policy: str,
    unknown_start: datetime,
    closed_to_servicing_days: int,
) -> ProductView:

    version_starts = _version_starts(record, policy, unknown_start)

    spans = _status_spans(
        record,
        policy,
        unknown_start,
        closed_to_servicing_days,
        version_starts[0][0],
    )

    rules = _holding_rules(record)

    milestone_periods = {
        kind: record.milestone(kind).period if record.milestone(kind) else None
        for kind in ("announced", "sales_start", "sales_end", "service_end")
    }

    milestone_dates = {
        kind: (
            simulation_date(period, policy, unknown_start) if period is not None else None
        )
        for kind, period in milestone_periods.items()
    }

    sales_start_milestone = record.milestone("sales_start")

    boundaries = sorted(
        {start for start, _ in version_starts}
        | {start for start, _ in spans}
    )

    rows: list[CatalogRow] = []

    for index, start in enumerate(boundaries):

        stop = boundaries[index + 1] if index + 1 < len(boundaries) else FAR_FUTURE

        version = None
        for version_start, candidate in version_starts:
            if version_start <= start:
                version = candidate
        if version is None:
            version = version_starts[0][1]

        status = STATUS_PLANNED
        for span_start, value in spans:
            if span_start <= start:
                status = value

        rows.append(
            CatalogRow(
                product_id=record.product_id,
                product_code=record.product_code,
                product_family=record.family,
                product_name=record.name,
                group=record.group,
                product_version=version.product_version,
                tariff_version=version.tariff_version,
                status=status,
                valid_from=start,
                valid_to=stop,
                announced=milestone_periods["announced"],
                sales_start=milestone_periods["sales_start"],
                sales_end=milestone_periods["sales_end"],
                service_end=milestone_periods["service_end"],
                announced_at=milestone_dates["announced"],
                sales_start_at=milestone_dates["sales_start"],
                sales_end_at=milestone_dates["sales_end"],
                service_end_at=milestone_dates["service_end"],
                eligibility=version.eligibility,
                channels=version.channels,
                terms=version.terms,
                applies_to=version.applies_to,
                notice_days=version.notice_days if version.notice_days is not None else record.notice_days,
                allow_multiple=rules["allow_multiple"],
                max_active_holdings=rules["max_active_holdings"],
                compatibility_rules=rules["compatibility_rules"],
                replacement_rules=rules["replacement_rules"],
                predecessor=record.predecessor,
                successor=record.successor,
                migration_policy=record.migration_policy,
                source_url=version.source,
                evidence_at=sales_start_milestone.evidence_at if sales_start_milestone else None,
                confidence="synthetic" if record.is_synthetic else version.confidence,
                unresolved_source=version.unresolved_source,
                is_synthetic=record.is_synthetic,
                note=version.note,
            )
        )

    return ProductView(
        record=record,
        rows=tuple(rows),
        version_starts=version_starts,
        status_spans=spans,
    )


# ============================================================
# ВЫМЫШЛЕННЫЕ ПРОДУКТЫ
# ============================================================


def _synthetic_records(specs: tuple) -> tuple:
    """
    Превращает описания SYNTH_* из параметров в записи того же
    вида, что и подтверждённая хронология. Реальные продукты
    синтетических дат и тарифов не получают никогда.
    """

    records: list[ProductRecord] = []

    for spec in specs:

        code = str(spec["product_code"])

        if not code.startswith("SYNTH_"):
            raise ValueError(f"вымышленный продукт {code} обязан иметь код SYNTH_*")

        launch = dict(spec.get("launch") or {})

        milestones: list[hcb_timeline.Milestone] = []

        def _day(value: str) -> Period:
            moment = date.fromisoformat(value)
            return Period(start=moment, end=moment, precision="day")

        if launch.get("announced_at"):
            milestones.append(
                hcb_timeline.Milestone(
                    kind="announced",
                    period=_day(launch["announced_at"]),
                    source="params.products.synthetic_products",
                    confidence="low",
                    title="синтетическое объявление",
                )
            )

        pilot = launch.get("pilot")

        if pilot:
            milestones.append(
                hcb_timeline.Milestone(
                    kind="pilot_start",
                    period=_day(pilot["from"]),
                    source="params.products.synthetic_products",
                    confidence="low",
                    title="синтетический пилот",
                )
            )

        if launch.get("sales_start_at"):
            milestones.append(
                hcb_timeline.Milestone(
                    kind="sales_start",
                    period=_day(launch["sales_start_at"]),
                    source="params.products.synthetic_products",
                    confidence="low",
                    title="синтетический запуск",
                )
            )

        versions: list[Version] = []

        for item in spec.get("versions") or ():
            versions.append(
                Version(
                    product_version=int(item["product_version"]),
                    tariff_version=int(item["tariff_version"]),
                    effective=_day(launch.get("sales_start_at") or launch["announced_at"]),
                    applies_to=item.get("applies_to", "new_clients_only"),
                    eligibility=dict(item.get("eligibility") or {}),
                    channels=tuple(item.get("channels") or ()),
                    terms=dict(item.get("terms") or {}),
                    source="params.products.synthetic_products",
                    confidence="low",
                    notice_days=item.get("notice_days"),
                    note="синтетический продукт",
                )
            )

        successor = spec.get("successor")

        for event in spec.get("events") or ():

            kind = event["kind"]

            if kind == "tariff_change":
                last = versions[-1]
                versions.append(
                    Version(
                        product_version=last.product_version,
                        tariff_version=last.tariff_version + 1,
                        effective=_day(event["at"]),
                        applies_to=event.get("applies_to", "new_clients_only"),
                        eligibility=last.eligibility,
                        channels=last.channels,
                        terms=dict(event.get("terms") or last.terms),
                        source="params.products.synthetic_products",
                        confidence="low",
                        notice_days=event.get("notice_days"),
                        note="синтетическая смена тарифа",
                    )
                )
            elif kind == "suspension":
                milestones.append(
                    hcb_timeline.Milestone(
                        kind="sales_suspended",
                        period=_day(event["from"]),
                        source="params.products.synthetic_products",
                        confidence="low",
                        title="синтетическая приостановка",
                    )
                )
                if event.get("to"):
                    milestones.append(
                        hcb_timeline.Milestone(
                            kind="sales_resumed",
                            period=_day(event["to"]),
                            source="params.products.synthetic_products",
                            confidence="low",
                            title="синтетическое возобновление",
                        )
                    )
            elif kind == "sales_close":
                milestones.append(
                    hcb_timeline.Milestone(
                        kind="sales_end",
                        period=_day(event["at"]),
                        source="params.products.synthetic_products",
                        confidence="low",
                        title="синтетическое закрытие продаж",
                    )
                )
                successor = event.get("successor", successor)
            elif kind == "service_end":
                milestones.append(
                    hcb_timeline.Milestone(
                        kind="service_end",
                        period=_day(event["at"]),
                        source="params.products.synthetic_products",
                        confidence="low",
                        title="синтетическое окончание обслуживания",
                    )
                )

        records.append(
            ProductRecord(
                product_code=code,
                product_id=f"prd_{code.lower()}",
                family=spec["family"],
                name=str(spec.get("name") or code),
                milestones=tuple(milestones),
                versions=tuple(versions),
                holding_rules=dict(spec.get("holding_rules") or {"allow_multiple": False, "max_active_holdings": 1}),
                predecessor=spec.get("predecessor"),
                successor=successor,
                migration_policy=spec.get("migration_policy", "none"),
                notice_days=spec.get("notice_days"),
                is_synthetic=True,
            )
        )

    return tuple(records)


# ============================================================
# КАТАЛОГ
# ============================================================


@dataclass(frozen=True)
class ProductCatalog:
    views: dict
    rows: tuple
    timeline_sha256: str
    bank_timeline: tuple
    unresolved_sources: tuple
    unknown_start: datetime
    policy: str

    def view(self, code: str) -> ProductView:
        return self.views[code]

    def has(self, code: str) -> bool:
        return code in self.views

    def by_family(self, family: str) -> tuple:
        return tuple(
            view for view in self.views.values() if view.family == family
        )

    def sellable(self, ts: datetime, family: str | None = None) -> tuple:
        return tuple(
            view
            for view in self.views.values()
            if view.sellable_at(ts) and (family is None or view.family == family)
        )

    def families(self) -> tuple:
        return tuple(sorted({view.family for view in self.views.values()}))

    def family_of(self, product_id: str | None) -> str | None:
        """
        Семейство по идентификатору продукта.

        Нужно там, где семейства нет под рукой: в событие оно
        больше не копируется, потому что его знает каталог.
        """

        if not product_id:
            return None

        for view in self.views.values():
            if view.record.product_id == product_id:
                return view.family

        return None

    def version_at(self, code: str, ts: datetime) -> Version:
        return self.views[code].version_at(ts)


@state_cache
def _build(policy: str, unknown_policy: str, closed_to_servicing_days: int, synthetic_key: str) -> ProductCatalog:

    settings = params_module.active()

    timeline: Timeline = hcb_timeline.load()

    unknown_start = REGISTRY_START if unknown_policy == "registry_start" else config.HISTORY_START

    records = list(timeline.products)
    records.extend(_synthetic_records(settings.products.synthetic_products))

    views: dict[str, ProductView] = {}
    rows: list[CatalogRow] = []

    for record in records:
        view = _build_view(record, policy, unknown_start, closed_to_servicing_days)
        views[record.product_code] = view
        rows.extend(view.rows)

    rows.sort(key=lambda row: (row.product_code, row.valid_from))

    return ProductCatalog(
        views=views,
        rows=tuple(rows),
        timeline_sha256=timeline.sha256,
        bank_timeline=timeline.bank_timeline,
        unresolved_sources=tuple(timeline.unresolved_sources()),
        unknown_start=unknown_start,
        policy=policy,
    )


def catalog() -> ProductCatalog:
    """
    Временной каталог продуктов для активных параметров.
    """

    settings = params_module.active().products

    return _build(
        settings.effective_date_policy,
        getattr(settings, "unknown_start_policy", "registry_start"),
        int(getattr(settings, "closed_to_servicing_days", 365)),
        str(len(settings.synthetic_products)),
    )


__all__ = [
    "CatalogRow",
    "ProductCatalog",
    "ProductView",
    "SELLABLE_STATUSES",
    "SERVICED_STATUSES",
    "STATUSES",
    "STATUS_ACTIVE",
    "STATUS_ARCHIVED",
    "STATUS_CLOSED_TO_NEW",
    "STATUS_PILOT",
    "STATUS_PLANNED",
    "STATUS_SERVICING",
    "STATUS_SUSPENDED",
    "catalog",
    "simulation_date",
]
