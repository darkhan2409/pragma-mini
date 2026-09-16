from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from ..config import PRODUCT_TIMELINE_PATH
from ..rng import state_cache


# ============================================================
# ПОДТВЕРЖДЁННАЯ ХРОНОЛОГИЯ
# ============================================================
#
# Python здесь только ЗАГРУЖАЕТ и ПРОВЕРЯЕТ. Все факты живут в
# data/reference/home_product_timeline.yaml, и добавить дату,
# тариф или ссылку можно только туда.
#
# Дата это период. Месяц или год не превращаются в день:
# конкретный день для симуляции выбирается отдельно политикой
# и хранится рядом с периодом.
# ============================================================


PRECISIONS = ("day", "month", "year", "interval", "unknown")

CONFIDENCES = ("high", "medium", "low")

MILESTONE_KINDS = (
    "announced",
    "pilot_start",
    "sales_start",
    "sales_suspended",
    "sales_resumed",
    "sales_end",
    "service_end",
    "feature_added",
)

APPLIES_TO = ("new_clients_only", "existing_from_date", "on_renewal")

MIGRATION_POLICIES = (
    "none",
    "voluntary",
    "auto_on_renewal",
    "forced_with_notice",
    "servicing_only",
)

FAMILIES = (
    "debit_card",
    "credit_card",
    "cash_loan",
    "refinance",
    "installment",
    "deposit",
    "deposit_certificate",
    "bonds",
    "insurance",
    "service",
)


class TimelineError(ValueError):
    """
    Хронология не прошла проверку. Генерация не начинается.
    """


@dataclass(frozen=True)
class Period:
    """
    Границы знания о дате. start может отсутствовать, если
    известно только, что событие уже произошло к end.
    """

    start: date | None
    end: date | None
    precision: str

    def as_dict(self) -> dict:
        return {
            "known_period_start": self.start.isoformat() if self.start else None,
            "known_period_end": self.end.isoformat() if self.end else None,
            "date_precision": self.precision,
        }


@dataclass(frozen=True)
class Milestone:
    kind: str
    period: Period
    source: str | None
    confidence: str
    unresolved_source: bool = False
    title: str = ""
    evidence_at: date | None = None
    note: str = ""


@dataclass(frozen=True)
class Version:
    product_version: int
    tariff_version: int
    effective: Period
    applies_to: str
    eligibility: dict
    channels: tuple
    terms: dict
    source: str | None
    confidence: str
    unresolved_source: bool = False
    notice_days: int | None = None
    note: str = ""


@dataclass(frozen=True)
class ProductRecord:
    product_code: str
    product_id: str
    family: str
    name: str
    milestones: tuple
    versions: tuple
    holding_rules: dict = field(default_factory=dict)
    pages: tuple = ()
    group: str | None = None
    predecessor: str | None = None
    successor: str | None = None
    migration_policy: str = "none"
    notice_days: int | None = None
    is_synthetic: bool = False

    def milestone(self, kind: str) -> Milestone | None:
        for item in self.milestones:
            if item.kind == kind:
                return item
        return None

    def milestones_of(self, kind: str) -> tuple:
        return tuple(item for item in self.milestones if item.kind == kind)


@dataclass(frozen=True)
class Timeline:
    version: int
    collected_at: date | None
    products: tuple
    bank_timeline: tuple
    bank_rules: tuple
    sha256: str

    def by_code(self, code: str) -> ProductRecord | None:
        for product in self.products:
            if product.product_code == code:
                return product
        return None

    def unresolved_sources(self) -> list[dict]:
        """
        Записи, которым не хватает источника. Отчёт реализма
        перечисляет их как требующие подтверждения.
        """

        rows: list[dict] = []

        for product in self.products:

            for item in product.milestones:
                if item.unresolved_source:
                    rows.append(
                        {
                            "product_code": product.product_code,
                            "entry": f"milestone:{item.kind}",
                            "period": item.period.as_dict(),
                            "confidence": item.confidence,
                            "note": item.note,
                        }
                    )

            for item in product.versions:
                if item.unresolved_source:
                    rows.append(
                        {
                            "product_code": product.product_code,
                            "entry": f"version:{item.product_version}.{item.tariff_version}",
                            "period": item.effective.as_dict(),
                            "confidence": item.confidence,
                            "note": item.note,
                        }
                    )

        return rows


# ============================================================
# РАЗБОР
# ============================================================


def _as_date(value: Any, where: str) -> date | None:

    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise TimelineError(f"{where}: дата {value!r} не разбирается") from error

    raise TimelineError(f"{where}: ожидалась дата, получено {value!r}")


def _period(raw: Any, where: str) -> Period:

    if not isinstance(raw, dict):
        raise TimelineError(f"{where}: период должен быть объектом")

    precision = raw.get("date_precision")

    if precision not in PRECISIONS:
        raise TimelineError(f"{where}: date_precision {precision!r} вне {PRECISIONS}")

    start = _as_date(raw.get("known_period_start"), where)
    end = _as_date(raw.get("known_period_end"), where)

    if start is not None and end is not None and start > end:
        raise TimelineError(f"{where}: known_period_start позже known_period_end")

    if precision == "day" and start is not None and end is not None and start != end:
        raise TimelineError(f"{where}: точность day требует одинаковых границ периода")

    if precision == "unknown" and start is not None:
        raise TimelineError(f"{where}: точность unknown не может иметь начала периода")

    if start is None and end is None:
        raise TimelineError(f"{where}: период пуст")

    return Period(start=start, end=end, precision=precision)


def _source(raw: dict, where: str) -> tuple[str | None, str, bool]:

    source = raw.get("source")
    confidence = raw.get("confidence", "low")
    unresolved = bool(raw.get("unresolved_source", False))

    if confidence not in CONFIDENCES:
        raise TimelineError(
            f"{where}: confidence {confidence!r} вне {CONFIDENCES}; "
            "значение synthetic в файле хронологии запрещено"
        )

    if source is None:
        if confidence != "low" or not unresolved:
            raise TimelineError(
                f"{where}: source отсутствует, значит обязательны confidence=low "
                "и unresolved_source=true"
            )
    else:
        if not isinstance(source, str) or not source.strip():
            raise TimelineError(f"{where}: source должен быть непустой строкой")
        if unresolved:
            raise TimelineError(f"{where}: unresolved_source=true при заполненном source")

    return source, confidence, unresolved


def _milestone(raw: dict, where: str) -> Milestone:

    kind = raw.get("kind")

    if kind not in MILESTONE_KINDS:
        raise TimelineError(f"{where}: kind {kind!r} вне {MILESTONE_KINDS}")

    source, confidence, unresolved = _source(raw, where)

    return Milestone(
        kind=kind,
        period=_period(raw, where),
        source=source,
        confidence=confidence,
        unresolved_source=unresolved,
        title=str(raw.get("title", "")),
        evidence_at=_as_date(raw.get("evidence_at"), where),
        note=str(raw.get("note", "")),
    )


def _version(raw: dict, where: str) -> Version:

    for key in ("product_version", "tariff_version", "effective", "channels", "terms"):
        if key not in raw:
            raise TimelineError(f"{where}: нет обязательного ключа {key}")

    applies_to = raw.get("applies_to", "new_clients_only")

    if applies_to not in APPLIES_TO:
        raise TimelineError(f"{where}: applies_to {applies_to!r} вне {APPLIES_TO}")

    source, confidence, unresolved = _source(raw, where)

    return Version(
        product_version=int(raw["product_version"]),
        tariff_version=int(raw["tariff_version"]),
        effective=_period(raw["effective"], f"{where}.effective"),
        applies_to=applies_to,
        eligibility=dict(raw.get("eligibility") or {}),
        channels=tuple(raw.get("channels") or ()),
        terms=dict(raw.get("terms") or {}),
        source=source,
        confidence=confidence,
        unresolved_source=unresolved,
        notice_days=raw.get("notice_days"),
        note=str(raw.get("note", "")),
    )


def _product(raw: dict) -> ProductRecord:

    code = raw.get("product_code")

    if not code or not isinstance(code, str):
        raise TimelineError("продукт без product_code")

    where = f"продукт {code}"

    if code.startswith("SYNTH_"):
        raise TimelineError(
            f"{where}: коды SYNTH_* запрещены в файле подтверждённой хронологии"
        )

    family = raw.get("family")

    if family not in FAMILIES:
        raise TimelineError(f"{where}: family {family!r} вне {FAMILIES}")

    policy = raw.get("migration_policy", "none")

    if policy not in MIGRATION_POLICIES:
        raise TimelineError(f"{where}: migration_policy {policy!r} вне {MIGRATION_POLICIES}")

    milestones = tuple(
        _milestone(item, f"{where}.milestone[{index}]")
        for index, item in enumerate(raw.get("milestones") or ())
    )

    versions = tuple(
        _version(item, f"{where}.version[{index}]")
        for index, item in enumerate(raw.get("versions") or ())
    )

    if not versions:
        raise TimelineError(f"{where}: нет ни одной версии условий")

    return ProductRecord(
        product_code=code,
        product_id=str(raw.get("product_id") or f"prd_{code.lower()}"),
        family=family,
        name=str(raw.get("name") or code),
        milestones=milestones,
        versions=versions,
        holding_rules=dict(raw.get("holding_rules") or {}),
        pages=tuple(raw.get("pages") or ()),
        group=raw.get("group"),
        predecessor=raw.get("predecessor"),
        successor=raw.get("successor"),
        migration_policy=policy,
        notice_days=raw.get("notice_days"),
    )


def parse(raw: dict, sha256: str) -> Timeline:

    if not isinstance(raw, dict):
        raise TimelineError("корень файла хронологии должен быть объектом")

    products = tuple(_product(item) for item in (raw.get("products") or ()))

    if not products:
        raise TimelineError("в хронологии нет ни одного продукта")

    codes = [product.product_code for product in products]

    duplicates = {code for code in codes if codes.count(code) > 1}

    if duplicates:
        raise TimelineError(f"дублирующиеся product_code: {sorted(duplicates)}")

    known = set(codes)

    identifiers = [product.product_id for product in products]

    if len(set(identifiers)) != len(identifiers):
        raise TimelineError("product_id не уникальны")

    for product in products:
        for link in ("predecessor", "successor"):
            value = getattr(product, link)
            if value is not None and value not in known:
                raise TimelineError(
                    f"продукт {product.product_code}: {link} ссылается на неизвестный код {value!r}"
                )

    return Timeline(
        version=int(raw.get("version", 1)),
        collected_at=_as_date(raw.get("collected_at"), "collected_at"),
        products=products,
        bank_timeline=tuple(dict(item) for item in (raw.get("bank_timeline") or ())),
        bank_rules=tuple(dict(item) for item in (raw.get("bank_rules") or ())),
        sha256=sha256,
    )


@state_cache
def _load_cached(path_text: str) -> Timeline:

    path = Path(path_text)

    if not path.exists():
        raise TimelineError(f"файл хронологии не найден: {path}")

    payload = path.read_bytes()

    sha256 = hashlib.sha256(payload).hexdigest()

    return parse(yaml.safe_load(payload.decode("utf-8")), sha256)


def load(path: str | Path | None = None) -> Timeline:
    """
    Загружает и проверяет подтверждённую хронологию.
    """

    return _load_cached(str(Path(path) if path is not None else PRODUCT_TIMELINE_PATH))


__all__ = [
    "APPLIES_TO",
    "CONFIDENCES",
    "FAMILIES",
    "MIGRATION_POLICIES",
    "MILESTONE_KINDS",
    "PRECISIONS",
    "Milestone",
    "Period",
    "ProductRecord",
    "Timeline",
    "TimelineError",
    "Version",
    "load",
    "parse",
]
