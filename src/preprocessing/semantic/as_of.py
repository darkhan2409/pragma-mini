from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

from ..calendar import calendar_features
from ..history import CanonicalStore, ClientHistory, history_as_of, product_key
from ..projection import ENTITY_REFS, EVENT_TYPE_FIELD, LocalRefs, model_event
from ..settings import BASE_SOURCES
from . import activity as activity_module
from . import chains as chains_module
from . import formulas as formulas_module
from . import time as time_module
from .keys import (
    DYNAMIC_FIELDS,
    ENVELOPE_KEYS,
    NUMERIC,
    PRODUCT_KEYS,
    PROFILE_KEYS,
    REFERENCE_KEYS,
    RELATION_KEYS,
    SemanticKey,
    key_for,
    profile_change_keys,
)
from .merchants import MerchantCatalog


# ============================================================
# ИДЕЯ
# ============================================================
#
# Единый интерфейс смыслового слоя: semantic_as_of(client, cutoff)
# поверх истории на дату.
#
# Слой ничего не решает заново про видимость: события, версии,
# профиль и покрытие приходят из этапа 3, а модельную границу
# держит проекция. Здесь к ним добавляется смысл: у значения
# появляется ключ, у точки и продукта — расшифровка по
# справочнику, у события-следствия — смысл связи вместо
# идентификатора причины, у суммы — отношение к доходу и лимиту с
# причиной, когда знаменатель неизвестен.
#
# Значения остаются исходными: 12 500 это 12 500, «Europharma»
# это текст. Ни квантилей, ни нормализации, ни корзин здесь нет.
#
# Календарь не дублируется: шесть чисел берутся из общего канала
# и кладутся рядом с событием как есть.
# ============================================================


SEMANTIC_VERSION = "2.0.0"


@dataclass
class SemanticEvent:
    """
    Одно событие со смыслом.

    stable_event_index внутренний: он адресует логическое событие
    внутри слоя, чтобы связь и расчёт можно было надёжно
    приложить к своему событию. В values он не попадает и границу
    модели не пересекает.

    event_id тоже внутренний и тоже не становится
    значениями. Они названы отдельно от порядкового номера,
    потому что номер устойчив только внутри одной истории: он
    зависит от состава видимых событий, а трассировка к исходной
    записи обязана пережить другой срез и другую сборку.
    """

    client_id: str
    event_time: datetime
    source: str
    stable_event_index: int
    event_id: str
    values: dict[str, object]
    calendar: tuple[float, ...]
    timing: time_module.EventTiming
    derived: list[formulas_module.Derived] = field(default_factory=list)
    # Наблюдался ли час события. Признак достоверности времени, а
    # не значение: в model_values он не входит.

    def as_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "event_time": self.event_time,
            "source": self.source,
            "stable_event_index": self.stable_event_index,
            "event_id": self.event_id,
            "values": dict(self.values),
            "calendar": list(self.calendar),
            "timing": self.timing.as_dict(),
            "derived": [item.as_dict() for item in self.derived],
        }

    def model_values(self) -> dict[str, object]:
        """
        Всё, что событие отдаёт модели под объявленными ключами:
        значения, интервалы и удавшиеся расчёты.
        """

        out: dict[str, object] = dict(self.values)

        out.update(self.timing.model_values())

        for item in self.derived:
            if item.value is not None:
                out[item.key] = item.value

        return out


@dataclass
class SemanticHistory:
    client_id: str
    client_idx: int
    cutoff: datetime
    events: list[SemanticEvent]
    profile: dict
    profile_meta: dict
    coverage: list
    entities: list
    relationship: object
    products: dict
    product_ages: dict
    activity: list
    activity_summary: dict
    chains: list
    relations: list
    limitations: list[str]

    @property
    def n_events(self) -> int:
        return len(self.events)

    def summary(self) -> dict:
        return {
            "client_id": self.client_id,
            "cutoff": self.cutoff,
            "events": self.n_events,
            "keys_used": len({key for item in self.events for key in item.model_values()}),
            "profile": self.profile_meta,
            "activity": self.activity_summary,
            "chains": chains_module.chain_summary(self.chains),
            "relations": len(self.relations),
            "product_ages": self.product_ages,
            "limitations": self.limitations,
        }


def _semantic_values(row: dict, source: str, refs: LocalRefs) -> tuple[dict[str, object], list[str]]:
    """
    Значения одного события под смысловыми ключами.

    Состав берётся у модельной проекции: сюда попадает ровно то,
    что разрешено пересекать границу модели.
    """

    fields = model_event(row, refs).fields

    values: dict[str, object] = {}

    reference_names = {name for name, _prefix in ENTITY_REFS.values()}

    for name, value in fields.items():

        if name == EVENT_TYPE_FIELD:
            values[ENVELOPE_KEYS[EVENT_TYPE_FIELD].key] = value
            continue

            continue

        if name in reference_names:
            values[REFERENCE_KEYS[name].key] = value
            continue

        if name in DYNAMIC_FIELDS:
            # Смысл задаёт field_name, разбор ниже.
            continue

        values[key_for(name, source).key] = value

    changed, notes = _profile_change_values(fields)

    values.update(changed)

    return values, notes


def _profile_change_values(fields: dict) -> tuple[dict[str, object], list[str]]:
    """
    Прежнее и новое значение изменившегося поля профиля в его
    собственном смысле.

    Доход остаётся числом, город категорией, число детей числом.
    Один текстовый ключ на всё это был бы неправдой: BPE резал бы
    доход так же, как название города.
    """

    name = fields.get("field_name")

    if name is None:
        return {}, []

    old_key, new_key = profile_change_keys(name)

    out: dict[str, object] = {}
    notes: list[str] = []

    for key, raw in ((old_key, fields.get("old_value")), (new_key, fields.get("new_value"))):

        if raw is None:
            continue

        value, note = _typed_profile_value(key, raw, name)

        if note is not None:
            notes.append(note)
            continue

        out[key.key] = value

    return out, notes


def _typed_profile_value(key: SemanticKey, raw: object, field_name: str) -> tuple[object, str | None]:
    """
    Значение профиля в виде своего ключа. Неразобранное число не
    подменяется текстом и признаком не становится.
    """

    if key.kind != NUMERIC:
        return raw, None

    text = str(raw)

    try:
        return int(text), None
    except ValueError:
        pass

    try:
        return float(text), None
    except ValueError:
        return None, f"значение профиля {field_name} не разобрано как число: {text!r}"


def _product_values(row: dict, products: dict) -> tuple[dict[str, object], str | None]:
    """
    Расшифровка продукта по справочнику: название той версии
    условий, которую называет само событие, и прежний продукт
    перехода.

    Идентификатор каталога наружу не выходит. Версии, неизвестной
    справочнику к cutoff, название не придумывается.
    """

    out: dict[str, object] = {}

    product_id = row.get("product_id")

    if product_id:

        entry = products.get(product_key(product_id, row.get("product_version"), row.get("tariff_version")))

        if entry is None or entry.get("state") != "known":
            return out, (entry or {}).get("state", "not_in_catalogue")

        if entry.get("product_name") is not None:
            out[PRODUCT_KEYS["product_name"].key] = entry["product_name"]

    previous_id = row.get("previous_product_id")

    if previous_id:

        entry = products.get(product_key(previous_id))

        if entry is None or entry.get("state") != "known":
            return out, (entry or {}).get("state", "not_in_catalogue")

        if entry.get("product_name") is not None:
            out[PRODUCT_KEYS["previous_product_name"].key] = entry["product_name"]

        if entry.get("product_family") is not None:
            out[PRODUCT_KEYS["previous_product_family"].key] = entry["product_family"]

    return out, None


def _relation_values(relation: chains_module.Relation) -> dict[str, object]:
    """
    Смысл связи вместо идентификатора события-причины.
    """

    out: dict[str, object] = {
        RELATION_KEYS["related_event_type"].key: relation.related_event_type,
        RELATION_KEYS["relation_type"].key: relation.relation_type,
    }

    # Сам факт связи известен всегда, а её длительность — нет.
    # Неизвестный интервал признаком не становится: ни
    # отрицательным числом, ни нулём.
    if relation.days_since_related_event is not None:
        out[RELATION_KEYS["days_since_related_event"].key] = relation.days_since_related_event

    if relation.same_merchant is not None:
        out[RELATION_KEYS["same_merchant"].key] = relation.same_merchant

    return out


def _profile_values(profile: dict | None) -> dict[str, object]:

    if profile is None:
        return {}

    return {
        key.key: profile[name]
        for name, key in PROFILE_KEYS.items()
        if profile.get(name) is not None
    }


def semantic_as_of(
    store: CanonicalStore,
    client: str | int,
    cutoff: datetime,
    merchants: MerchantCatalog | None = None,
) -> SemanticHistory:
    """
    Смысловая история клиента на дату.
    """

    history: ClientHistory = history_as_of(store, client, cutoff)

    rows = history.events.to_pylist()

    catalog = merchants or MerchantCatalog(None)

    refs = LocalRefs()

    limitations = list(history.limitations)

    # --- значения, расшифровка точки и продукта ---

    values_per_event: list[dict[str, object]] = []
    merchant_notes: set[str] = set()
    product_notes: set[str] = set()
    profile_notes: set[str] = set()

    for row in rows:

        values, notes = _semantic_values(row, row["source"], refs)

        profile_notes.update(notes)

        decoded, reason = catalog.decode(row.get("outlet_id"))

        values.update(decoded)

        if reason is not None and row.get("outlet_id") is not None:
            merchant_notes.add(reason)

        product, product_reason = _product_values(row, history.products)

        values.update(product)

        if product_reason is not None:
            product_notes.add(product_reason)

        values_per_event.append(values)

    for note in sorted(merchant_notes):
        limitations.append(f"расшифровка точки недоступна: {note}")

    for note in sorted(product_notes):
        limitations.append(f"расшифровка продукта недоступна: {note}")

    limitations.extend(sorted(profile_notes))

    # Строка события вместе с её смыслом. Дальше считают по ней:
    # сущность названа локальной ссылкой, а не сырым
    # идентификатором, поэтому в лимит и в сравнение точек
    # идентификатор не попадает.
    resolved = [{**row, **values} for row, values in zip(rows, values_per_event)]

    # --- связи ---

    relations = chains_module.relations(resolved)

    relation_of_event = {item.stable_event_index: item for item in relations}

    for index, row in enumerate(rows):

        relation = relation_of_event.get(row["stable_event_index"])

        if relation is not None:
            values_per_event[index].update(_relation_values(relation))

    # --- время, формулы, календарь ---

    observed_start = history.relationship.observed_start

    timings = time_module.timings(rows, observed_start)

    calendars = calendar_features([row["event_time"] for row in rows]) if rows else []

    profile_values = _profile_values(history.profile)

    events: list[SemanticEvent] = []
    past_amounts: dict[str, list[float]] = {}
    limits: dict[str, dict] = {}

    # ОГРАНИЧЕНИЕ, которое надо знать читателю признаков.
    #
    # Анкета клиента теперь одна на всю историю — итоговая, на
    # границу выгрузки. Доход операции двухлетней давности
    # делится на СЕГОДНЯШНИЙ доход, потому что вчерашнего в
    # данных больше нет. Раньше здесь скользил курсор версий и
    # брал ту, что действовала в момент операции.
    #
    # На конечном срезе группы это честно: итоговая анкета и
    # есть анкета на срез. На более раннем срезе было бы знание
    # из будущего, и потому датасет ранние срезы запрещает.
    acting_profile = history.profile

    for index, row in enumerate(resolved):

        # Лимит берётся из того, что видно К ЭТОМУ моменту:
        # договор, открытый позже, к прошлой операции не
        # применяется.
        derived = formulas_module.amount_ratios(row, acting_profile, limits)

        formulas_module.note_limit(limits, row)

        same_type = past_amounts.setdefault(row["event_type"], [])

        derived.append(formulas_module.deviation_from_past(row, list(same_type)))

        if row.get("amount") is not None:
            same_type.append(float(row["amount"]))

        merchant_ref = values_per_event[index].get(REFERENCE_KEYS["outlet_ref"].key)

        for item in derived:
            if item.key.startswith("merchant"):
                item.derived_from += formulas_module.merchant_provenance(merchant_ref)

        events.append(
            SemanticEvent(
                client_id=history.client_id,
                event_time=row["event_time"],
                source=row["source"],
                stable_event_index=row["stable_event_index"],
                event_id=row["event_id"],
                values=values_per_event[index],
                calendar=tuple(float(value) for value in calendars[index]) if len(calendars) else (),
                timing=timings[index],
                derived=derived,
            )
        )

    # --- активность и цепочки ---

    months = activity_module.activity_months(
        rows, history.coverage, observed_start, cutoff, BASE_SOURCES
    )

    return SemanticHistory(
        client_id=history.client_id,
        client_idx=history.client_idx,
        cutoff=cutoff,
        events=events,
        profile=profile_values,
        profile_meta=history.profile_meta,
        coverage=history.coverage,
        entities=history.entities,
        relationship=history.relationship,
        products=history.products,
        product_ages=time_module.product_ages(resolved, cutoff),
        activity=months,
        activity_summary=activity_module.activity_summary(months),
        chains=chains_module.chains(resolved),
        relations=relations,
        limitations=limitations,
    )


def open_merchants(raw_dir: Path | None) -> MerchantCatalog:
    return MerchantCatalog.open(raw_dir)


def open_products(raw_dir: Path | None):
    """
    Справочник продуктов выгрузки. Без него названия продукта не
    придумывается: остаётся код и семейство.
    """

    if raw_dir is None:
        return None

    path = Path(raw_dir) / "catalog" / "products.parquet"

    return pq.read_table(path) if path.exists() else None


__all__ = [
    "SEMANTIC_VERSION",
    "SemanticEvent",
    "SemanticHistory",
    "open_merchants",
    "open_products",
    "semantic_as_of",
]
