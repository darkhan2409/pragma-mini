from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .. import params as params_module
from .. import config
from ..rng import NS_GRAPH, keyed_rng, stable_hash


# ============================================================
# СОЦИАЛЬНЫЙ ГРАФ
# ============================================================
#
# Граф строится ДО симуляции и полностью определяется seed,
# составом сообщества и параметрами.
#
# Домохозяйство связывает аренду, коммуналку, продукты, поездки
# и финансовые шоки. Работодатель связан с доходом, собственный
# счёт в другом банке с регулярными внешними переводами.
#
# Повторные переводы идут преимущественно ЗНАКОМЫМ контрагентам.
#
# В RAW не раскрываются ни household_id, ни тип отношения: виден
# только устойчивый маскированный контрагент.
# ============================================================


_SURNAMES = (
    "Serikov", "Akhmetova", "Zhumabek", "Nurlanov", "Sagyndyk", "Bekova",
    "Tulegenov", "Amanzhol", "Karimova", "Dosmukhamed", "Zhaksylyk", "Ospanov",
    "Iskakova", "Baimurat", "Sultanbek", "Kenzhebek", "Muratova", "Aliyev",
    "Tanirbergen", "Yerzhanova", "Kaliyev", "Smagulov", "Orazbek", "Nurpeis",
    "Abenov", "Zhanibekova", "Sarsenbay", "Utegenov", "Beisembay", "Kudaibergen",
)

_GIVEN = ("A", "B", "D", "E", "G", "I", "K", "M", "N", "O", "R", "S", "T", "Z")


@dataclass(frozen=True)
class Counterpart:
    counterpart_id: str
    kind: str
    client_ordinal: int | None
    masked_name: str


@dataclass(frozen=True)
class Relationship:
    client_ordinal: int
    counterpart: Counterpart
    relation_type: str
    strength: float
    typical_frequency: float
    typical_amount_low: int
    typical_amount_high: int
    valid_from: datetime
    valid_to: datetime | None
    household_id: str | None
    # Деньги ходят в обе стороны: сколько раз в месяц эта связь
    # присылает деньги клиенту, а не получает их.
    inbound_frequency: float = 0.0

    def active_at(self, ts: datetime) -> bool:
        if ts < self.valid_from:
            return False
        return self.valid_to is None or ts < self.valid_to


@dataclass(frozen=True)
class CommunityGraph:
    community_id: int
    relationships: tuple
    households: dict

    def of(self, client_ordinal: int) -> tuple:
        return tuple(
            item for item in self.relationships if item.client_ordinal == client_ordinal
        )

    def active(self, client_ordinal: int, ts: datetime) -> tuple:
        return tuple(item for item in self.of(client_ordinal) if item.active_at(ts))

    def household_of(self, client_ordinal: int) -> str | None:
        return self.households.get(client_ordinal)

    def household_members(self, household_id: str) -> tuple:
        return tuple(
            sorted(
                ordinal
                for ordinal, value in self.households.items()
                if value == household_id
            )
        )


def masked_name(seed_text: str) -> str:
    """
    Устойчивое маскированное имя контрагента: одно и то же
    во всех операциях с ним.
    """

    value = stable_hash("counterparty", seed_text)

    given = _GIVEN[value % len(_GIVEN)]
    surname = _SURNAMES[(value // len(_GIVEN)) % len(_SURNAMES)]

    return f"{given}. {surname}"


def _external(kind: str, seed_text: str) -> Counterpart:
    return Counterpart(
        counterpart_id=f"cp_{stable_hash('external', seed_text) % 10 ** 10:010d}",
        kind=kind,
        client_ordinal=None,
        masked_name=masked_name(seed_text),
    )


def _inbound_frequency(relation_type: str, rng) -> float:
    """
    Как часто связь присылает деньги клиенту.

    Отдельный поток розыгрыша: добавление входящих переводов не
    должно сдвинуть уже существующие рёбра графа.
    """

    settings = params_module.active().relationships

    low, high = settings.inbound_frequency_per_month.get(relation_type, (0.0, 0.0))

    if high <= 0.0:
        return 0.0

    return float(rng.uniform(low, high))


def _internal(ordinal: int, client_id: str) -> Counterpart:
    return Counterpart(
        counterpart_id=f"cp_{stable_hash('internal', client_id) % 10 ** 10:010d}",
        kind="client",
        client_ordinal=ordinal,
        masked_name=masked_name(client_id),
    )


def build_graph(community_id: int, members: tuple, personas: dict) -> CommunityGraph:
    """
    Граф отношений сообщества.
    """

    settings = params_module.active().relationships

    households: dict[int, str] = {}

    pool = list(members)

    # --- домохозяйства ---

    index = 0

    while index + 1 < len(pool):

        left, right = pool[index], pool[index + 1]

        pair_rng = keyed_rng(NS_GRAPH, community_id, 1, left)

        if pair_rng.random() < settings.household_pair_share:
            household_id = f"hh_{stable_hash('household', community_id, left) % 10 ** 9:09d}"
            households[left] = household_id
            households[right] = household_id
            index += 2
            continue

        index += 1

    relationships: list[Relationship] = []

    for ordinal in members:

        persona = personas[ordinal]

        income = max(50_000, persona.true_income)

        client_rng = keyed_rng(NS_GRAPH, community_id, 2, ordinal)

        start = max(config.HISTORY_START, persona.relationship_start)

        # --- работодатель ---

        if persona.employer_id:
            relationships.append(
                Relationship(
                    client_ordinal=ordinal,
                    counterpart=_external("employer", persona.employer_id),
                    relation_type="employer",
                    strength=client_rng.uniform(*settings.strength_range["employer"]),
                    typical_frequency=0.0,
                    typical_amount_low=income,
                    typical_amount_high=income,
                    valid_from=start,
                    valid_to=None,
                    household_id=households.get(ordinal),
                )
            )

        # --- арендодатель ---

        if persona.housing_type == "rented":
            low, high = settings.amount_share_of_income["landlord"]
            relationships.append(
                Relationship(
                    client_ordinal=ordinal,
                    counterpart=_external("landlord", f"landlord:{ordinal}"),
                    relation_type="landlord",
                    strength=client_rng.uniform(*settings.strength_range["landlord"]),
                    typical_frequency=1.0,
                    typical_amount_low=int(income * low),
                    typical_amount_high=int(income * high),
                    valid_from=start,
                    valid_to=None,
                    household_id=households.get(ordinal),
                )
            )

        # --- собственный счёт в другом банке ---

        if client_rng.random() < settings.own_other_bank_share.get(persona.hcb_role, 0.8):
            low, high = settings.amount_share_of_income["own_account_other_bank"]
            frequency_low, frequency_high = settings.frequency_per_month["own_account_other_bank"]
            relationships.append(
                Relationship(
                    client_ordinal=ordinal,
                    counterpart=_external("own_account", f"own:{persona.client_id}"),
                    relation_type="own_account_other_bank",
                    strength=client_rng.uniform(*settings.strength_range["own_account_other_bank"]),
                    typical_frequency=client_rng.uniform(frequency_low, frequency_high),
                    typical_amount_low=int(income * low),
                    typical_amount_high=int(income * high),
                    valid_from=start,
                    valid_to=None,
                    household_id=households.get(ordinal),
                )
            )

        # --- супруг ---

        household_id = households.get(ordinal)

        if household_id:
            partners = [
                other
                for other, value in households.items()
                if value == household_id and other != ordinal
            ]
            for partner in partners:
                low, high = settings.amount_share_of_income["spouse"]
                frequency_low, frequency_high = settings.frequency_per_month["spouse"]
                relationships.append(
                    Relationship(
                        client_ordinal=ordinal,
                        counterpart=_internal(partner, personas[partner].client_id),
                        relation_type="spouse",
                        strength=client_rng.uniform(*settings.strength_range["spouse"]),
                        typical_frequency=client_rng.uniform(frequency_low, frequency_high),
                        typical_amount_low=int(income * low),
                        typical_amount_high=int(income * high),
                        valid_from=start,
                        valid_to=None,
                        household_id=household_id,
                        inbound_frequency=_inbound_frequency(
                            "spouse",
                            keyed_rng(NS_GRAPH, community_id, 5, ordinal * 100 + partner),
                        ),
                    )
                )

        # --- прочие связи ---

        sociality = persona.trait("sociality")

        for relation_type, (degree_low, degree_high) in settings.degree.items():

            type_rng = keyed_rng(
                NS_GRAPH, community_id, 3, ordinal * 100 + hash_index(relation_type)
            )

            span = degree_high - degree_low
            count = degree_low + int(round(span * (0.35 + 0.9 * sociality) * type_rng.random()))
            count = max(degree_low, min(degree_high, count))

            for position in range(count):

                item_rng = keyed_rng(
                    NS_GRAPH,
                    community_id,
                    4,
                    ordinal * 1000 + hash_index(relation_type) * 50 + position,
                )

                internal_share = settings.internal_share.get(relation_type, 0.15)

                candidates = [other for other in members if other != ordinal]

                if candidates and item_rng.random() < internal_share:
                    partner = candidates[item_rng.integers(0, len(candidates))]
                    counterpart = _internal(partner, personas[partner].client_id)
                else:
                    counterpart = _external(
                        relation_type, f"{relation_type}:{ordinal}:{position}"
                    )

                low, high = settings.amount_share_of_income.get(relation_type, (0.01, 0.15))
                frequency_low, frequency_high = settings.frequency_per_month.get(
                    relation_type, (0.0, 1.0)
                )

                valid_to = None

                end_share = settings.relation_end_share_per_year.get(relation_type)

                if end_share and item_rng.random() < end_share:
                    offset = item_rng.integers(60, max(61, (config.PLANNING_END - start).days))
                    valid_to = start + timedelta(days=int(offset))

                relationships.append(
                    Relationship(
                        client_ordinal=ordinal,
                        counterpart=counterpart,
                        relation_type=relation_type,
                        strength=item_rng.uniform(*settings.strength_range[relation_type]),
                        typical_frequency=item_rng.uniform(frequency_low, frequency_high),
                        typical_amount_low=int(income * low),
                        typical_amount_high=int(income * high),
                        valid_from=start,
                        valid_to=valid_to,
                        household_id=households.get(ordinal),
                        inbound_frequency=_inbound_frequency(
                            relation_type,
                            keyed_rng(
                                NS_GRAPH,
                                community_id,
                                5,
                                ordinal * 1000 + hash_index(relation_type) * 50 + position,
                            ),
                        ),
                    )
                )

    relationships.sort(
        key=lambda item: (item.client_ordinal, item.relation_type, item.counterpart.counterpart_id)
    )

    return CommunityGraph(
        community_id=community_id,
        relationships=tuple(relationships),
        households=households,
    )


_RELATION_INDEX = {
    "relative": 1,
    "friend": 2,
    "colleague": 3,
    "regular_counterparty": 4,
    "random_counterparty": 5,
}


def hash_index(relation_type: str) -> int:
    return _RELATION_INDEX.get(relation_type, 9)


__all__ = [
    "CommunityGraph",
    "Counterpart",
    "Relationship",
    "build_graph",
    "masked_name",
]
