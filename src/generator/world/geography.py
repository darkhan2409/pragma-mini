from __future__ import annotations

from dataclasses import dataclass

from .. import params as params_module
from ..rng import NS_CATALOG, keyed_rng, stable_hash, state_cache


# ============================================================
# ГЕОГРАФИЯ КАЗАХСТАНА
# ============================================================
#
# Весь Казахстан, а не несколько показательных городов: двадцать
# регионов, города республиканского значения, областные центры,
# моногорода, малые города, районные центры и сельские
# территории.
#
# Тип поселения задаёт доступность категорий, плотность точек,
# уровень цен, долю наличных, транспорт и часы работы.
# ============================================================


# (имя, регион, тип, вес населения)
SETTLEMENTS: tuple[tuple[str, str, str, float], ...] = (
    # --- города республиканского значения ---
    ("Almaty", "Almaty", "metropolis", 2200.0),
    ("Astana", "Astana", "metropolis", 1450.0),
    ("Shymkent", "Shymkent", "metropolis", 1200.0),

    # --- Алматинская область ---
    ("Konaev", "Almaty region", "regional_centre", 75.0),
    ("Kaskelen", "Almaty region", "small_town", 80.0),
    ("Talgar", "Almaty region", "small_town", 52.0),
    ("Esik", "Almaty region", "small_town", 44.0),
    ("Uzynagash", "Almaty region", "district_centre", 38.0),
    ("Sarkand", "Almaty region", "district_centre", 12.0),
    ("Almaty region rural", "Almaty region", "rural", 210.0),

    # --- Жетысу ---
    ("Taldykorgan", "Zhetisu", "regional_centre", 160.0),
    ("Tekeli", "Zhetisu", "industrial_town", 32.0),
    ("Ushtobe", "Zhetisu", "small_town", 24.0),
    ("Zharkent", "Zhetisu", "small_town", 42.0),
    ("Zhetisu rural", "Zhetisu", "rural", 155.0),

    # --- Туркестанская область ---
    ("Turkistan", "Turkistan", "regional_centre", 180.0),
    ("Kentau", "Turkistan", "industrial_town", 90.0),
    ("Saryagash", "Turkistan", "small_town", 55.0),
    ("Lenger", "Turkistan", "small_town", 28.0),
    ("Arys", "Turkistan", "small_town", 47.0),
    ("Shardara", "Turkistan", "district_centre", 35.0),
    ("Turkistan rural", "Turkistan", "rural", 420.0),

    # --- Карагандинская область ---
    ("Karaganda", "Karaganda", "major_city", 500.0),
    ("Temirtau", "Karaganda", "industrial_town", 170.0),
    ("Balkhash", "Karaganda", "industrial_town", 75.0),
    ("Saran", "Karaganda", "industrial_town", 47.0),
    ("Shakhtinsk", "Karaganda", "industrial_town", 38.0),
    ("Abai town", "Karaganda", "small_town", 26.0),
    ("Priozersk", "Karaganda", "small_town", 13.0),
    ("Karaganda rural", "Karaganda", "rural", 95.0),

    # --- Улытау ---
    ("Zhezkazgan", "Ulytau", "industrial_town", 90.0),
    ("Satpayev", "Ulytau", "industrial_town", 65.0),
    ("Karazhal", "Ulytau", "small_town", 11.0),
    ("Ulytau rural", "Ulytau", "rural", 22.0),

    # --- Актюбинская область ---
    ("Aktobe", "Aktobe", "major_city", 530.0),
    ("Kandyagash", "Aktobe", "small_town", 32.0),
    ("Khromtau", "Aktobe", "industrial_town", 29.0),
    ("Alga", "Aktobe", "small_town", 22.0),
    ("Shalkar", "Aktobe", "district_centre", 27.0),
    ("Emba", "Aktobe", "small_town", 12.0),
    ("Aktobe rural", "Aktobe", "rural", 180.0),

    # --- Атырауская область ---
    ("Atyrau", "Atyrau", "major_city", 360.0),
    ("Kulsary", "Atyrau", "industrial_town", 65.0),
    ("Makat", "Atyrau", "small_town", 18.0),
    ("Atyrau rural", "Atyrau", "rural", 180.0),

    # --- Мангистауская область ---
    ("Aktau", "Mangystau", "major_city", 260.0),
    ("Zhanaozen", "Mangystau", "industrial_town", 130.0),
    ("Beyneu", "Mangystau", "district_centre", 40.0),
    ("Mangystau rural", "Mangystau", "rural", 120.0),

    # --- Западно-Казахстанская область ---
    ("Oral", "West Kazakhstan", "major_city", 340.0),
    ("Aksai", "West Kazakhstan", "industrial_town", 45.0),
    ("Zhanibek", "West Kazakhstan", "district_centre", 9.0),
    ("Chapaev", "West Kazakhstan", "district_centre", 8.0),
    ("West Kazakhstan rural", "West Kazakhstan", "rural", 230.0),

    # --- Костанайская область ---
    ("Kostanay", "Kostanay", "major_city", 250.0),
    ("Rudny", "Kostanay", "industrial_town", 115.0),
    ("Lisakovsk", "Kostanay", "industrial_town", 40.0),
    ("Arkalyk", "Kostanay", "small_town", 28.0),
    ("Zhitikara", "Kostanay", "industrial_town", 33.0),
    ("Kostanay rural", "Kostanay", "rural", 330.0),

    # --- Акмолинская область ---
    ("Kokshetau", "Akmola", "regional_centre", 155.0),
    ("Stepnogorsk", "Akmola", "industrial_town", 48.0),
    ("Kosshy", "Akmola", "small_town", 60.0),
    ("Akkol", "Akmola", "small_town", 14.0),
    ("Atbasar", "Akmola", "small_town", 32.0),
    ("Shchuchinsk", "Akmola", "small_town", 47.0),
    ("Esil town", "Akmola", "district_centre", 13.0),
    ("Akmola rural", "Akmola", "rural", 310.0),

    # --- Северо-Казахстанская область ---
    ("Petropavl", "North Kazakhstan", "regional_centre", 220.0),
    ("Bulaevo", "North Kazakhstan", "district_centre", 10.0),
    ("Mamlyutka", "North Kazakhstan", "district_centre", 8.0),
    ("Taiynsha", "North Kazakhstan", "small_town", 12.0),
    ("Sergeevka", "North Kazakhstan", "district_centre", 7.0),
    ("North Kazakhstan rural", "North Kazakhstan", "rural", 260.0),

    # --- Павлодарская область ---
    ("Pavlodar", "Pavlodar", "major_city", 360.0),
    ("Ekibastuz", "Pavlodar", "industrial_town", 150.0),
    ("Aksu town", "Pavlodar", "industrial_town", 45.0),
    ("Kachiry", "Pavlodar", "district_centre", 9.0),
    ("Pavlodar rural", "Pavlodar", "rural", 190.0),

    # --- Восточно-Казахстанская область ---
    ("Oskemen", "East Kazakhstan", "major_city", 340.0),
    ("Ridder", "East Kazakhstan", "industrial_town", 50.0),
    ("Altai", "East Kazakhstan", "industrial_town", 33.0),
    ("Serebryansk", "East Kazakhstan", "small_town", 10.0),
    ("Shemonaikha", "East Kazakhstan", "district_centre", 17.0),
    ("East Kazakhstan rural", "East Kazakhstan", "rural", 200.0),

    # --- Абайская область ---
    ("Semey", "Abai", "major_city", 350.0),
    ("Kurchatov", "Abai", "small_town", 12.0),
    ("Ayagoz", "Abai", "small_town", 35.0),
    ("Abai rural", "Abai", "rural", 210.0),

    # --- Жамбылская область ---
    ("Taraz", "Zhambyl", "major_city", 400.0),
    ("Shu", "Zhambyl", "small_town", 35.0),
    ("Karatau", "Zhambyl", "industrial_town", 30.0),
    ("Zhanatas", "Zhambyl", "industrial_town", 22.0),
    ("Merke", "Zhambyl", "district_centre", 28.0),
    ("Zhambyl rural", "Zhambyl", "rural", 320.0),

    # --- Кызылординская область ---
    ("Kyzylorda", "Kyzylorda", "major_city", 250.0),
    ("Baikonur", "Kyzylorda", "small_town", 75.0),
    ("Aral", "Kyzylorda", "small_town", 35.0),
    ("Zhosaly", "Kyzylorda", "district_centre", 32.0),
    ("Shieli", "Kyzylorda", "district_centre", 35.0),
    ("Kyzylorda rural", "Kyzylorda", "rural", 220.0),
)


DISTRICT_NAMES = (
    "centre", "north", "south", "east", "west",
    "old_town", "new_town", "industrial", "riverside",
    "station", "market_quarter", "microdistrict_1",
    "microdistrict_2", "microdistrict_3", "suburb",
)


@dataclass(frozen=True)
class Settlement:
    settlement_id: str
    name: str
    region: str
    settlement_type: str
    population_weight: float
    districts: tuple
    regional_capital: str
    neighbours: tuple

    @property
    def is_rural(self) -> bool:
        return self.settlement_type == "rural"

    @property
    def public_name(self) -> str | None:
        """
        Название места для выгрузки.

        Сельская территория области это КОРЗИНА населения, а не
        населённый пункт: «North Kazakhstan rural» задаёт
        доступность категорий, плотность точек, долю наличных и
        часы работы, но такого места на карте нет. Наружу вместо
        него не выходит ничего: сочинять название села нечем, а
        область клиента и так названа отдельным полем.
        """

        return None if self.is_rural else self.name


@state_cache
def _geography() -> tuple:

    settings = params_module.active().geography

    by_region: dict[str, list[tuple]] = {}

    for row in SETTLEMENTS:
        by_region.setdefault(row[1], []).append(row)

    # Столица региона это самое крупное поселение региона.
    capitals = {
        region: max(rows, key=lambda item: item[3])[0]
        for region, rows in by_region.items()
    }

    settlements: list[Settlement] = []

    for name, region, kind, weight in SETTLEMENTS:

        rng = keyed_rng(NS_CATALOG, stable_hash("settlement", name) % (2 ** 31))

        low, high = settings.districts[kind]
        count = rng.integers(low, high + 1)

        districts = tuple(DISTRICT_NAMES[:count]) if count <= len(DISTRICT_NAMES) else tuple(
            DISTRICT_NAMES + tuple(f"zone_{index}" for index in range(len(DISTRICT_NAMES), count))
        )

        neighbours = tuple(
            item[0] for item in by_region[region] if item[0] != name
        )

        settlements.append(
            Settlement(
                settlement_id=f"st_{stable_hash('settlement', name) % 10 ** 9:09d}",
                name=name,
                region=region,
                settlement_type=kind,
                population_weight=weight,
                districts=districts,
                regional_capital=capitals[region],
                neighbours=neighbours,
            )
        )

    settlements.sort(key=lambda item: item.name)

    return tuple(settlements)


def settlements() -> tuple:
    return _geography()


@state_cache
def _index() -> dict:
    return {item.name: item for item in _geography()}


def by_name(name: str) -> Settlement:
    return _index()[name]


def regions() -> tuple:
    return tuple(sorted({item.region for item in _geography()}))


def metropolises() -> tuple:
    return tuple(item.name for item in _geography() if item.settlement_type == "metropolis")


def weights() -> tuple:
    return tuple(item.population_weight for item in _geography())


def names() -> tuple:
    return tuple(item.name for item in _geography())


__all__ = [
    "DISTRICT_NAMES",
    "SETTLEMENTS",
    "Settlement",
    "by_name",
    "metropolises",
    "names",
    "regions",
    "settlements",
    "weights",
]
