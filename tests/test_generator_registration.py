from __future__ import annotations

import json
from datetime import datetime, timedelta

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from src.generator import config, emit
from src.generator.params.population import PopulationParams


# ============================================================
# ИДЕЯ
# ============================================================
#
# Персонаж, приходящий в банк внутри окна, приходит не позже
# registration_end группы (config.DATASETS). Раньше дата прихода
# раскладывалась до горизонта планирования, и пришедший после
# конца выгрузки в неё не попадал: клиентов было меньше, чем
# объявлено (train 963 из 1000). Проверяется, что:
#
#   - с границей на конце окна в выгрузке каждый персонаж — с
#     анкетой и событиями;
#   - приход в банк не позже чем за registration_margin_days до
#     границы, у каждого есть хотя бы месяц истории;
#   - сами персонажи от границы не меняются: у общих клиентов
#     прежние дата рождения, пол и город;
#   - граница записана в паспорт выгрузки, а у групп DATASETS она
#     равна концу окна.
# ============================================================


CLIENTS = 40
COMMUNITY = 4

START = datetime(2024, 1, 1)
END = datetime(2024, 7, 1)


def generate(out, registration_end: datetime | None) -> dict:

    return emit.generate_dataset(
        total_clients=CLIENTS,
        out_dir=out,
        seed=300,
        world_seed=42,
        history_start=START,
        history_end=END,
        registration_end=registration_end,
        workers=1,
        community_size=COMMUNITY,
        quiet=True,
    )


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict:
    """
    Обе выгрузки в ОДНОМ процессе, без границы первой: персоны
    кэшируются по состоянию розыгрыша, и граница обязана входить в
    его ключ — иначе вторая выгрузка получила бы персон первой.
    """

    base = tmp_path_factory.mktemp("registration")

    generate(base / "planning", None)
    generate(base / "window", END)

    return {"planning": base / "planning", "window": base / "window"}


def clients(directory, table: str) -> set[str]:
    column = pq.read_table(directory / f"{table}.parquet", columns=["client_id"])["client_id"]
    return set(pc.unique(column).to_pylist())


def test_with_the_window_as_the_bound_every_persona_is_a_client(runs):

    # Без границы часть персонажей приходит после конца окна.
    assert len(clients(runs["planning"], "profile")) < CLIENTS

    assert len(clients(runs["window"], "profile")) == CLIENTS

    # У каждого клиента ленты есть анкета. Обратное не обязано:
    # давний клиент, ушедший из банка ещё до окна, может не оставить
    # за короткое окно ни одной строки — банк держит его в базе, а
    # событий у него нет (Generator V1, behaviour/engagement).
    assert clients(runs["window"], "events") <= clients(runs["window"], "profile")


def test_newcomers_arrive_at_least_a_margin_before_the_bound(runs):

    margin = timedelta(days=PopulationParams().registration_margin_days)

    arrivals = [
        datetime.fromisoformat(item["event_time"].isoformat()).replace(tzinfo=None)
        for row in pq.read_table(runs["window"] / "profile.parquet").to_pylist()
        for item in row["lifelong"]
        if item["type"] == "bank_registered"
    ]

    newcomers = [moment for moment in arrivals if moment >= START]

    assert newcomers, "в окне обязан быть хоть один пришедший"
    assert all(moment <= END - margin + timedelta(days=1) for moment in newcomers)


def test_the_personas_themselves_do_not_change(runs):

    def people(directory) -> dict:
        return {
            row["client_id"]: (row["birth_date"], row["gender"], row["city"])
            for row in pq.read_table(directory / "profile.parquet").to_pylist()
        }

    planning, window = people(runs["planning"]), people(runs["window"])

    for client_id, person in planning.items():
        assert window[client_id] == person, client_id


def test_the_bound_is_in_the_manifest_and_each_group_ends_its_window(runs):

    manifest = json.loads((runs["window"] / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["registration_end"] == config.event_time_text(END)

    for group, settings in config.DATASETS.items():
        assert settings.registration_end == settings.history_end, group


def test_a_bound_outside_the_plan_is_refused():

    saved = (config.HISTORY_START, config.HISTORY_END, config.REGISTRATION_END)

    try:
        with pytest.raises(ValueError, match="последний приход"):
            config.activate_horizon(START, END, config.PLANNING_END + timedelta(days=1))
        with pytest.raises(ValueError, match="последний приход"):
            config.activate_horizon(START, END, START)
    finally:
        config.activate_horizon(*saved)
