from __future__ import annotations

import copy
from datetime import datetime, timezone

import pyarrow.parquet as pq
import pytest

from src.generator import emit
from src.preprocessing.canonical.build import build_group
from src.preprocessing.read import Group
from src.preprocessing.settings import PreprocessingConfig
from src.tokenization.scan import scan
from src.tokenization.schema import SemanticSchema
from src.tokenization.settings import TokenizerConfig


# ============================================================
# ИДЕЯ
# ============================================================
#
# Fit читает train одним проходом, и этот проход обязан дать то же,
# что и чтение клиента по одному:
#
#   - histories() без списка клиентов обходит ленту курсорами по
#     группам строк (каждая читается один раз), а истории те же, что
#     у history(client_id): события, их значения в том же порядке
#     ключей, календарь, анкета, вехи и ограничения. В том числе у
#     клиента, разрезанного границей группы строк;
#   - лента, где клиенты внутри группы строк не по возрастанию, идёт
#     прежним путём — и истории те же;
#   - scan только читает значения событий: копию model_values он не
#     делает и словари событий не меняет.
# ============================================================


CUTOFF = datetime(2024, 4, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict:
    """
    Небольшой мир генератора, препроцессированный, и та же лента,
    переписанная мелкими группами строк: клиенты режутся границами.
    """

    base = tmp_path_factory.mktemp("fit-scan")

    raw = base / "raw"

    emit.generate_dataset(
        total_clients=12, out_dir=raw, seed=77, world_seed=42,
        history_start=datetime(2024, 1, 1), history_end=datetime(2024, 5, 1),
        workers=1, community_size=4, quiet=True,
    )

    whole = base / "whole"
    build_group(raw, whole, PreprocessingConfig.load(None), "train")

    table = pq.read_table(whole / "events.parquet")

    ids = table.column("client_id").to_pylist()

    def by_client(reverse: bool):
        # Клиенты по id, строки клиента — в прежнем порядке.
        clients = sorted(set(ids), reverse=reverse)
        return table.take([index for client in clients for index, value in enumerate(ids) if value == client])

    # Мелкие группы строк по возрастанию id: клиенты режутся границами.
    small = base / "small"
    small.mkdir()
    pq.write_table(by_client(False), small / "events.parquet", row_group_size=97)

    # Клиенты внутри группы строк по убыванию: прежний путь.
    shuffled = base / "shuffled"
    shuffled.mkdir()
    pq.write_table(by_client(True), shuffled / "events.parquet")

    return {"profile": raw / "profile.parquet", "whole": whole, "small": small, "shuffled": shuffled}


def records(histories) -> list:
    return [
        (h.client_id, h.has_profile, list(h.profile.items()), h.limitations, h.lifelong,
         [(e.client_id, e.event_time, e.source, list(e.values.items()), e.calendar, e.lifelong_source)
          for e in h.events])
        for h in histories
    ]


def test_streamed_histories_are_the_histories_of_each_client(world):

    reference = Group("train", directory=world["whole"], profile_path=world["profile"])
    expected = records(reference.history(client, CUTOFF) for client in reference.client_ids)

    assert sum(len(item[5]) for item in expected) > 0

    for name in ("whole", "small"):

        group = Group("train", directory=world[name], profile_path=world["profile"])

        assert records(group.histories(CUTOFF)) == expected, name
        assert group._sorted_runs, name

    small = pq.ParquetFile(world["small"] / "events.parquet")
    edges = [small.read_row_group(index, columns=["client_id"]).column("client_id") for index in
             range(small.num_row_groups)]

    # Проверка не вырождена: хоть один клиент лежит на двух группах строк.
    assert any(edges[index][-1] == edges[index + 1][0] for index in range(len(edges) - 1))


def test_a_tape_without_sorted_clients_takes_the_old_path(world):

    reference = Group("train", directory=world["whole"], profile_path=world["profile"])
    expected = records(reference.history(client, CUTOFF) for client in reference.client_ids)

    group = Group("train", directory=world["shuffled"], profile_path=world["profile"])

    assert records(group.histories(CUTOFF)) == expected
    assert not group._sorted_runs


def test_scan_does_not_change_the_values_of_events(world):

    config = TokenizerConfig.load(None)
    group = Group("train", directory=world["whole"], profile_path=world["profile"])

    histories = list(group.histories(CUTOFF))
    before = copy.deepcopy([[list(event.values.items()) for event in item.events] for item in histories])

    scan(histories, SemanticSchema.open(), sample_k=config.quantile_sample_k, distinct_cap=config.distinct_cap,
         splits={key: spec.split_by for key, spec in config.numeric_encoders.items() if spec.split_by})

    assert [[list(event.values.items()) for event in item.events] for item in histories] == before
