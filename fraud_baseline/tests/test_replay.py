from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from fraud.labels import attach_rows
from fraud.replay import replay


def test_the_replay_lives_the_same_world_when_arrivals_end_before_the_planning_horizon(tmp_path, monkeypatch):
    """
    Повтор настоящим генератором на крошечной группе. Последний приход
    в банк у группы раньше PLANNING_END: он задаёт окно прихода клиентов
    (persona), и повтор без него прожил бы другой мир — хеши лент не
    сошлись бы, и меток не было бы.
    """

    from src.generator import config as generator_config
    from src.generator import emit

    group = "val"
    settings = replace(
        generator_config.DATASETS[group],
        clients=32,
        history_start=datetime(2025, 11, 1),
        history_end=datetime(2026, 1, 1),
        registration_end=datetime(2026, 1, 1),
    )

    assert settings.registration_end != generator_config.PLANNING_END

    monkeypatch.setitem(generator_config.DATASETS, group, settings)

    raw = tmp_path / "raw"
    horizon = (generator_config.HISTORY_START, generator_config.HISTORY_END, generator_config.REGISTRATION_END)

    try:
        emit.generate_group(group, workers=1, quiet=True, root=raw)

        captured, digests = replay(group, workers=1, raw_dir=raw)
    finally:
        generator_config.activate_horizon(*horizon)

    assert digests, "повтор без единой ленты ничего не проверяет"

    # Тот же набор клиентов и тот же хеш ленты каждого, что в RAW;
    # отказ — исключение.
    attach_rows(raw / group / "events.parquet", captured, digests)
