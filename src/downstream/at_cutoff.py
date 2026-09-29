from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator

import numpy as np

from src.dataset.sample import build_sample
from src.dataset.settings import META_FILE, DatasetConfig, dataset_dir
from src.dataset.tokenized import EVENT_TYPE_KEY, TokenizedClient, TokenizedEvent, event_type_of
from src.mlm.inputs import IGNORE, Client
from src.preprocessing.artifacts import read_json
from src.preprocessing.read import Group
from src.preprocessing.settings import GroupWindow, PreprocessingConfig
from src.temporal.position import profile_time_log, time_log
from src.tokenization.encode import encode_event, encode_profile
from src.tokenization.finalvocab import FrozenArtifacts
from src.tokenization.settings import TokenizerConfig


# ============================================================
# КЛИЕНТ НА МОМЕНТ T
# ============================================================
#
# Вход модели на произвольный момент T — те же функции, что у
# этапов 04–07, но в памяти и без файлов:
#
#   02 + анкета 01 -> Group.history(клиент, T)    события < T, анкета на T
#   -> encode_event / encode_profile (словарь 03)  как этап 04
#   -> build_sample с окном, кончающимся в T       как этап 05
#   -> time_log / profile_time_log от T             как этап 06
#   -> Client без масок                              как 07 при чтении
#
# Обрезать готовые 07_batches по T нельзя: анкета, вехи и
# временные позиции там посчитаны на конец окна группы, и будущее
# (T, конец) осталось бы во входе.
#
# Маски нет: метки -100, значения видимы — это вход для вектора
# клиента, а не для MLM. Контекст — тот же отбор истории, на
# котором модель училась (05_dataset/train/meta.json).
# ============================================================


class CutoffError(ValueError):
    """
    Вход на момент T собрать нельзя.
    """


def window_at(group: str, cutoff: datetime) -> GroupWindow:
    """
    Окно группы, обрезанное моментом T: контекст [начало, T).

    Окно маскирования покрывает весь контекст: целей здесь нет, но
    build_sample требует вложенных окон.
    """

    window = PreprocessingConfig.load(None).windows[group]

    if cutoff.tzinfo is None:
        raise CutoffError(f"момент {cutoff.isoformat()} без пояса: нужен UTC")

    moment = cutoff.astimezone(timezone.utc)

    if not window.history_start < moment <= window.final_cutoff:
        raise CutoffError(
            f"момент {moment.isoformat()} вне окна группы {group}: "
            f"({window.history_start.isoformat()}, {window.final_cutoff.isoformat()}]"
        )

    return GroupWindow(
        history_start=window.history_start,
        final_cutoff=moment,
        target_start=window.history_start,
        target_end=moment,
    )


class ClientsAtCutoff:
    """
    Клиенты группы на момент T.
    """

    def __init__(self, group: str, cutoff: datetime):

        self.group = group
        self.window = window_at(group, cutoff)
        self.cutoff = self.window.final_cutoff

        self.artifacts = FrozenArtifacts.load()
        self.limit = TokenizerConfig.load(None).max_pieces_per_value
        self.policy = DatasetConfig.load(None).context

        # Отбор истории обязан быть тем, на котором модель училась:
        # иной предел длины молча дал бы другой вход.
        trained = read_json(dataset_dir("train") / META_FILE).get("context")

        if trained != self.policy.as_dict():
            raise CutoffError(
                f"отбор истории {self.policy.as_dict()} не тот, на котором собран "
                f"05_dataset/train ({trained})"
            )

        self._source = Group(group)
        self._event_type_key = self.artifacts.key_id(EVENT_TYPE_KEY)

    @property
    def client_ids(self) -> list[str]:
        return self._source.client_ids

    def client(self, client_id: str) -> Client:
        """
        Один клиент на T. У клиента без событий до T вход есть, но
        модель такого не видела — вызывающий решает, брать ли его.
        """

        history = self._source.history(client_id, self.cutoff)

        events = []

        for event in history.events:

            record = encode_event(self.artifacts, event, self.limit)

            row = {"key_ids": record.key_ids, "value_ids": record.value_ids,
                   "positions": record.positions}

            events.append(
                TokenizedEvent(
                    event_time=event.event_time,
                    event_type=event_type_of(self.artifacts, row, self._event_type_key),
                    key_ids=list(record.key_ids),
                    value_ids=list(record.value_ids),
                    positions=list(record.positions),
                    calendar=list(event.calendar),
                    lifelong_source=event.lifelong_source,
                )
            )

        # Тот же устойчивый порядок, что у этапа 05.
        events.sort(key=lambda item: item.event_time)

        profile, times = encode_profile(self.artifacts, history, self.limit)

        sample = build_sample(
            artifacts=self.artifacts,
            client=TokenizedClient(
                client_id=client_id,
                events=events,
                profile_key_ids=list(profile.key_ids),
                profile_value_ids=list(profile.value_ids),
                profile_positions=list(profile.positions),
                profile_time=list(times),
            ),
            window=self.window,
            policy=self.policy,
        )

        # Время примера — datetime64 UTC без пояса; позиции считаются
        # от тех же моментов, что читает этап 06 из parquet.
        moments = sample.event_time.tolist()
        profile_moments = sample.profile_time.tolist()

        n_tokens = int(sample.key_ids.size)

        return Client(
            batch_index=0,
            client_id=client_id,
            key_ids=sample.key_ids.astype(np.int64),
            value_ids=sample.value_ids.astype(np.int64),
            positions=sample.positions.astype(np.int64),
            labels=np.full(n_tokens, IGNORE, dtype=np.int64),
            reason=[""] * n_tokens,
            event_starts=sample.event_starts.astype(np.int64),
            event_lengths=sample.event_lengths.astype(np.int64),
            event_time_log=np.asarray(time_log(client_id, moments), dtype=np.float32),
            calendar=sample.calendar.reshape(-1, 6),
            event_time=[moment.replace(tzinfo=timezone.utc) for moment in moments],
            profile_key_ids=sample.profile_key_ids.astype(np.int64),
            profile_value_ids=sample.profile_value_ids.astype(np.int64),
            profile_positions=sample.profile_positions.astype(np.int64),
            profile_time_log=np.asarray(
                profile_time_log(client_id, profile_moments, self.cutoff), dtype=np.float32
            ),
        )

    def clients(self, client_ids: list[str] | None = None) -> Iterator[Client]:
        """
        Клиенты по одному, в порядке client_ids (по умолчанию — все
        клиенты анкеты группы).
        """

        for client_id in client_ids if client_ids is not None else self.client_ids:
            yield self.client(client_id)


__all__ = ["ClientsAtCutoff", "CutoffError", "window_at"]
