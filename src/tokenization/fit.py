from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.preprocessing.read import Group, ReadError
from src.preprocessing.settings import PreprocessingConfig

from .scan import FitStatistics, scan
from .schema import SemanticSchema
from .settings import METHOD_UNFITTED, TokenizerConfig


# ============================================================
# ИДЕЯ
# ============================================================
#
# Учиться разрешено ровно на одном: на обработанной группе
# train до её конечного cutoff.
#
#   data/preprocessed/train/events.parquet
#   data/raw/train/profile.parquet
#
# Все три обучающих этапа (значения, границы, BPE) читают этот
# корпус одинаково и через эту функцию. Второго способа
# добраться до данных у токенизатора нет: иначе один этап учился
# бы на одном срезе, а другой на другом, и разошлись бы они
# молча.
#
# Проверки здесь же и они останавливающие: ключ вне реестра
# или число без объявленного кодировщика это не предупреждение.
# ============================================================


class FitError(ValueError):
    """
    Учиться на этом корпусе нельзя.
    """


@dataclass
class TrainCorpus:
    """
    Прочитанный train: статистика одного прохода и описание
    того, что именно было прочитано.
    """

    group: str
    fit_end: datetime
    statistics: FitStatistics

    @property
    def content_sha256(self) -> str:
        return self.statistics.content.value()

    def as_dict(self) -> dict:
        stats = self.statistics

        return {
            "group": self.group,
            "fit_end": self.fit_end.isoformat(),
            "fit_content_sha256": self.content_sha256,
            "clients": stats.clients,
            "events": stats.events,
            "values": stats.values,
            "profiles": stats.profiles,
            "clients_without_profile": stats.clients_without_profile,
        }


def read_train(config: TokenizerConfig, schema: SemanticSchema) -> TrainCorpus:
    """
    Один проход по train до его конечного cutoff.
    """

    group = config.fit_group

    windows = PreprocessingConfig.load(None).windows

    window = windows.get(group)

    if window is None:
        raise FitError(f"для группы {group} не объявлено окно наблюдения")

    try:
        source = Group(group)
    except ReadError as error:
        raise FitError(str(error)) from error

    if not source.client_ids:
        raise FitError(f"в группе {group} нет ни одного клиента: учиться не на чем")

    _check_config(config, schema)

    statistics = scan(
        source.histories(window.final_cutoff),
        schema,
        sample_k=config.quantile_sample_k,
        distinct_cap=config.distinct_cap,
    )

    _check_scan(statistics, source, schema)

    return TrainCorpus(group=group, fit_end=window.final_cutoff, statistics=statistics)


# ------------------------------------------------------------
# ПРОВЕРКИ
# ------------------------------------------------------------


def _check_config(config: TokenizerConfig, schema: SemanticSchema) -> None:
    """
    Конфигурация обязана описывать ровно те ключи, что есть в
    реестре.
    """

    config.validate()

    numeric = set(schema.numeric_keys)
    declared = set(config.numeric_encoders)

    missing = sorted(numeric - declared)

    if missing:
        raise FitError(
            "числовые ключи без кодировщика: " + ", ".join(missing) + ". "
            "Каждое число обязано иметь объявленный способ кодирования"
        )

    extra = sorted(declared - numeric)

    if extra:
        raise FitError(
            "кодировщики объявлены для ключей, которых нет среди числовых: " + ", ".join(extra)
        )

    unknown_domain_keys = sorted(
        key for domain in config.value_domains for key in domain.keys if key not in schema.keys
    )

    if unknown_domain_keys:
        raise FitError("домены значений ссылаются на неизвестные ключи: " + ", ".join(unknown_domain_keys))

    for domain in config.value_domains:

        kinds = {schema.info(key).value_kind for key in domain.keys}

        if len(kinds) > 1:
            raise FitError(f"домен {domain.name} объединяет ключи разных видов значения: {sorted(kinds)}")

        for group, reason in schema.ambiguous:
            shared = sorted(set(domain.keys) & set(group))
            if len(shared) > 1:
                raise FitError(
                    f"домен {domain.name} объединяет ключи, объявленные несовместимыми в реестре "
                    f"({', '.join(shared)}): {reason}"
                )

    unknown_overrides = sorted(set(config.text_keys_as_categorical) - set(schema.text_keys))

    if unknown_overrides:
        raise FitError(
            "text_keys_as_categorical называет ключи, которые реестр текстом не объявлял: "
            + ", ".join(unknown_overrides)
        )


def _check_scan(stats: FitStatistics, source: Group, schema: SemanticSchema) -> None:
    """
    Прочитанное обязано совпасть с составом группы.
    """

    if stats.unknown_keys:
        raise FitError(
            "в значениях встретились ключи вне смыслового реестра: "
            + ", ".join(sorted(stats.unknown_keys))
            + ". Смысл обязан быть объявлен заранее"
        )

    if stats.clients != len(source.client_ids):
        raise FitError(
            f"клиентов прочитано {stats.clients}, а в группе {len(source.client_ids)}: "
            "молчащий клиент из группы не убирается, его история просто пуста"
        )

    if stats.unknown_event_types:
        raise FitError(
            "типы событий без объявленных полей: " + ", ".join(sorted(stats.unknown_event_types))
            + ". Реестр полей собран не на этих данных"
        )

    conflicts = {
        key: sorted(types)
        for key, types in sorted(stats.key_types.items())
        if len(types) > 1 and schema.info(key).value_kind != "numeric"
    }

    if conflicts:
        raise FitError(
            "у ключа больше одного физического типа значения: "
            + "; ".join(f"{key}: {types}" for key, types in conflicts.items())
            + ". Объединять их молча нельзя: true и \"true\" это разные значения"
        )


def unit_warnings(schema: SemanticSchema, config: TokenizerConfig) -> list[str]:
    """
    Числовые ключи, у которых единица это не единица.

    Решённое так и названо решённым: ключ без шкалы получает
    числовое [UNK] по объявленному решению, и повторять это
    предупреждением не нужно.
    """

    out: list[str] = []

    for key in schema.numeric_keys:

        info = schema.info(key)

        if info.unit is None:
            out.append(f"{key}: числовой ключ без объявленной единицы, сравнивать его не с чем")
            continue

        if info.unit in schema.keys and config.numeric_encoders[key].method != METHOD_UNFITTED:
            out.append(
                f"{key}: единица значения лежит в соседнем ключе {info.unit}, "
                "одной шкалы на разные единицы быть не может"
            )

    return out


__all__ = [
    "FitError",
    "TrainCorpus",
    "read_train",
    "unit_warnings",
]
