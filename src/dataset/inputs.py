from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.preprocessing.artifacts import dumps_json, read_json, sha256_bytes, sha256_file
from src.preprocessing.semantic.keys import KEYS_VERSION
from src.preprocessing.semantic.as_of import SEMANTIC_VERSION
from src.preprocessing.projection import PROJECTION_VERSION
from src.preprocessing.settings import GroupWindow
from src.preprocessing.corpus import CORPUS_MANIFEST_FILE
from src.preprocessing.corpus import STAGE as CORPUS_STAGE
from src.tokenization.contract import CONFIG_FILE, FIT_MANIFEST_FILE
from src.tokenization.corpus import CorpusError, GroupCorpus
from src.tokenization.layout import FrozenArtifacts, LayoutError
from src.tokenization.run import frozen_config
from src.tokenization.schema import SemanticSchema
from src.tokenization.settings import ConfigError as TokenizerConfigError

from .settings import DatasetConfig
from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


# ============================================================
# ИДЕЯ
# ============================================================
#
# У датасета три входа, и все три обязаны быть согласованы между
# собой: смысловой слой препроцессинга, фактическое разделение и
# ОДИН замороженный комплект словаря на все группы.
#
# Несогласованность здесь означает молчаливую беду дальше.
# Словарь, замороженный на другом разделении, закодирует val
# правилами чужого train; реестр смысла другой версии даст те же
# имена ключей при другом их значении. Поэтому сверка идёт до
# первой прочитанной строки данных.
#
# Правила кодирования берутся из замороженного комплекта и
# нигде не переопределяются: датасет применяет словарь, а не
# спорит с ним.
#
# Границу «что позже cutoff» держит смысловой слой, а границу
# «какой cutoff вообще разрешён» — здесь: срез позже выгрузки или
# позже конечного момента своей группы это ошибка конфигурации,
# а не повод тихо пропустить срез. От числа срезов зависит вес
# примера, и молчаливый пропуск исказил бы его.
# ============================================================


class InputsError(ValueError):
    """
    Входы датасета несовместимы между собой.
    """


@dataclass(frozen=True)
class GroupInputs:
    """
    Одна группа: её клиенты, окно, срезы и чтение истории.
    """

    group: str
    corpus: GroupCorpus
    window: GroupWindow
    cutoffs: tuple[datetime, ...]
    clients: tuple[str, ...]
    declared_eligible: int | None

    @property
    def weight(self) -> float:
        """
        Вес одного примера этой группы.

        Все клиенты группы делят один список срезов, поэтому вес
        общий: пример с двадцатью срезами не должен весить
        двадцать примеров одного.
        """

        return 1.0 / len(self.cutoffs)

    def history(self, client_id: str, cutoff: datetime):
        return self.corpus.history(client_id, cutoff)

    def as_dict(self) -> dict:
        return {
            "group": self.group,
            "window": self.window.as_dict(),
            "cutoffs": [item.isoformat() for item in self.cutoffs],
            "clients": len(self.clients),
            "weight": self.weight,
            "declared_eligible_at_final_cutoff": self.declared_eligible,
            "extract_time": self.corpus.extract_time.isoformat(),
        }


@dataclass
class DatasetInputs:
    """
    Всё, что датасет читает, и всё, чем он себя опознаёт.
    """

    artifacts: FrozenArtifacts
    tokenizer_config: object
    corpus_manifest: dict
    groups: dict[str, GroupInputs]
    processed_dir: Path
    raw_root: Path
    vocab_dir: Path
    config: DatasetConfig
    readiness: dict
    inputs_sha256: dict[str, str] = field(default_factory=dict)
    versions: dict = field(default_factory=dict)
    limitations: tuple[str, ...] = ()

    # --- тождество набора ---

    def identity(self) -> dict:
        """
        Из чего складывается имя набора.

        Ни путей, ни времени запуска, ни версий библиотек: набор,
        собранный на другой машине из тех же входов той же
        конфигурацией, обязан получить то же имя.
        """

        return {
            "format_version": FORMAT_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "schema_version": SCHEMA_VERSION,
            "config_sha256": self.config.sha256(),
            "vocabulary": {
                "artifact_id": self.artifacts.manifest["artifact_id"],
                "vocab_sha256": self.artifacts.manifest["vocab_sha256"],
                "config_sha256": self.artifacts.manifest["config_sha256"],
            },
            "inputs_sha256": dict(sorted(self.inputs_sha256.items())),
            "versions": self.versions,
            "groups": {
                name: {
                    "clients_sha256": self.corpus_manifest["groups"][name]["clients_sha256"],
                    "cutoffs": [item.isoformat() for item in group.cutoffs],
                }
                for name, group in sorted(self.groups.items())
            },
        }

    def dataset_id(self) -> str:
        return sha256_bytes(dumps_json(self.identity()).encode("utf-8"))[:12]

    # --- источники покрытия ---

    def sources(self) -> tuple[str, ...]:
        """
        Имена источников покрытия в том порядке, в каком они лягут
        колонкой доступности.

        Порядок фиксируется здесь и записывается в манифест:
        колонка чисел без списка имён рядом ничего не значит.
        """

        names: set[str] = set()

        for group in self.groups.values():

            store = group.corpus.store

            for client in store.clients:
                for row in store.coverage_rows(client["client_id"]):
                    names.add(row["source"])

        return tuple(sorted(names))

    def as_dict(self) -> dict:
        return {
            "vocabulary": {
                "artifact_id": self.artifacts.manifest["artifact_id"],
                "vocab_sha256": self.artifacts.manifest["vocab_sha256"],
                "directory": self.vocab_dir.name,
                "fit_group": self.artifacts.manifest["group"],
                "fit_end": self.artifacts.manifest["fit_end"],
            },
            "inputs_sha256": dict(sorted(self.inputs_sha256.items())),
            "versions": self.versions,
            "readiness": self.readiness,
            "groups": {name: group.as_dict() for name, group in sorted(self.groups.items())},
            "limitations": list(self.limitations),
        }

    # --- открытие ---

    @staticmethod
    def open(
        processed_dir: Path,
        raw_root: Path,
        vocab_dir: Path,
        config: DatasetConfig,
    ) -> "DatasetInputs":

        processed = Path(processed_dir)
        raw = Path(raw_root)
        target = Path(vocab_dir)

        # --- словарь ---

        try:
            artifacts = FrozenArtifacts.load(target)
            tokenizer_config = frozen_config(target, artifacts)
        except (LayoutError, TokenizerConfigError) as error:
            raise InputsError(str(error)) from error

        # --- разделение ---

        manifest_path = processed / CORPUS_STAGE / CORPUS_MANIFEST_FILE

        if not manifest_path.exists():
            raise InputsError(
                f"нет {manifest_path}: состав групп и окна целей выдаёт этап разделения, "
                "выполните его для этого набора"
            )

        manifest = read_json(manifest_path)

        if not manifest.get("usable", False):
            raise InputsError(
                "разделение объявлено непригодным: собирать примеры по нему нельзя. "
                "Причины перечислены в corpus_manifest.json"
            )

        corpus_sha256 = sha256_file(manifest_path)

        # Словарь замораживался на конкретном разделении, и оно
        # записано в его манифесте входов. Другое разделение
        # значит другой состав групп: val кодировался бы словарём
        # чужого train.
        fit_manifest = read_json(target / FIT_MANIFEST_FILE)

        declared_corpus = (fit_manifest.get("input_files") or {}).get(CORPUS_MANIFEST_FILE)

        if declared_corpus is not None and declared_corpus != corpus_sha256:
            raise InputsError(
                "словарь заморожен на другом разделении: состав групп и окна целей с тех пор "
                "изменились. Соберите словарь заново либо возьмите то разделение, на котором "
                "он построен"
            )

        DatasetInputs._check_versions(artifacts, processed, config)

        # --- группы ---

        groups: dict[str, GroupInputs] = {}
        inputs_sha256: dict[str, str] = {CORPUS_MANIFEST_FILE: corpus_sha256}

        for group in config.groups:

            entry = (manifest.get("groups") or {}).get(group)

            if entry is None:
                raise InputsError(f"группы {group!r} нет в разделении: собирать её не из чего")

            raw_dir = _resolve_raw(raw, group)

            try:
                corpus = GroupCorpus.open(processed, raw_dir, group)
            except CorpusError as error:
                raise InputsError(f"группа {group}: {error}") from error

            window = GroupWindow.from_dict(entry["window"])

            cutoffs = config.cutoffs_for(group, window.final_cutoff)

            _check_cutoffs(group, cutoffs, window, corpus.extract_time)

            groups[group] = GroupInputs(
                group=group,
                corpus=corpus,
                window=window,
                cutoffs=tuple(cutoffs),
                clients=tuple(corpus.client_ids),
                declared_eligible=entry.get("eligible_events"),
            )

            semantic_path, canonical_path = SemanticSchema.paths(processed, group)

            inputs_sha256[f"semantic/{group}/semantic_registry.json"] = sha256_file(semantic_path)
            inputs_sha256[f"canonical/{group}/field_registry.json"] = sha256_file(canonical_path)

        _check_no_overlap(groups)

        readiness = _readiness(manifest, artifacts)

        return DatasetInputs(
            artifacts=artifacts,
            tokenizer_config=tokenizer_config,
            corpus_manifest=manifest,
            groups=groups,
            processed_dir=processed,
            raw_root=raw,
            vocab_dir=target,
            config=config,
            readiness=readiness,
            inputs_sha256=inputs_sha256,
            versions=dict(artifacts.manifest["versions"]),
            limitations=tuple(manifest.get("limitations", ())),
        )

    # --- проверки ---

    @staticmethod
    def _check_versions(artifacts: FrozenArtifacts, processed: Path, config: DatasetConfig) -> None:
        """
        Словарь, реестры групп и импортированный код говорят об
        одном и том же смысле.

        Расхождение не безобидно: имена ключей остались бы теми
        же, а значение за ними стало бы другим, и заметили бы это
        только по странному поведению модели.
        """

        declared = artifacts.manifest["versions"]

        actual = {
            "keys": KEYS_VERSION,
            "semantic": SEMANTIC_VERSION,
            "projection": PROJECTION_VERSION,
        }

        for name, value in sorted(actual.items()):
            if declared.get(name) != value:
                raise InputsError(
                    f"словарь заморожен при версии {name} = {declared.get(name)!r}, "
                    f"а код работает с {value!r}: смысл ключей с тех пор мог измениться"
                )

        for group in config.groups:

            semantic_path, _canonical = SemanticSchema.paths(processed, group)

            if not semantic_path.exists():
                raise InputsError(f"нет {semantic_path}: смысловой слой группы {group} не собран")

            registry = read_json(semantic_path)

            for name, key in (("keys", "keys_version"), ("semantic", "semantic_version"),
                              ("projection", "projection_version")):

                value = (
                    registry.get("registry", {}).get(key)
                    if key == "keys_version"
                    else registry.get(key)
                )

                if value != declared.get(name):
                    raise InputsError(
                        f"группа {group}: реестр смысла версии {name} = {value!r}, "
                        f"а словарь заморожен при {declared.get(name)!r}"
                    )


def _resolve_raw(root: Path, group: str) -> Path:
    """
    Каталог RAW группы: справочники продуктов и мерчантов лежат
    там, и без них смысловой слой не расшифрует ни продукт, ни
    торговую точку.
    """

    candidates = [root / group] + ([root / "validation"] if group == "val" else [])

    found = next((path for path in candidates if path.exists()), None)

    if found is None:
        raise InputsError(f"в {root} нет подкаталога группы {group}")

    return found


def _check_cutoffs(group: str, cutoffs: list[datetime], window: GroupWindow,
                   extract_time: datetime) -> None:
    """
    Срез обязан быть внутри своей группы и внутри выгрузки.

    Пропустить негодный срез молча нельзя: вес примера равен
    единице, делённой на число срезов, и тихий пропуск сделал бы
    вес неверным у всех остальных.
    """

    for cutoff in cutoffs:

        if cutoff > extract_time:
            raise InputsError(
                f"группа {group}: срез {cutoff.isoformat()} позже границы выгрузки "
                f"{extract_time.isoformat()}: такой истории ещё не существует"
            )

        if cutoff > window.final_cutoff:
            raise InputsError(
                f"группа {group}: срез {cutoff.isoformat()} позже её конечного момента "
                f"{window.final_cutoff.isoformat()}: контекст вышел бы за окно группы"
            )

        if cutoff <= window.history_start:
            raise InputsError(
                f"группа {group}: срез {cutoff.isoformat()} не позже начала её окна "
                f"{window.history_start.isoformat()}: истории до него нет вовсе"
            )


def _check_no_overlap(groups: dict[str, GroupInputs]) -> None:
    """
    Клиент принадлежит одной группе.

    Разделение это уже проверило, но набор собирается по составу
    групп, и проверка повторяется здесь: пересечение означало бы
    обучение на клиентах, которыми потом меряют.
    """

    names = sorted(groups)

    for first in range(len(names)):
        for second in range(first + 1, len(names)):

            left, right = names[first], names[second]

            shared = sorted(set(groups[left].clients) & set(groups[right].clients))

            if shared:
                raise InputsError(
                    f"группы {left} и {right} делят {len(shared)} клиентов "
                    f"(например {', '.join(shared[:3])}): обучение шло бы на тех, кем меряют"
                )


def _readiness(manifest: dict, artifacts: FrozenArtifacts) -> dict:
    """
    Вердикт готовности: собственные причины разделения плюс те, с
    которыми заморожен словарь.

    Свой вердикт датасет не выдумывает. Диагностический вход
    остаётся диагностическим до самого конца, и причина едет
    вместе с ним.
    """

    reasons: list[str] = []

    if not manifest.get("contract_met", True):
        for item in (manifest.get("contract") or {}).get("reasons", ()):
            reasons.append(f"горизонт: {item}")

    if not manifest.get("shared_world", True):
        for item in (manifest.get("world") or {}).get("mismatches", ()) or ["общий мир групп не подтверждён"]:
            reasons.append(f"мир: {item}")

    for item in (artifacts.manifest.get("readiness") or {}).get("reasons", ()):
        if item not in reasons:
            reasons.append(f"словарь: {item}")

    return {"status": "diagnostic" if reasons else "ready", "reasons": reasons}


__all__ = [
    "CONFIG_FILE",
    "DatasetInputs",
    "GroupInputs",
    "InputsError",
]
