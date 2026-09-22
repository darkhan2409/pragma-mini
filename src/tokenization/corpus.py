from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from src.preprocessing.artifacts import sha256_file
from src.preprocessing.canonical.build import REGISTRY_FILE as CANONICAL_REGISTRY_FILE
from src.preprocessing.canonical.build import STAGE as CANONICAL_STAGE
from src.preprocessing.history import CanonicalStore
from src.preprocessing.manifest import fingerprint_path, load_fingerprint, outputs_intact
from src.preprocessing.semantic.as_of import SemanticHistory, semantic_as_of
from src.preprocessing.semantic.build import REGISTRY_FILE as SEMANTIC_REGISTRY_FILE
from src.preprocessing.semantic.build import STAGE as SEMANTIC_STAGE
from src.preprocessing.corpus import CORPUS_MANIFEST_FILE, TRAIN_INDEX_FILE
from src.preprocessing.corpus import STAGE as CORPUS_STAGE
from src.preprocessing.corpus import CorpusError as TrainCorpusError
from src.preprocessing.corpus import TrainCorpus

from .schema import SemanticSchema


# ============================================================
# ИДЕЯ
# ============================================================
#
# Единственный вход fit: train на fit_end через разрешённый
# корпус разделения, поверх него смысловой слой.
#
# Второй реализации видимости здесь нет и быть не может. Всё,
# что знает токенизатор о том, какие строки ему разрешены,
# приходит из TrainCorpus этапа corpus: он проверяет пригодность,
# свежесть canonical, горизонт и сам индекс, а клиента выдаёт
# только из разрешённой группы.
#
# Справочники продуктов и мерчантов передаются сюда явно и
# сверяются с теми, на которых построено разделение: иначе
# название продукта и расшифровка точки пришли бы из другого
# мира, а отпечаток этого не заметил бы.
# ============================================================


READY = "ready"
DIAGNOSTIC = "diagnostic"


class CorpusError(ValueError):
    """
    Разрешённый корпус открыть нельзя.
    """


@dataclass(frozen=True)
class Readiness:
    """
    Готов набор целиком или используется как диагностика.

    Короткий горизонт и неподтверждённый общий мир не делают
    работу бессмысленной, но и молчать о них нельзя: вердикт
    едет во все манифесты токенизатора.
    """

    status: str
    reasons: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.status == READY

    def as_dict(self) -> dict:
        return {"status": self.status, "reasons": list(self.reasons)}


def _readiness(manifest: dict) -> Readiness:
    """
    Вердикт готовности по двум вердиктам разделения.

    Ограничения разделения сюда не переписываются: они и так
    едут в ограничения входа, а вердикт должен называть причину
    один раз.
    """

    reasons: list[str] = []

    if not manifest.get("contract_met", True):
        for item in (manifest.get("contract") or {}).get("reasons", ()):
            reasons.append(f"горизонт: {item}")

    if not manifest.get("shared_world", True):
        for item in (manifest.get("world") or {}).get("mismatches", ()) or ["общий мир групп не подтверждён"]:
            reasons.append(f"мир: {item}")

    if not manifest.get("usable", True):
        reasons.append("разделение объявлено непригодным")

    return Readiness(DIAGNOSTIC if reasons else READY, tuple(reasons))


def _marker(processed: Path, stage: str, group: str | None, what: str) -> dict:
    """
    Маркер этапа препроцессинга вместе с проверкой его выходов.
    """

    stored = load_fingerprint(fingerprint_path(processed, stage, group))

    if stored is None:
        raise CorpusError(f"нет маркера этапа {what}: выполните препроцессинг заново")

    if not outputs_intact(stored, processed):
        raise CorpusError(
            f"выходы этапа {what} изменились после сборки: словарь учился бы на других файлах, "
            "выполните этап заново"
        )

    return stored


class FitCorpus:
    """
    Train на fit_end со смыслом. Ничего не учит и не пишет.
    """

    def __init__(
        self,
        corpus: TrainCorpus,
        schema: SemanticSchema,
        manifest: dict,
        group: str,
        raw_dir: Path,
        processed_dir: Path,
        readiness: Readiness,
        inputs: dict[str, str],
    ):
        self.corpus = corpus
        self.schema = schema
        self.manifest = manifest
        self.group = group
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.readiness = readiness
        self.inputs = dict(inputs)

    # --- границы ---

    @property
    def fit_end(self) -> datetime:
        return self.corpus.fit_end

    @property
    def client_ids(self) -> list[str]:
        return sorted(self.corpus.client_ids)

    @property
    def declared_rows(self) -> int:
        """
        Сколько видимых строк объявило разделение.
        """

        return int(self.manifest["train_corpus"]["checksum"]["events_rows"])

    @property
    def declared_profile_rows(self) -> int:
        return int(self.manifest["train_corpus"]["checksum"]["profile_rows"])

    @property
    def content_sha256(self) -> str:
        return self.manifest["train_corpus"]["checksum"]["content_sha256"]

    @property
    def declared_clients_without_profile(self) -> int:
        return int(self.manifest["groups"][self.group]["clients_without_profile"])

    @property
    def corpus_limitations(self) -> tuple[str, ...]:
        return tuple(self.manifest.get("limitations", ()))

    # --- чтение ---

    def history(self, client_id: str) -> SemanticHistory:
        """
        Смысловая история разрешённого клиента на fit_end.
        """

        return semantic_as_of(
            self.corpus.store,
            self.corpus.require(client_id),
            self.fit_end,
        )

    def iter_histories(self) -> Iterator[SemanticHistory]:
        """
        Клиенты по одному, в устойчивом порядке. Историй в памяти
        не накапливается: статистику считают на лету.
        """

        for client_id in self.client_ids:
            yield self.history(client_id)

    # --- открытие ---

    @staticmethod
    def open(
        processed_dir: Path,
        raw_dir: Path,
        group: str = "train",
        allow_short_horizon: bool = False,
    ) -> "FitCorpus":

        processed = Path(processed_dir)
        raw = Path(raw_dir)

        corpus_dir = processed / CORPUS_STAGE
        canonical_dir = processed / CANONICAL_STAGE / group

        manifest_path = corpus_dir / CORPUS_MANIFEST_FILE

        if not manifest_path.exists():
            raise CorpusError(
                f"нет {manifest_path}: разрешённый train-корпус выдаёт этап разделения, "
                "выполните его для этого набора"
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if manifest.get("train_corpus", {}).get("group") != group:
            raise CorpusError(
                f"разрешённая для fit группа это {manifest.get('train_corpus', {}).get('group')!r}, "
                f"а запрошена {group!r}: учиться на другой группе нельзя"
            )

        # Маркеры этапов: отпечаток говорит о согласии маркера с
        # манифестом, а целы ли сами файлы, проверяют выходы.
        _marker(processed, CANONICAL_STAGE, group, f"canonical/{group}")
        _marker(processed, CORPUS_STAGE, None, "corpus")
        _marker(processed, SEMANTIC_STAGE, group, f"semantic/{group}")

        try:
            corpus = TrainCorpus.open(
                corpus_dir,
                canonical_dir,
                processed_dir=processed,
                allow_short_horizon=allow_short_horizon,
            )
        except TrainCorpusError as error:
            raise CorpusError(str(error)) from error

        schema = SemanticSchema.open(processed, group)

        semantic_path, field_path = SemanticSchema.paths(processed, group)

        inputs = {
            CORPUS_MANIFEST_FILE: sha256_file(manifest_path),
            TRAIN_INDEX_FILE: sha256_file(corpus_dir / TRAIN_INDEX_FILE),
            SEMANTIC_REGISTRY_FILE: sha256_file(semantic_path),
            CANONICAL_REGISTRY_FILE: sha256_file(field_path),
        }

        return FitCorpus(
            corpus=corpus,
            schema=schema,
            manifest=manifest,
            group=group,
            raw_dir=raw,
            processed_dir=processed,
            readiness=_readiness(manifest),
            inputs=inputs,
        )

class GroupCorpus:
    """
    Смысловая история любой группы на любой разрешённый момент.

    Ничему не учит: этим читают данные, которые кодируются уже
    замороженными артефактами. Разрешение на клиента здесь даёт
    не разделение, а принадлежность группе: val и test это её
    собственные клиенты, и границу держит cutoff.
    """

    def __init__(self, store: CanonicalStore, group: str, raw_dir: Path,
                 processed_dir: Path, client_ids: list[str], readiness: Readiness,
                 final_cutoff: datetime | None = None):
        self.store = store
        self.group = group
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.client_ids = list(client_ids)
        self.readiness = readiness
        # Конечный cutoff группы по фактическому разделению.
        # None значит, что разделения рядом нет и момент придётся
        # взять из конфигурации или назвать явно.
        self.final_cutoff = final_cutoff

    @property
    def period_end(self) -> datetime:
        return self.store.period_end

    def history(self, client_id: str, cutoff: datetime) -> SemanticHistory:
        return semantic_as_of(self.store, client_id, cutoff)

    @staticmethod
    def open(processed_dir: Path, raw_dir: Path, group: str) -> "GroupCorpus":

        processed = Path(processed_dir)
        raw = Path(raw_dir)

        canonical_dir = processed / CANONICAL_STAGE / group

        if not canonical_dir.exists():
            raise CorpusError(f"нет {canonical_dir}: группа {group} не собрана этапом canonical")

        _marker(processed, CANONICAL_STAGE, group, f"canonical/{group}")

        manifest_path = processed / CORPUS_STAGE / CORPUS_MANIFEST_FILE

        readiness = Readiness(READY, ())
        clients: list[str] | None = None
        final_cutoff: datetime | None = None

        if manifest_path.exists():

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            readiness = _readiness(manifest)

            entry = (manifest.get("groups") or {}).get(group)

            if entry is not None:
                clients = list(entry["clients"])
                final_cutoff = datetime.fromisoformat(entry["window"]["final_cutoff"])

        store = CanonicalStore(canonical_dir)

        if clients is None:
            # Разделения нет: состав группы берётся из адресной
            # книги canonical, тестовые аккаунты исключаются.
            clients = [row["client_id"] for row in store.clients]

        return GroupCorpus(
            store=store,
            group=group,
            raw_dir=raw,
            processed_dir=processed,
            client_ids=sorted(clients),
            readiness=readiness,
            final_cutoff=final_cutoff,
        )


__all__ = [
    "DIAGNOSTIC",
    "READY",
    "CorpusError",
    "FitCorpus",
    "GroupCorpus",
    "Readiness",
]
