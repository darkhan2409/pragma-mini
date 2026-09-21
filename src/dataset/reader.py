from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq

from src.preprocessing.artifacts import dumps_json, read_json, sha256_bytes, sha256_file

from .collate import Batch, collate
from .sample import Sample, sample_from_row
from .storage import (
    BUILDING_SUFFIX,
    INDEX_FILE,
    MANIFEST_FILE,
    REPLACED_SUFFIX,
    SHARDS_DIR,
    StorageError,
)
from .version import FORMAT_VERSION


# ============================================================
# ИДЕЯ
# ============================================================
#
# Чтение по частям и в объявленном порядке.
#
# Единица чтения это группа строк, а не файл: один пример
# длинной истории занимает сотни тысяч чисел, и «прочитать
# shard» означало бы держать в памяти сотни мегабайт ради
# batch из четырёх примеров.
#
# Порядок val и test фиксирован навсегда: оценка обязана мерить
# одной линейкой. Порядок train перемешивается по seed и номеру
# эпохи, двумя уровнями — сначала группы строк, потом строки
# внутри окна. Полная перестановка по всему набору потребовала
# бы держать его целиком.
#
# Группы никогда не смешиваются в одном batch: это разные
# клиенты с разными окнами целей.
# ============================================================


# Сколько групп строк держать в окне перемешивания.
SHUFFLE_WINDOW = 8


class DatasetError(ValueError):
    """
    Набор открыть нельзя.
    """


@dataclass(frozen=True)
class ShardPiece:
    path: Path
    row_group: int


class Dataset:
    """
    Готовый набор: манифест, указатель и чтение по частям.
    """

    def __init__(self, directory: Path, manifest: dict):
        self.directory = Path(directory)
        self.manifest = manifest
        self._index = None

    # --- открытие ---

    @staticmethod
    def open(directory: Path, artifacts=None, verify_files: bool = True) -> "Dataset":
        """
        Открывает набор, проверяя его целиком.

        Порядок проверок от дешёвых к дорогим, и самая дорогая —
        отпечатки файлов — последняя. Пропустить её можно только
        осознанно: набор, собранный наполовину, обязан выглядеть
        сломанным, а не рабочим.
        """

        directory = Path(directory)

        name = directory.name

        if name.endswith(BUILDING_SUFFIX) or name.endswith(REPLACED_SUFFIX):
            raise DatasetError(
                f"{directory} это каталог незавершённой сборки, а не готовый набор"
            )

        path = directory / MANIFEST_FILE

        if not path.exists():
            raise DatasetError(
                f"нет {path}: набор не собран или сборка не завершилась. Готовый набор всегда "
                "несёт манифест"
            )

        manifest = read_json(path)

        if manifest.get("format_version") != FORMAT_VERSION:
            raise DatasetError(
                f"набор собран в формате {manifest.get('format_version')!r}, "
                f"а код читает {FORMAT_VERSION!r}"
            )

        stored = manifest.get("dataset_id")

        if stored != name:
            raise DatasetError(
                f"каталог называется {name!r}, а манифест объявляет набор {stored!r}"
            )

        actual = sha256_bytes(dumps_json(manifest["identity"]).encode("utf-8"))[:12]

        if actual != stored:
            raise DatasetError(
                f"манифест изменён после сборки: имя набора {stored!r}, а по его входам {actual!r}"
            )

        files = manifest.get("files_sha256") or {}

        for relative in sorted(files):
            if not (directory / relative).exists():
                raise DatasetError(f"файл {relative} пропал из набора {stored}")

        unlisted = sorted(
            item.relative_to(directory).as_posix()
            for item in (directory / SHARDS_DIR).glob("*.parquet")
            if item.relative_to(directory).as_posix() not in files
        ) if (directory / SHARDS_DIR).exists() else []

        if unlisted:
            raise DatasetError(
                f"в наборе {stored} лежат неучтённые файлы: {', '.join(unlisted[:3])}. "
                "Манифест обязан называть всё содержимое"
            )

        dataset = Dataset(directory, manifest)

        rows = dataset.index.num_rows
        declared = int(manifest["counts"]["samples"])

        if rows != declared:
            raise DatasetError(
                f"в указателе {rows} примеров, а манифест объявляет {declared}"
            )

        if verify_files:
            for relative, digest in sorted(files.items()):
                if sha256_file(directory / relative) != digest:
                    raise DatasetError(
                        f"файл {relative} изменился после сборки: читать этот набор нельзя"
                    )

        if artifacts is not None:
            expected = manifest["identity"]["vocabulary"]["artifact_id"]
            if artifacts.manifest["artifact_id"] != expected:
                raise DatasetError(
                    f"набор собран словарём {expected}, а передан {artifacts.manifest['artifact_id']}"
                )

        return dataset

    # --- указатель ---

    @property
    def index(self):

        if self._index is None:
            self._index = pq.read_table(self.directory / INDEX_FILE)

        return self._index

    @property
    def dataset_id(self) -> str:
        return self.manifest["dataset_id"]

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(self.manifest["counts"]["by_group"])

    def shards_of(self, group: str) -> list[str]:

        declared = self.manifest["shards"]

        return sorted(name for name in declared if declared[name]["group"] == group)

    def pieces(self, group: str) -> list[ShardPiece]:
        """
        Группы строк этой группы в порядке записи.
        """

        out: list[ShardPiece] = []

        for name in self.shards_of(group):

            path = self.directory / SHARDS_DIR / f"{name}.parquet"

            handle = pq.ParquetFile(path)

            for number in range(handle.metadata.num_row_groups):
                out.append(ShardPiece(path=path, row_group=number))

        return out

    # --- чтение ---

    def iter_samples(
        self,
        group: str,
        order: str = "fixed",
        seed: int = 0,
        epoch: int = 0,
        columns: list[str] | None = None,
    ) -> Iterator[Sample]:
        """
        Примеры группы по одному.

        order = "fixed" даёт порядок указателя, «shuffled» —
        воспроизводимую перестановку по seed и номеру эпохи.
        """

        if group not in self.groups:
            raise DatasetError(f"группы {group!r} в наборе нет: есть {list(self.groups)}")

        pieces = self.pieces(group)

        if order == "fixed":
            for piece in pieces:
                for row in _read(piece, columns):
                    yield sample_from_row(row)
            return

        if order != "shuffled":
            raise DatasetError(f"неизвестный порядок чтения {order!r}: есть fixed и shuffled")

        rng = np.random.default_rng([seed, epoch])

        shuffled = [pieces[index] for index in rng.permutation(len(pieces))]

        for start in range(0, len(shuffled), SHUFFLE_WINDOW):

            window: list[dict] = []

            for piece in shuffled[start : start + SHUFFLE_WINDOW]:
                window.extend(_read(piece, columns))

            for index in rng.permutation(len(window)):
                yield sample_from_row(window[index])

    def iter_batches(
        self,
        group: str,
        batch_size: int,
        order: str = "fixed",
        seed: int = 0,
        epoch: int = 0,
        max_batch_tokens: int | None = None,
    ) -> Iterator[Batch]:
        """
        Batch'и одной группы.

        max_batch_tokens считается ПОСЛЕ выравнивания: память
        модели определяет прямоугольник «событий × самое длинное
        событие», а не сумма настоящих токенов. Потолок закрывает
        batch раньше срока и при одном порядке всегда в одном и
        том же месте.
        """

        if batch_size < 1:
            raise DatasetError("размер batch обязан быть положительным")

        current: list[Sample] = []
        events = 0
        widest = 0

        for sample in self.iter_samples(group, order=order, seed=seed, epoch=epoch):

            width = int(sample.event_lengths.max()) if sample.n_events else 0

            if current and max_batch_tokens is not None:

                padded = (events + sample.n_events) * max(widest, width)

                if padded > max_batch_tokens:
                    yield collate(current)
                    current, events, widest = [], 0, 0

            current.append(sample)
            events += sample.n_events
            widest = max(widest, width)

            if len(current) >= batch_size:
                yield collate(current)
                current, events, widest = [], 0, 0

        if current:
            yield collate(current)


def _read(piece: ShardPiece, columns: list[str] | None) -> list[dict]:

    handle = pq.ParquetFile(piece.path)

    table = handle.read_row_group(piece.row_group, columns=columns)

    return table.to_pylist()


def open_dataset(directory: Path, artifacts=None, verify_files: bool = True) -> Dataset:

    try:
        return Dataset.open(directory, artifacts=artifacts, verify_files=verify_files)
    except StorageError as error:
        raise DatasetError(str(error)) from error


__all__ = [
    "SHUFFLE_WINDOW",
    "Dataset",
    "DatasetError",
    "ShardPiece",
    "open_dataset",
]
