from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# ИДЕЯ
# ============================================================
#
# Всё, что попадает на диск, проходит через этот модуль, чтобы
# два запуска на одних данных давали одинаковые байты:
#
#   - JSON с sort_keys, фиксированным отступом, '\n' на любой ОС;
#   - numpy-скаляры и datetime приводятся к обычным типам явно;
#   - parquet пишется из pyarrow с явной схемой и без метаданных
#     pandas (там лежит версия pandas);
#   - ни путей, ни времени запуска в содержимом.
# ============================================================


def json_ready(value: Any) -> Any:
    """
    Рекурсивно приводит объект к тому, что json.dumps сериализует
    одинаково между запусками.
    """

    if value is None or isinstance(value, (bool, str)):
        return value

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number != number:
            raise ValueError("NaN в artifact недопустим: пропуски хранятся отдельно")
        return number

    if isinstance(value, (datetime, date, np.datetime64)):
        if isinstance(value, np.datetime64):
            value = value.astype("datetime64[us]").astype(datetime)
        return value.isoformat()

    if isinstance(value, Path):
        raise ValueError("пути не пишутся в artifacts")

    if is_dataclass(value) and not isinstance(value, type):
        return json_ready(asdict(value))

    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]

    if isinstance(value, (set, frozenset)):
        return [json_ready(item) for item in sorted(value)]

    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]

    raise TypeError(f"не знаю, как сериализовать {type(value).__name__}")


def dumps_json(value: Any) -> str:
    return json.dumps(json_ready(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # newline="\n": иначе на Windows байты отличаются от Linux.
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_json(path: Path, value: Any) -> None:
    write_text(path, dumps_json(value))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_table(path: Path, table: pa.Table, schema: pa.Schema | None = None) -> None:
    """
    Одна запись, явная схема, без метаданных.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    if schema is not None:
        table = table.select(schema.names).cast(schema)

    table = table.replace_schema_metadata(None)

    pq.write_table(table, path, compression="zstd")


class TableWriter:
    """
    Потоковая запись батчами с фиксированной схемой. Пустой файл
    всё равно создаётся: набор файлов не зависит от данных.
    """

    def __init__(self, path: Path, schema: pa.Schema):
        self.path = Path(path)
        self.schema = schema
        self.rows = 0
        self._writer: pq.ParquetWriter | None = None

    def write(self, table: pa.Table) -> None:

        if table.num_rows == 0:
            return

        table = table.select(self.schema.names).cast(self.schema).replace_schema_metadata(None)

        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")

        self._writer.write_table(table)
        self.rows += table.num_rows

    def close(self) -> int:

        if self._writer is None:
            write_table(self.path, self.schema.empty_table())
        else:
            self._writer.close()

        return self.rows


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def sha256_ints(values: list[int]) -> str:
    return sha256_bytes(",".join(str(value) for value in sorted(values)).encode("utf-8"))


def tree_digests(root: Path) -> dict[str, str]:
    """
    sha256 каждого файла под каталогом, ключ это относительный
    путь с прямыми слэшами.
    """

    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# ============================================================
# MARKDOWN
# ============================================================


def _md_table(rows: list[list[Any]], header: list[str]) -> str:

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]

    for row in rows:
        lines.append("| " + " | ".join(_md_cell(cell) for cell in row) + " |")

    return "\n".join(lines) + "\n"


def _md_cell(value: Any) -> str:

    if value is None:
        return "—"

    if isinstance(value, bool):
        return "да" if value else "нет"

    if isinstance(value, float):
        return f"{value:.4f}"

    if isinstance(value, dict):
        if not value:
            return "—"
        # Вложенный словарь в ячейке нечитаем: показываем размер.
        if any(isinstance(item, (dict, list, tuple)) for item in value.values()):
            return f"{len(value)} записей"
        return ", ".join(f"{key}={_md_cell(item)}" for key, item in sorted(value.items()))

    if isinstance(value, (list, tuple)):
        return ", ".join(_md_cell(item) for item in value)

    return str(value)


def render_validation_md(report: dict) -> str:

    out: list[str] = []

    out.append("# Проверка RAW\n")
    out.append(f"Статус: **{report['status']}**. Проверок: {len(report['checks'])}, "
               f"не пройдено: {sum(1 for c in report['checks'] if c['status'] == 'failed')}.\n")

    out.append("## Проверки\n")
    out.append(
        _md_table(
            [
                [
                    check["name"],
                    check["status"],
                    "да" if check["hard"] else "нет",
                    check.get("details", {}).get("violations", "—"),
                ]
                for check in report["checks"]
            ],
            ["проверка", "статус", "останавливает", "нарушений"],
        )
    )

    out.append("\nПодробности каждой проверки лежат в `validation_report.json`.\n")

    summary = report.get("summary", {})

    if summary:
        out.append("\n## Сводка\n")

        for section, content in sorted(summary.items()):

            out.append(f"\n### {section}\n")

            if isinstance(content, dict):
                out.append(_md_table([[key, value] for key, value in sorted(content.items())], ["ключ", "значение"]))
            else:
                out.append(f"{_md_cell(content)}\n")

    return "\n".join(out)
