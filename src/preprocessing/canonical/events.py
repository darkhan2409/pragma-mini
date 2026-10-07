from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from src.generator.profile import LIFELONG_SOURCE_EVENTS, LIFELONG_SOURCE_FIELD

from ..rawdata import (
    TYPE_KEY,
    RawContractError,
    RawDataset,
    RawManifest,
    event_times,
    event_types_of,
    parse_payloads,
)
from ..settings import PreprocessingConfig
from .schema import (
    ENVELOPE_NAMES,
    LIFELONG_SOURCE_COLUMN,
    NORMALIZED_FIELDS,
    TS_UTC,
    events_schema,
    payload_columns,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Разбор ленты в типизированный canonical. Одна строка RAW —
# одна строка canonical, ничего не исчезает.
#
# Работа идёт пачками целых клиентов: все версии, дубли и
# ссылки одного события лежат у одного клиента, поэтому пачка
# самодостаточна, а память ограничена её размером. Клиент,
# попавший на границу row group, переносится в следующую пачку
# целиком.
#
# Разбор payload векторный: для каждого типа события строится
# таблица полной ширины, затем все они складываются и
# возвращаются в исходный порядок одной перестановкой.
# ============================================================


class CanonicalError(ValueError):
    """
    Вход необрабатываем: строка не разбирается по контракту или
    не даёт построить порядок. Этап останавливается с названием
    поля и номером строки, а не падает исключением Python где-то
    глубже и не выбрасывает строку молча.
    """


# Структурно обязательные поля конверта. Паспорт блокирует
# выгрузку с пустым значением здесь, но canonical не полагается
# на то, что его запускали: проверка повторяется до сортировки.
REQUIRED_ENVELOPE_FIELDS: tuple[str, ...] = (
    "client_id",
    "event_time",
    "source",
)


def require_envelope(batch: pa.Table) -> None:
    """
    Останавливает разбор на первой строке с пустым обязательным
    полем конверта.
    """

    for name in REQUIRED_ENVELOPE_FIELDS:

        column = batch.column(name)

        if not column.null_count:
            continue

        position = pc.indices_nonzero(pc.is_null(column))[0].as_py()
        row = batch.column("raw_row")[position].as_py() if "raw_row" in batch.column_names else position

        raise CanonicalError(
            f"строка {row}: пустое обязательное поле конверта {name}. "
            f"Всего таких строк в пачке: {column.null_count}. "
            "Выгрузка необрабатываема: проверка RAW называет такую строку до начала обработки"
        )


def normalize_text(value: str | None) -> str | None:
    """
    Нормализованная копия текста. Исходное значение остаётся в
    своей колонке: транслитерации и переименования здесь нет.
    """

    if value is None:
        return None

    text = unicodedata.normalize("NFKC", value)
    text = " ".join(text.split())

    return text.casefold() or None


def normalize_column(column: pa.Array | pa.ChunkedArray) -> pa.Array:
    """
    normalize_text каждого значения колонки. Названия повторяются, и
    нормализуется каждое РАЗНОЕ значение один раз: та же функция над
    словарём колонки, затем раскладка по строкам.
    """

    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()

    encoded = pc.dictionary_encode(column)

    dictionary = pa.array([normalize_text(value) for value in encoded.dictionary.to_pylist()], pa.string())

    return dictionary.take(encoded.indices)


# ============================================================
# ПАЧКИ КЛИЕНТОВ
# ============================================================


def _client_codes(column: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """
    Номер значения client_id у каждой строки: равные id — равные
    номера. Сравнивать номера дешевле, чем строки Python; пустой
    client_id (его останавливает проверка RAW) сравнивается как
    раньше, строками.
    """

    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()

    if column.null_count:
        return np.asarray(column.to_pylist(), dtype=object)

    return pc.dictionary_encode(column).indices.to_numpy()


def _client_runs(client_id: np.ndarray) -> list[tuple[int, int]]:
    """
    Границы [lo, hi) непрерывных блоков клиента.
    """

    if client_id.size == 0:
        return []

    change = np.flatnonzero(client_id[1:] != client_id[:-1]) + 1

    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [client_id.size]])

    return [(int(lo), int(hi)) for lo, hi in zip(starts, ends)]


def iter_client_batches(raw: RawDataset, batch_clients: int, seen: set | None = None) -> Iterator[pa.Table]:
    """
    Пачки строк ленты, содержащие только целых клиентов.

    seen — множество, в которое складываются клиенты ленты по мере
    обхода (его же проверка разорванной ленты); передав своё,
    вызывающий знает клиентов без второго прохода по файлу.

    Клиент, чей блок обрывается на границе row group, переносится
    в следующую пачку: делить его нельзя, иначе версии и ссылки
    события окажутся в разных пачках.

    Клиент, встретившийся второй раз после другого, это уже не
    перенос, а разорванная лента. Молчать об этом нельзя: второй
    блок перезаписал бы первый в индексе, и часть истории
    исчезла бы без следа.
    """

    carry: pa.Table | None = None
    offset = 0
    seen = set() if seen is None else seen

    for group_index, chunk in raw.iter_row_groups("events"):

        rows = chunk.num_rows

        chunk = chunk.append_column("raw_row_group", pa.array(np.full(rows, group_index, dtype=np.int32)))
        chunk = chunk.append_column("raw_row", pa.array(np.arange(offset, offset + rows, dtype=np.int64)))

        offset += rows

        table = chunk if carry is None else pa.concat_tables([carry, chunk])

        runs = _client_runs(_client_codes(table.column("client_id")))

        if not runs:
            carry = None
            continue

        # Последний блок может продолжиться в следующем row group.
        last_start = runs[-1][0]

        head = table.slice(0, last_start)
        carry = table.slice(last_start)

        if head.num_rows == 0:
            continue

        head_runs = _client_runs(_client_codes(head.column("client_id")))

        for start in range(0, len(head_runs), batch_clients):
            window = head_runs[start : start + batch_clients]
            lo = window[0][0]
            hi = window[-1][1]
            piece = head.slice(lo, hi - lo)
            _refuse_split_clients(piece, seen)
            yield piece

    if carry is not None and carry.num_rows:
        _refuse_split_clients(carry, seen)
        yield carry


def _refuse_split_clients(batch: pa.Table, seen: set) -> None:
    """
    Клиент не может начаться заново после другого клиента.
    """

    column = batch.column("client_id")

    for lo, _hi in _client_runs(_client_codes(column)):

        client = column[lo].as_py()

        if client in seen:
            raise CanonicalError(
                f"строки клиента {client} лежат в ленте не подряд: "
                "пачка целых клиентов собирается только из непрерывного блока"
            )

        seen.add(client)


# ============================================================
# РАЗБОР PAYLOAD
# ============================================================


@dataclass
class ParsedBatch:
    # Колонки payload, строки сгруппированы по типу события.
    table: pa.Table
    # Тип события каждой строки (ключ type payload) в ИСХОДНОМ порядке,
    # разобранный один раз на пачку: им же пользуется пометка вех.
    event_type: pa.Array
    # Исходный номер каждой строки table: table[i] — строка positions[i]
    # пачки.
    positions: np.ndarray


def parse_batch(manifest: RawManifest, batch: pa.Table, payload_names: list[str]) -> ParsedBatch:
    """
    Типизированные колонки payload для всей пачки, сгруппированные по
    типу события, и исходный номер каждой строки. В исходный порядок
    широкая таблица не переставляется: сортировка этапа и так
    переставит строки, и номер строки входит в неё последним ключом.
    """

    rows = batch.num_rows

    columns = {name: field_type for name, field_type in payload_columns(manifest)}

    if rows == 0:
        return ParsedBatch(
            pa.table({name: pa.nulls(0, columns[name]) for name in payload_names}),
            pa.array([], pa.string()),
            np.zeros(0, dtype=np.int64),
        )

    order: list[np.ndarray] = []
    pieces: list[pa.Table] = []

    # Тип события читается из payload ключом type: отдельной
    # колонки у него нет ни в выгрузке, ни в слое. Строка без
    # разобранного типа уходит в unknown_event_type, а не
    # получает выдуманный тип.
    event_type_column = event_types_of(batch.column("payload"))

    payload = batch.column("payload")
    raw_rows = batch.column("raw_row")

    # Строки по типам одной устойчивой сортировкой номера типа, а не
    # маской на каждый тип: группы те же, строки внутри типа — в
    # исходном порядке, типы — в порядке первого появления. Берётся
    # только payload: остальные колонки пачки разбору не нужны.
    encoded = pc.dictionary_encode(event_type_column)
    code_of = {name: code for code, name in enumerate(encoded.dictionary.to_pylist())}
    codes = pc.fill_null(encoded.indices, -1).to_numpy()
    grouped = np.argsort(codes, kind="stable")
    bounds = codes[grouped]

    for event_type in pc.unique(event_type_column).to_pylist():

        code = -1 if event_type is None else code_of[event_type]
        lo, hi = np.searchsorted(bounds, [code, code + 1])
        indices = grouped[lo:hi].astype(np.int64)

        info = manifest.catalogue.get(event_type) if event_type is not None else None

        if info is None:
            row = raw_rows[int(indices[0])].as_py()
            raise CanonicalError(
                f"строка {row}: тип события {event_type!r} не объявлен каталогом ключей; "
                "разобрать такую запись нечем"
            )

        parsed = parse_payloads(info, payload.take(pa.array(indices)))

        # Любое расхождение с контрактом останавливает этап:
        # журнала отказов больше нет, и молча принять строку
        # нельзя.
        if parsed.by_row:
            local = min(parsed.by_row)
            row = raw_rows[int(indices[local])].as_py()
            raise CanonicalError(
                f"строка {row}, событие {event_type}: "
                + "; ".join(parsed.by_row[local])
            )

        piece = pa.table(
            {
                name: (
                    parsed.table.column(name)
                    if name in parsed.table.column_names
                    else pa.nulls(len(indices), columns[name])
                )
                for name in payload_names
            }
        )

        order.append(indices)
        pieces.append(piece)

    return ParsedBatch(pa.concat_tables(pieces), event_type_column, np.concatenate(order))


# ============================================================
# РАЗРЫВ ЦЕПОЧКИ ОСТАТКОВ
# ============================================================
#
# Наблюдаемый остаток счёта обязан продолжать предыдущий
# наблюдаемый остаток того же счёта. Если не продолжает, между
# строками потеряно движение денег: сбой источника не донёс
# запись до выгрузки.
#
# Это признак КАЧЕСТВА НАБЛЮДЕНИЯ, а не ошибка арифметики, и
# считается он по самой выгрузке: ничего, кроме неё, у
# canonical нет.
#
# ============================================================
# СОБЫТИЯ-ИСТОЧНИКИ ВЕХ
# ============================================================
#
# Веха анкеты о продукте называет свой источник: source_id — это
# card_id первой карты или contract_id первого кредита и вклада.
# Акт источника записан в ленте строками типов
# LIFELONG_SOURCE_EVENTS, у которых в поле LIFELONG_SOURCE_FIELD
# тот же идентификатор; открытие договора со счётом — это две
# строки одного момента, account_opened и product_opened.
#
# Строка находится по ссылке, а не по времени: соседнее событие
# того же момента с другим идентификатором источником не
# становится. Время здесь только сверяется — ссылка, указавшая на
# строку другого момента, это сломанная выгрузка. Веха внутри
# окна обязана найти свой акт, веха до окна — нет: её акт в
# ленту не попал.
# ============================================================


# Тип строки -> вехи, источником которых она может быть.
_SOURCE_KINDS: dict[str, tuple[str, ...]] = {
    event_type: tuple(kind for kind, types in LIFELONG_SOURCE_EVENTS.items() if event_type in types)
    for types in LIFELONG_SOURCE_EVENTS.values()
    for event_type in types
}


# Типы строк, которые бывают источниками вех.
_SOURCE_TYPES = pa.array(sorted(_SOURCE_KINDS), pa.string())


def lifelong_sources(
    batch: pa.Table,
    event_types: pa.Array,
    moments: pa.Array,
    milestones: dict[str, list[dict]],
    period_start: datetime,
) -> pa.Array:
    """
    Тип вехи у каждой строки её акта-источника, у остальных null.

    milestones — вехи анкеты по client_id, как в выгрузке; moments —
    моменты строк в UTC (timestamp), в Python переводится только
    момент строки-источника.
    """

    clients = batch.column("client_id")
    payloads = batch.column("payload")
    raw_rows = batch.column("raw_row")

    # (клиент, тип вехи, source_id) -> момент вехи.
    wanted: dict[tuple[str, str, str], datetime] = {
        (client_id, item["type"], item["source_id"]): item["event_time"]
        for client_id in set(pc.unique(clients).to_pylist())
        for item in milestones.get(client_id, ())
        if item["source_id"] is not None
    }

    marks: list[str | None] = [None] * batch.num_rows
    found: set[tuple[str, str, str]] = set()

    # Обходятся только строки типов-источников, по порядку: у прочих
    # строк источника вехи не бывает, и прежний обход их пропускал.
    candidates = pc.indices_nonzero(pc.is_in(event_types, value_set=_SOURCE_TYPES)).to_numpy()

    for row, client_id, event_type in zip(
        candidates.tolist(),
        clients.take(candidates).to_pylist(),
        event_types.take(candidates).to_pylist(),
    ):

        kinds = _SOURCE_KINDS[event_type]

        payload = json.loads(payloads[row].as_py())

        for kind in kinds:

            link = (client_id, kind, payload.get(LIFELONG_SOURCE_FIELD[kind]))

            if link not in wanted:
                continue

            where = f"строка {raw_rows[row].as_py()} ({event_type})"

            if marks[row] is not None:
                raise CanonicalError(f"{where}: источник сразу двух вех, {marks[row]} и {kind}")

            moment = moments[row].as_py()

            if moment != wanted[link]:
                raise CanonicalError(
                    f"{where}: источник вехи {kind} ({link[2]}) записан в "
                    f"{moment.isoformat()}, а веха — в {wanted[link].isoformat()}"
                )

            marks[row] = kind
            found.add(link)

    for link, moment in wanted.items():
        if moment >= period_start and link not in found:
            raise CanonicalError(
                f"клиент {link[0]}: веха {link[1]} в {moment.isoformat()} внутри окна "
                f"выгрузки, а её источника {link[2]} в ленте нет"
            )

    return pa.array(marks, pa.string())


# ============================================================
# СБОРКА ПАЧКИ
# ============================================================


@dataclass
class BatchResult:
    table: pa.Table


def build_batch(
    raw: RawDataset,
    config: PreprocessingConfig,
    batch: pa.Table,
    payload_names: list[str],
    schema: pa.Schema,
    milestones: dict[str, list[dict]],
) -> BatchResult:
    """
    Пачка строк RAW в очищенную таблицу смысловых полей и пометку
    событий-источников вех.
    """

    manifest = raw.manifest
    rows = batch.num_rows

    # До разбора и сортировки: дальше по коду пустое
    # обязательное поле превращается либо в исключение
    # сравнения, либо в невидимую строку.
    require_envelope(batch)

    parsed = parse_batch(manifest, batch, payload_names)

    # --- время ---
    #
    # Время выгрузки строкой приводится к UTC здесь и больше
    # нигде: дальше по конвейеру ездит нормализованный момент.
    try:
        moments = event_times(batch.column("event_time"))
    except RawContractError as error:
        raise CanonicalError(str(error)) from error

    # --- состав строки ---

    # Строки собираются в порядке разобранного payload (по типам):
    # конверт, время и пометка вех переставляются к нему, а не широкая
    # таблица payload к исходному порядку.
    grouped = pa.array(parsed.positions)

    columns: dict[str, pa.Array | pa.ChunkedArray] = {
        name: batch.column(name).take(grouped) for name in ENVELOPE_NAMES if name != "event_time"
    }

    columns["event_time"] = moments.cast(TS_UTC).take(grouped)

    columns[LIFELONG_SOURCE_COLUMN] = lifelong_sources(
        batch,
        parsed.event_type,
        moments,
        milestones,
        manifest.period_start,
    ).take(grouped)

    for name in payload_names:

        column = parsed.table.column(name)

        # Нормализованный текст ложится в само поле: исходное
        # написание рядом не хранится.
        if name in NORMALIZED_FIELDS:
            column = normalize_column(column)

        columns[name] = column

    table = pa.table(columns).select(schema.names).cast(schema)

    if table.num_rows != rows:
        raise CanonicalError(f"пачка собрана из {table.num_rows} строк вместо {rows}")

    return BatchResult(table.take(_order(table, manifest.event_type_priority, parsed.positions)))


# Колонка приоритета существует только на время сортировки.
# Имя с двумя подчёркиваниями не может совпасть со смысловым
# ключом: ключи приходят из каталога payload, а там такие
# имена не объявляются.
_PRIORITY = "__type_priority"
_POSITION = "__row_position"


def _order(table: pa.Table, priority: dict[str, int], positions: np.ndarray) -> pa.Array:
    """
    Устойчивый порядок строк внутри клиента:

      1. event_time по возрастанию;
      2. при равном времени — причинный приоритет типа события:
         заявка раньше решения по ней, открытие счёта раньше
         движения по нему. Одинаковое время это ограничение
         точности записи, а не свидетельство, что решение
         приняли раньше заявки;
      3. при равном времени и типе — источник и дальше
         остальные смысловые поля в порядке имени.

    Приоритет объявлен порядком EVENT_TYPES в контракте
    генератора и приходит сюда манифестом выгрузки. Тип, которого
    в контракте нет, сюда не доходит: его останавливает разбор
    payload. Значение приоритета в ленту не пишется — по нему
    только сортируют.

    Порядок зависит только от самих данных: ни номера строки в
    RAW, ни другого служебного ключа рядом нет, и два прогона
    одной выгрузки дают один файл.

    Строки table идут по типам событий, positions — их номера в
    исходной пачке. Номер — последний ключ: при равенстве всех
    смысловых полей строки остаются в исходном порядке, то есть
    порядок тот же, что у устойчивой сортировки исходной пачки.
    """

    leading = ["client_id", "event_time", _PRIORITY, TYPE_KEY, "source"]

    # Пометка источника вехи — не данные события, и порядок от неё
    # не зависит.
    rest = sorted(
        name for name in table.column_names
        if name not in leading and name != LIFELONG_SOURCE_COLUMN
    )

    unknown = len(priority)

    # Приоритет типа: номер типа в справочнике -> его приоритет; тип
    # вне справочника и пустой тип получают unknown. То же, что
    # priority.get(name, unknown), но без перевода колонки в Python.
    names = list(priority)
    position = pc.index_in(table.column(TYPE_KEY), value_set=pa.array(names, pa.string()))
    ranks = pa.array([priority[name] for name in names] + [unknown], pa.int16())

    ranked = table.append_column(
        _PRIORITY,
        ranks.take(pc.fill_null(position, len(names))),
    ).append_column(_POSITION, pa.array(positions, pa.int64()))

    keys = [(name, "ascending", "at_end") for name in leading + rest + [_POSITION]]

    return pc.sort_indices(ranked, sort_keys=keys)


def canonical_schema(manifest: RawManifest) -> pa.Schema:
    return events_schema(manifest)


__all__ = [
    "CanonicalError",
    "BatchResult",
    "build_batch",
    "canonical_schema",
    "iter_client_batches",
    "lifelong_sources",
    "normalize_column",
    "normalize_text",
    "parse_batch",
]
