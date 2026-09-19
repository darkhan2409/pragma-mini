from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import _md_table, dumps_json, sha256_bytes, sha256_file, write_json, write_table, write_text
from .canonical.build import CLIENT_INDEX_FILE
from .canonical.build import STAGE as CANONICAL_STAGE
from .canonical.schema import DERIVED_NAMES, ENVELOPE_NAMES
from .history import STAGE_VERSION as HISTORY_VERSION
from .history import CanonicalStore, ClientHistory, history_as_of
from .manifest import fingerprint_path, load_fingerprint
from .rawdata import TABLE_FILES, ContentDigest
from .settings import GroupWindow, PreprocessingConfig


# ============================================================
# ИДЕЯ
# ============================================================
#
# Этап ничего не делит: клиенты уже разведены по трём независимым
# выгрузкам. Здесь это закрепляется и проверяется.
#
# Что закрепляется:
#   какие клиенты принадлежат группе и почему кто-то исключён;
#   какое окно истории и какой период будущих целей у группы;
#   что именно разрешено видеть будущему fit — train на fit_end,
#   и контрольная сумма ровно этого содержимого.
#
# Что проверяется:
#   популяции независимы: свой seed, ни одного общего client_id;
#   мир общий: справочники продуктов, мерчантов и географии
#   совпадают по содержимому, как и продуктовая хронология с
#   конфигурацией генерации;
#   конечный cutoff группы лежит внутри её выгрузки.
#
# Несовпадение мира это не ошибка препроцессинга, а свойство
# входа: отчёт называет расхождения и говорит, что набор не
# пригоден для обучения, но артефакты пишет.
#
# Видимое содержимое каждой группы читается единственным
# интерфейсом этапа 3 — history_as_of. Второй реализации
# видимости здесь нет, поэтому «разрешено» значит ровно то же
# самое, что и в истории на дату.
#
# Никаких словарей, квантилей, бакетов и масок здесь нет: этап
# только называет границы, внутри которых им позже разрешено
# учиться.
# ============================================================


STAGE = "split"
STAGE_VERSION = "3.5.0"
SCHEMA_VERSION = 1

SPLIT_MANIFEST_FILE = "split_manifest.json"
TRAIN_INDEX_FILE = "train_corpus_index.parquet"
REPORT_MD_FILE = "split_report.md"

TRAIN_GROUP = "train"

STATUS_OK = "ok"
STATUS_BLOCKED_BY_INPUT = "blocked_by_input"
STATUS_BLOCKED = "blocked"

EXCLUDED_TEST_ACCOUNT = "test_account"

# Справочники, которые обязаны быть общими у трёх групп.
WORLD_CATALOGS: tuple[str, ...] = ("products", "merchants", "geography")

# Поля конверта, входящие в контрольную сумму содержимого.
# is_test_account не входит: тестовые клиенты в рабочие группы не
# попадают вовсе, и признак в них всегда пуст.
CHECKSUM_ENVELOPE: tuple[str, ...] = (
    "event_id",
    "client_id",
    "event_type",
    "source",
    "event_time",
    "effective_at",
    "time_precision",
    "event_version",
    "change_initiator",
    "correlation_id",
    "link_type",
)

# Поля версии профиля, которые в сумму НЕ входят: служебный
# индекс, трассировка к RAW и valid_to. Закрытие версии выражает
# следующая строка, а на cutoff её может ещё не быть.
PROFILE_SKIP: frozenset[str] = frozenset({"client_idx", "raw_file", "raw_row_group", "raw_row", "valid_to"})

# Колонки, которых в сумме нет: производные canonical и поля
# конверта, не названные выше.
CHECKSUM_SKIP: frozenset[str] = frozenset(DERIVED_NAMES) | (frozenset(ENVELOPE_NAMES) - frozenset(CHECKSUM_ENVELOPE))

TRAIN_INDEX_SCHEMA = pa.schema(
    [
        ("client_idx", pa.int64()),
        ("stable_event_index", pa.int64()),
        ("event_time", pa.timestamp("us")),
        ("mlm_target_eligible", pa.bool_()),
    ]
)


# ============================================================
# ВХОД
# ============================================================


@dataclass(frozen=True)
class GroupInput:
    """
    Одна группа: её выгрузка, её canonical и вердикты этапов
    выше. Пути живут здесь и в артефакты не попадают.
    """

    group: str
    raw_dir: Path
    canonical_dir: Path
    canonical_fingerprint: str
    passport_status: str | None = None
    diagnostic_mode: bool = False


class SplitError(ValueError):
    """
    Корпус открывать нельзя: разделение непригодно, canonical
    пересобран после него либо клиент не входит в группу.
    """


@dataclass
class SplitResult:
    report: dict
    outputs: list[Path] = field(default_factory=list)


# ============================================================
# ОБЩИЙ МИР
# ============================================================


def catalog_digests(raw_dir: Path) -> dict[str, str]:
    """
    Отпечаток содержимого справочников: сумма по строкам, не
    зависящая ни от их порядка, ни от того, как записан parquet.

    Манифест генератора подписывает только ленту, профиль и
    покрытие, поэтому равенство справочников считается здесь, а
    не берётся на слово.
    """

    digests: dict[str, str] = {}

    for name in WORLD_CATALOGS:

        path = Path(raw_dir) / TABLE_FILES[name]

        if not path.exists():
            digests[name] = "<нет файла>"
            continue

        digest = ContentDigest()
        digest.extend(pq.read_table(path).to_pylist())
        digests[name] = digest.value()

    return digests


def world_check(worlds: dict[str, dict]) -> dict:
    """
    Один ли мир у групп: справочники, продуктовая хронология,
    конфигурация генерации и world_seed.

    Отсутствие world_seed само по себе мир не разводит: если
    содержимое всех справочников, продуктовая хронология и
    конфигурация генерации совпали, мир общий, а неподтверждённое
    происхождение остаётся ограничением.

    worlds: группа -> {catalogs, product_timeline_sha256,
    generation_config_sha256, world_seed}.
    """

    mismatches: list[str] = []
    notes: list[str] = []

    def compare(label: str, values: dict[str, object]) -> None:
        if len({str(value) for value in values.values()}) > 1:
            listed = ", ".join(f"{group}: {str(values[group])[:12]}" for group in sorted(values))
            mismatches.append(f"{label} различается ({listed})")

    for catalog in WORLD_CATALOGS:
        compare(f"справочник {catalog}", {group: item["catalogs"][catalog] for group, item in worlds.items()})

    compare("продуктовая хронология", {group: item["product_timeline_sha256"] for group, item in worlds.items()})
    compare("конфигурация генерации", {group: item["generation_config_sha256"] for group, item in worlds.items()})

    seeds = {group: item["world_seed"] for group, item in worlds.items()}

    if any(value is None for value in seeds.values()):
        notes.append(
            "происхождение общего мира не подтверждено: world_seed не объявлен генератором, "
            "равенство установлено по содержимому справочников"
        )
    else:
        compare("world_seed", seeds)

    return {
        "shared_world": not mismatches,
        "mismatches": mismatches,
        "notes": notes,
        "world_seed": dict(sorted(seeds.items())),
        "catalogs": {group: item["catalogs"] for group, item in sorted(worlds.items())},
        "rule": (
            "мир общий, если совпали справочники, продуктовая хронология и конфигурация генерации; "
            "world_seed подтверждает происхождение, а его отсутствие остаётся ограничением"
        ),
    }


def population_check(clients: dict[str, list[str]], seeds: dict[str, int]) -> dict:
    """
    Популяции независимы: свой seed у каждой группы и ни одного
    общего client_id.
    """

    overlaps: list[dict] = []

    names = sorted(clients)

    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            left, right = names[first], names[second]
            shared = sorted(set(clients[left]) & set(clients[right]))
            if shared:
                overlaps.append({"groups": [left, right], "clients": len(shared), "examples": shared[:5]})

    values = list(seeds.values())
    repeated = sorted({value for value in values if values.count(value) > 1})

    return {
        "seeds": dict(sorted(seeds.items())),
        "distinct_seeds": not repeated,
        "repeated_seeds": repeated,
        "overlaps": overlaps,
        "shared_clients": sum(item["clients"] for item in overlaps),
    }


# ============================================================
# ВИДИМОЕ СОДЕРЖИМОЕ ГРУППЫ
# ============================================================


def mlm_target_eligible(event_time: datetime, window: GroupWindow) -> bool:
    """
    Событие попадает в период будущих целей группы.

    Это признак периода и ничего больше: ни выбор масок, ни
    окончательная допустимость цели для конкретного Masker здесь
    не решаются.
    """

    return window.target_start <= event_time < window.target_end


def _event_rows(history: ClientHistory) -> list[dict]:
    """
    Видимые события без производных колонок: конверт как он есть
    и типизированный payload.
    """

    table = history.events

    keep = [name for name in table.column_names if name not in CHECKSUM_SKIP]

    return table.select(keep).to_pylist()


def _profile_rows(store: CanonicalStore, client_id: str, cutoff: datetime) -> list[dict]:
    """
    Все версии профиля, действовавшие к cutoff. В сумму входят
    именно они, а не только последняя: правка старой версии
    меняет знание банка о прошлом.
    """

    rows = [row for row in store.profile_rows(client_id) if row["valid_from"] < cutoff]

    return [{name: value for name, value in row.items() if name not in PROFILE_SKIP} for row in rows]


@dataclass
class VisibleSet:
    """
    Что видно у группы на её конечном cutoff: индекс строк,
    контрольная сумма содержимого и сводка по клиентам.
    """

    index: pa.Table
    checksum: dict
    stats: dict


def collect_visible(store: CanonicalStore, client_ids: list[str], window: GroupWindow) -> VisibleSet:
    """
    Проходит клиентов группы через history_as_of на её конечном
    cutoff и собирает видимое содержимое.

    Для train это и есть разрешённый интерфейс будущего fit.

    Контрольная сумма считается по финальной очищенной истории
    событий до cutoff: конверт и payload видимых строк, версии
    профиля, действующие к этому моменту, датированные факты
    покрытия. Служебные индексы, флаги и
    итоговые статусы выгрузки в неё не входят, поэтому пересборка
    слоя её не меняет.
    """

    cutoff = window.final_cutoff

    events_digest = ContentDigest()
    profile_digest = ContentDigest()
    coverage_digest = ContentDigest()

    client_idx: list[int] = []
    stable_index: list[int] = []
    event_time: list[datetime] = []
    eligible: list[bool] = []

    stats = {
        "clients": len(client_ids),
        "silent_clients": 0,
        "clients_without_profile": 0,
        "clients_without_observed_start": 0,
        "clients_with_incomplete_history": 0,
    }

    for client_id in client_ids:

        history = history_as_of(store, client_id, cutoff)

        rows = _event_rows(history)

        events_digest.extend(rows)

        for row in rows:
            client_idx.append(history.client_idx)
            event_time.append(row["event_time"])
            eligible.append(mlm_target_eligible(row["event_time"], window))

        stable_index.extend(history.events.column("stable_event_index").to_pylist())

        profile_digest.extend(_profile_rows(store, client_id, cutoff))
        coverage_digest.extend([item.as_dict() for item in history.coverage])

        if not rows:
            stats["silent_clients"] += 1

        if history.profile is None:
            stats["clients_without_profile"] += 1

        if history.relationship.observed_start is None:
            stats["clients_without_observed_start"] += 1

        if history.relationship.history_incomplete:
            stats["clients_with_incomplete_history"] += 1

    index = pa.Table.from_pydict(
        {
            "client_idx": client_idx,
            "stable_event_index": stable_index,
            "event_time": event_time,
            "mlm_target_eligible": eligible,
        },
        schema=TRAIN_INDEX_SCHEMA,
    )

    checksum = {
        "cutoff": cutoff.isoformat(),
        "events": events_digest.value(),
        "events_rows": events_digest.rows,
        "profile": profile_digest.value(),
        "profile_rows": profile_digest.rows,
        "coverage": coverage_digest.value(),
        "coverage_rows": coverage_digest.rows,
        "rule": (
            "по финальной очищенной истории событий до cutoff: конверт и payload видимых строк, "
            "версии профиля с valid_from < cutoff, датированные факты покрытия; служебные индексы, "
            "флаги и итоговые статусы выгрузки не входят"
        ),
    }

    checksum["content_sha256"] = sha256_bytes(
        dumps_json({key: checksum[key] for key in ("events", "profile", "coverage")}).encode("utf-8")
    )

    return VisibleSet(index=index, checksum=checksum, stats=stats)


def canonical_fingerprint(processed_dir: Path, group: str = TRAIN_GROUP) -> str | None:
    """
    Отпечаток canonical группы по её маркеру: им проверяется, что
    слой не пересобран после разделения.
    """

    marker = load_fingerprint(fingerprint_path(Path(processed_dir), CANONICAL_STAGE, group))

    return None if marker is None else marker.get("fingerprint")


def edited_canonical_files(processed_dir: Path, canonical_dir: Path, group: str = TRAIN_GROUP) -> list[str]:
    """
    Файлы canonical, которые отличаются от выходов своего маркера.

    Отпечатка мало: он подтверждает, что маркер и манифест
    согласны между собой, но ничего не говорит о ФАЙЛАХ, из
    которых корпус будет читать строки. Правленый events.parquet
    оставляет отпечаток прежним, и обучение пошло бы по данным,
    которых разделение не видело.
    """

    marker = load_fingerprint(fingerprint_path(Path(processed_dir), CANONICAL_STAGE, group))

    if marker is None:
        return ["<нет маркера canonical>"]

    prefix = f"{CANONICAL_STAGE}/{group}/"

    changed: list[str] = []

    for name, digest in sorted(marker.get("outputs", {}).items()):

        if not name.startswith(prefix):
            continue

        path = Path(canonical_dir) / name[len(prefix):]

        if not path.exists() or sha256_file(path) != digest:
            changed.append(name[len(prefix):])

    return changed


class TrainCorpus:
    """
    Единственный разрешённый способ будущего fit получить строки:
    train на fit_end через историю на дату.

    Интерфейс держит границу сам, а не надеется на дисциплину
    вызывающего: непригодное разделение не открывается, fit_end
    берётся из манифеста, а не из переданного конфига, canonical
    сверяется с тем, на котором разделение было построено, и
    история выдаётся только по клиентам разрешённой группы.

    Индекс говорит, какие строки разрешены; содержимое читается
    той же функцией, что и любая другая история, поэтому второй
    реализации видимости не существует.
    """

    def __init__(self, store: CanonicalStore, index: pa.Table, fit_end: datetime, client_ids: list[str]):
        self.store = store
        self.index = index
        self.fit_end = fit_end
        self.client_ids = list(client_ids)
        self._allowed = frozenset(self.client_ids)

    @staticmethod
    def open(split_dir: Path, canonical_dir: Path, processed_dir: Path | None = None,
             allow_unusable: bool = False, allow_short_horizon: bool = False,
             products: pa.Table | None = None) -> "TrainCorpus":
        """
        allow_unusable — режим диагностики: снимает проверки
        пригодности и свежести. Обучение открывает корпус без него.

        allow_short_horizon — согласиться на короткий горизонт.
        Технически такое разделение исправно, но договорённый
        горизонт оно не выполняет, и молчаливое согласие на это
        принимать нельзя.

        products — справочник продуктов выгрузки. Без него
        название продукта не расшифровывается: смысловой слой
        поверх такого корпуса молча потерял бы product_name.
        Корпус справочник не читает сам: файл лежит в RAW, а
        разрешение на группу выдаёт разделение.
        """

        split_dir = Path(split_dir)

        # Каталог набора: рядом с ним лежат маркеры этапов.
        processed = Path(processed_dir) if processed_dir is not None else split_dir.parent

        manifest = json.loads((split_dir / SPLIT_MANIFEST_FILE).read_text(encoding="utf-8"))

        corpus = manifest.get("train_corpus")

        if not allow_unusable:

            if not manifest.get("usable"):
                reasons = manifest.get("errors") or manifest.get("input_dependencies") or []
                raise SplitError(
                    f"разделение со статусом {manifest.get('status')} непригодно для обучения: "
                    f"{'; '.join(reasons) or 'причина не названа'}. "
                    "Для диагностики откройте с allow_unusable=True"
                )

            actual = canonical_fingerprint(processed)

            if actual is None or actual != manifest["groups"][TRAIN_GROUP]["canonical_fingerprint"]:
                raise SplitError(
                    "canonical группы train не тот, на котором построено разделение: "
                    "выполните этап split заново"
                )

            # Отпечаток говорит о согласии маркера с манифестом, а
            # корпус читает ФАЙЛЫ. Сверяются именно они.
            changed = edited_canonical_files(processed, canonical_dir)

            if changed:
                raise SplitError(
                    "файлы canonical изменились после разделения ("
                    + ", ".join(changed[:5])
                    + "): выполните этапы canonical и split заново"
                )

            if not allow_short_horizon and not manifest.get("contract_met", True):
                reasons = (manifest.get("contract") or {}).get("reasons") or []
                raise SplitError(
                    "разделение технически исправно, но договорённый горизонт не выполнен: "
                    + ("; ".join(reasons) or "причина не названа")
                    + ". Откройте с allow_short_horizon=True, если это осознанное решение"
                )

        if corpus is None:
            raise SplitError("разрешённого train-корпуса в этом разделении нет")

        index_path = split_dir / TRAIN_INDEX_FILE

        index = pq.read_table(index_path)

        if index.num_rows != corpus["rows"]:
            raise SplitError(
                f"индекс корпуса разошёлся с манифестом: строк в файле {index.num_rows}, "
                f"в манифесте {corpus['rows']}. Выполните этап split заново"
            )

        if not allow_unusable:

            # Число строк правленый индекс сохраняет, поэтому
            # сверяется сам файл: разрешение на строки обучения
            # выдаёт разделение, а не тот, кто правил parquet.
            expected = corpus.get("index_sha256")

            if not expected or sha256_file(index_path) != expected:
                raise SplitError(
                    "индекс корпуса изменён после разделения или построен прежней версией "
                    "этапа: выполните этап split заново"
                )

        return TrainCorpus(
            store=CanonicalStore(canonical_dir, products=products),
            index=index,
            fit_end=datetime.fromisoformat(corpus["fit_end"]),
            client_ids=list(manifest["groups"][TRAIN_GROUP]["clients"]),
        )

    def require(self, client_id: str) -> str:
        """
        Проверка разрешения на клиента.

        Вынесена из history отдельным методом, потому что читать
        корпус можно не только лентой событий: смысловой слой
        строится поверх того же store, и обходить разрешение он
        не должен.
        """

        if client_id not in self._allowed:
            raise SplitError(
                f"клиент {client_id!r} не входит в разрешённую train-группу: "
                "он исключён, принадлежит другой группе либо отсутствует в разделении"
            )

        return client_id

    def history(self, client_id: str) -> ClientHistory:
        """
        История разрешённого клиента на fit_end. Обращение по
        client_idx не поддерживается: разрешение выдаётся по
        client_id из манифеста.
        """

        return history_as_of(self.store, self.require(client_id), self.fit_end)


# ============================================================
# ГРУППА
# ============================================================


def _client_rows(canonical_dir: Path) -> list[dict]:
    return pq.read_table(Path(canonical_dir) / CLIENT_INDEX_FILE).to_pylist()


def group_summary(source: GroupInput, store: CanonicalStore, clients: list[dict], window: GroupWindow,
                  visible: VisibleSet) -> dict:
    """
    Что группа собой представляет: размер, исключения, окно и
    покрытие времени на её конечном cutoff.
    """

    working = sorted(row["client_id"] for row in clients if not row["is_test_account"])
    excluded = sorted(row["client_id"] for row in clients if row["is_test_account"])

    times = visible.index.column("event_time").to_pylist()

    echo = store.report["raw"]

    return {
        "directory": source.group,
        "seed": echo["seed"],
        "world_seed": echo["world_seed"],
        "raw_history_start": echo["history_start"],
        "extract_time": echo["extract_time"],
        "window": window.as_dict(),
        "clients_total": len(clients),
        "clients_working": len(working),
        "clients": working,
        "clients_sha256": sha256_bytes("\n".join(working).encode("utf-8")),
        "excluded": {EXCLUDED_TEST_ACCOUNT: len(excluded)},
        "excluded_clients": excluded,
        "visible_events": visible.index.num_rows,
        "eligible_events": int(sum(visible.index.column("mlm_target_eligible").to_pylist())),
        "first_visible_event": min(times).isoformat() if times else None,
        "last_visible_event": max(times).isoformat() if times else None,
        "silent_clients": visible.stats["silent_clients"],
        "clients_without_profile": visible.stats["clients_without_profile"],
        "clients_without_observed_start": visible.stats["clients_without_observed_start"],
        "clients_with_incomplete_history": visible.stats["clients_with_incomplete_history"],
        "content_at_cutoff": visible.checksum,
        "canonical_fingerprint": source.canonical_fingerprint,
        "passport_status": source.passport_status,
        "diagnostic_mode": source.diagnostic_mode,
    }


# ============================================================
# ЭТАП
# ============================================================


def build_split(sources: list[GroupInput], config: PreprocessingConfig, target: Path) -> SplitResult:
    """
    Закрепляет группы, проверяет их независимость и общий мир,
    собирает разрешённый train-интерфейс и пишет артефакты.
    """

    target = Path(target)

    by_group = {source.group: source for source in sources}

    errors: list[str] = []
    input_dependencies: list[str] = []
    limitations: list[str] = []

    for name in sorted(set(config.windows) - set(by_group)):
        errors.append(f"группа {name} отсутствует: разделение описывает три независимые выгрузки")

    stores: dict[str, CanonicalStore] = {}
    clients: dict[str, list[dict]] = {}
    worlds: dict[str, dict] = {}
    seeds: dict[str, int] = {}
    visible: dict[str, VisibleSet] = {}

    # Выполнение договорённости о горизонте это ОТДЕЛЬНЫЙ вердикт:
    # разделение может быть технически исправным и при этом
    # описывать более короткую историю, чем согласовано.
    contract_groups: dict[str, dict] = {}

    for name, source in sorted(by_group.items()):

        if name not in config.windows:
            errors.append(f"группа {name} не описана окнами конфига")
            continue

        store = CanonicalStore(source.canonical_dir)
        window = config.windows[name]

        stores[name] = store
        clients[name] = _client_rows(source.canonical_dir)

        echo = store.report["raw"]
        seeds[name] = echo["seed"]

        worlds[name] = {
            "catalogs": catalog_digests(source.raw_dir),
            "product_timeline_sha256": echo["product_timeline_sha256"],
            "generation_config_sha256": echo["generation_config_sha256"],
            "world_seed": echo["world_seed"],
        }

        if window.final_cutoff > store.extract_time:
            errors.append(
                f"группа {name}: конечный cutoff {window.final_cutoff.isoformat()} позже границы выгрузки "
                f"{store.extract_time.isoformat()}"
            )
            continue

        contract_groups[name] = {
            "history_start": store.history_start.isoformat(),
            "horizon_ok": store.history_start <= config.required_history_start,
            "passport_status": source.passport_status,
        }

        if store.history_start > config.required_history_start:
            limitations.append(
                f"группа {name}: история начинается {store.history_start.date()}, "
                f"а согласованный горизонт с {config.required_history_start.date()}"
            )

        if source.diagnostic_mode:
            limitations.append(f"группа {name}: canonical собран в режиме диагностики (паспорт contract_mismatch)")

        working = [row["client_id"] for row in clients[name] if not row["is_test_account"]]

        visible[name] = collect_visible(store, working, window)

    world = world_check(worlds) if worlds else {
        "shared_world": False,
        "mismatches": ["групп нет"],
        "notes": [],
        "world_seed": {},
        "catalogs": {},
    }

    population = population_check({name: [row["client_id"] for row in rows] for name, rows in clients.items()}, seeds)

    for item in population["overlaps"]:
        errors.append(
            f"группы {item['groups'][0]} и {item['groups'][1]} делят {item['clients']} клиентов: "
            f"популяции не независимы (например {', '.join(item['examples'])})"
        )

    if population["repeated_seeds"]:
        input_dependencies.append(f"seed популяции повторяется у разных групп: {population['repeated_seeds']}")

    input_dependencies.extend(world["mismatches"])
    limitations.extend(world["notes"])

    groups = {
        name: group_summary(by_group[name], stores[name], clients[name], config.windows[name], visible[name])
        for name in sorted(visible)
    }

    # --- вердикт ---

    if errors:
        status = STATUS_BLOCKED
    elif input_dependencies:
        status = STATUS_BLOCKED_BY_INPUT
    else:
        status = STATUS_OK

    reasons: list[str] = []

    for name, item in sorted(contract_groups.items()):

        if not item["horizon_ok"]:
            reasons.append(
                f"группа {name}: история с {item['history_start'][:10]}, "
                f"согласовано с {config.required_history_start.date()}"
            )

        if item["passport_status"] == "horizon_short":
            reasons.append(f"группа {name}: паспорт назвал горизонт коротким")

    contract = {
        "required_history_start": config.required_history_start.isoformat(),
        "horizon_ok": all(item["horizon_ok"] for item in contract_groups.values()),
        "groups": contract_groups,
        "reasons": reasons,
        "rule": (
            "usable это техническая исправность разделения, contract_met — выполнение "
            "договорённости о горизонте наблюдения; это разные вердикты"
        ),
    }

    corpus = visible.get(TRAIN_GROUP) if not errors else None

    report = {
        "stage": STAGE,
        "schema_version": SCHEMA_VERSION,
        "stage_version": STAGE_VERSION,
        "history_version": HISTORY_VERSION,
        "status": status,
        "usable": status == STATUS_OK,
        "contract": contract,
        "contract_met": not reasons,
        "shared_world": world["shared_world"],
        "world": world,
        "population": population,
        "groups": groups,
        "train_corpus": (
            {
                "group": TRAIN_GROUP,
                "fit_end": config.windows[TRAIN_GROUP].final_cutoff.isoformat(),
                "clients": groups[TRAIN_GROUP]["clients_working"],
                "rows": corpus.index.num_rows,
                "eligible_rows": groups[TRAIN_GROUP]["eligible_events"],
                "index_file": TRAIN_INDEX_FILE,
                # sha256 файла индекса: заполняется после его записи.
                "index_sha256": None,
                "checksum": corpus.checksum,
                "rule": (
                    "события train, видимые на fit_end через историю на дату, без технических дублей "
                    "и тестовых аккаунтов, вместе с профилем и покрытием на ту же дату; "
                    "единственный разрешённый вход будущего fit"
                ),
            }
            if corpus is not None
            else None
        ),
        "required_history_start": config.required_history_start.isoformat(),
        "errors": errors,
        "input_dependencies": input_dependencies,
        "limitations": sorted(set(limitations)),
        "config_sha256": config.sha256(),
    }

    # --- артефакты ---

    outputs: list[Path] = []

    target.mkdir(parents=True, exist_ok=True)

    # Индекс пишется ДО манифеста: манифест несёт sha256 его файла,
    # и корпус при открытии сверяет файл с ним. Число строк
    # правленый индекс сохраняет, содержимое — нет.
    if corpus is not None:
        index_path = target / TRAIN_INDEX_FILE
        write_table(index_path, corpus.index, TRAIN_INDEX_SCHEMA)
        outputs.append(index_path)
        report["train_corpus"]["index_sha256"] = sha256_file(index_path)

    manifest_path = target / SPLIT_MANIFEST_FILE
    write_json(manifest_path, report)
    outputs.append(manifest_path)

    md_path = target / REPORT_MD_FILE
    write_text(md_path, render_split_md(report))
    outputs.append(md_path)

    return SplitResult(report=report, outputs=outputs)


# ============================================================
# ОТЧЁТ
# ============================================================


def _ordered(groups: dict) -> list[tuple[str, dict]]:
    """
    Группы в порядке конвейера, а не по алфавиту.
    """

    from .settings import GROUPS

    order = {name: index for index, name in enumerate(GROUPS)}

    return sorted(groups.items(), key=lambda item: (order.get(item[0], len(order)), item[0]))


def render_split_md(report: dict) -> str:

    out: list[str] = []

    out.append("# Разделение train / validation / test\n")
    out.append(
        f"Статус: **{report['status']}**. Общий мир: {'да' if report['shared_world'] else 'нет'}.\n"
    )
    out.append(
        f"\nТехнически пригодно: **{'да' if report['usable'] else 'нет'}**. "
        f"Договорённость о горизонте выполнена: **{'да' if report.get('contract_met') else 'нет'}**. "
        "Это разные вердикты: исправное разделение может описывать более короткую историю, "
        "чем согласовано.\n"
    )

    for item in (report.get("contract") or {}).get("reasons", ()):
        out.append(f"- {item}")

    if (report.get("contract") or {}).get("reasons"):
        out.append("")

    out.append("\n## Группы\n")
    out.append(
        _md_table(
            [
                [
                    name,
                    item["seed"],
                    item["clients_total"],
                    item["clients_working"],
                    item["excluded"][EXCLUDED_TEST_ACCOUNT],
                    item["window"]["history_start"][:10],
                    item["raw_history_start"][:10],
                    item["window"]["final_cutoff"][:10],
                    f"{item['window']['target_start'][:10]} … {item['window']['target_end'][:10]}",
                ]
                for name, item in _ordered(report["groups"])
            ],
            [
                "группа",
                "seed",
                "клиентов",
                "рабочих",
                "исключено",
                "окно с",
                "выгрузка с",
                "cutoff",
                "период целей",
            ],
        )
    )

    out.append("\n## Что видно на конечном cutoff группы\n")
    out.append(
        _md_table(
            [
                [
                    name,
                    item["visible_events"],
                    item["eligible_events"],
                    item["first_visible_event"],
                    item["last_visible_event"],
                    item["silent_clients"],
                    item["clients_with_incomplete_history"],
                ]
                for name, item in _ordered(report["groups"])
            ],
            [
                "группа",
                "видимых событий",
                "в периоде целей",
                "первое событие",
                "последнее событие",
                "молчащих клиентов",
                "с неполной историей",
            ],
        )
    )

    population = report["population"]

    out.append("\n## Независимость популяций\n")
    out.append(
        _md_table(
            [
                ["разные seed", "да" if population["distinct_seeds"] else f"нет: {population['repeated_seeds']}"],
                ["общих client_id", population["shared_clients"]],
            ],
            ["проверка", "результат"],
        )
    )

    if report["world"]["catalogs"]:
        out.append("\n## Общий мир\n")
        out.append(
            _md_table(
                [
                    [name, digests["products"][:12], digests["merchants"][:12], digests["geography"][:12]]
                    for name, digests in _ordered(report["world"]["catalogs"])
                ],
                ["группа", "продукты", "мерчанты", "география"],
            )
        )

    if report["train_corpus"]:

        corpus = report["train_corpus"]

        out.append("\n## Разрешённый train-интерфейс\n")
        out.append(f"{corpus['rule']}.\n")
        out.append(
            _md_table(
                [
                    ["клиентов", corpus["clients"]],
                    ["строк", corpus["rows"]],
                    ["в периоде целей", corpus["eligible_rows"]],
                    ["fit_end", corpus["fit_end"]],
                    ["sha256 индекса", (corpus.get("index_sha256") or "")[:16]],
                    ["сумма событий", corpus["checksum"]["events"][:16]],
                    ["сумма профиля", corpus["checksum"]["profile"][:16]],
                    ["сумма покрытия", corpus["checksum"]["coverage"][:16]],
                    ["сумма содержимого", corpus["checksum"]["content_sha256"][:16]],
                ],
                ["показатель", "значение"],
            )
        )
        out.append(f"\nВ сумму входит {corpus['checksum']['rule']}.\n")

    for title, key in (
        ("Ошибки", "errors"),
        ("Входные зависимости", "input_dependencies"),
        ("Ограничения наблюдения", "limitations"),
    ):
        if report[key]:
            out.append(f"\n## {title}\n")
            out.extend(f"- {item}" for item in report[key])
            out.append("")

    return "\n".join(out) + "\n"


__all__ = [
    "REPORT_MD_FILE",
    "SCHEMA_VERSION",
    "SPLIT_MANIFEST_FILE",
    "STAGE",
    "STAGE_VERSION",
    "STATUS_BLOCKED",
    "STATUS_BLOCKED_BY_INPUT",
    "STATUS_OK",
    "TRAIN_GROUP",
    "TRAIN_INDEX_FILE",
    "TRAIN_INDEX_SCHEMA",
    "GroupInput",
    "SplitError",
    "SplitResult",
    "TrainCorpus",
    "VisibleSet",
    "build_split",
    "canonical_fingerprint",
    "catalog_digests",
    "collect_visible",
    "group_summary",
    "mlm_target_eligible",
    "population_check",
    "render_split_md",
    "world_check",
]
