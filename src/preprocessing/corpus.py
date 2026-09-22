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
from .rawdata import ContentDigest
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
# Группы генерируются по одной, и этап описывает те, что есть.
# Без train он не запускается: разрешённый корпус собирается
# только из неё. Отсутствие val или test — ограничение отчёта,
# а не ошибка; независимость и общий мир при одной группе
# сравнивать не с чем, и отчёт говорит это прямо.
#
# Что проверяется:
#   популяции независимы: ни одного общего client_id;
#   мир общий: справочники продуктов, мерчантов и географии
#   совпадают по содержимому;
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


STAGE = "corpus"
STAGE_VERSION = "7.0.0"
SCHEMA_VERSION = 1

CORPUS_MANIFEST_FILE = "corpus_manifest.json"
TRAIN_INDEX_FILE = "train_corpus_index.parquet"
REPORT_MD_FILE = "corpus_report.md"

TRAIN_GROUP = "train"

STATUS_OK = "ok"
STATUS_BLOCKED_BY_INPUT = "blocked_by_input"
STATUS_BLOCKED = "blocked"

# Поля конверта, входящие в контрольную сумму содержимого.
# Это весь конверт, кроме payload: он идёт в сумму отдельно,
# разобранными колонками.
CHECKSUM_ENVELOPE: tuple[str, ...] = (
    "event_id",
    "client_id",
    "event_type",
    "source",
    "event_time",
)

# Поля профиля, которые в сумму НЕ входят: служебный индекс и
# трассировка к RAW. Сам профиль это одна строка на клиента, и
# в сумму она входит целиком.
PROFILE_SKIP: frozenset[str] = frozenset({"client_idx", "raw_file", "raw_row_group", "raw_row"})

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


class CorpusError(ValueError):
    """
    Корпус открывать нельзя: разделение непригодно, canonical
    пересобран после него либо клиент не входит в группу.
    """


@dataclass
class CorpusResult:
    report: dict
    outputs: list[Path] = field(default_factory=list)


# ============================================================
# НЕЗАВИСИМОСТЬ ПОПУЛЯЦИЙ
# ============================================================


def population_check(clients: dict[str, list[str]]) -> dict:
    """
    Популяции независимы: ни одного общего client_id.
    """

    overlaps: list[dict] = []

    names = sorted(clients)

    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            left, right = names[first], names[second]
            shared = sorted(set(clients[left]) & set(clients[right]))
            if shared:
                overlaps.append({"groups": [left, right], "clients": len(shared), "examples": shared[:5]})

    return {
        "compared": len(names) > 1,
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


def _profile_rows(store: CanonicalStore, client_id: str) -> list[dict]:
    """
    Профиль клиента: одна итоговая строка на границу выгрузки.

    Версий у профиля нет, поэтому отбирать по cutoff нечего:
    строка либо есть, либо клиента банк ещё не считал.
    """

    rows = store.profile_rows(client_id)

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

        profile_digest.extend(_profile_rows(store, client_id))

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
        "rule": (
            "по финальной очищенной истории событий до cutoff: конверт и payload видимых строк, "
            "итоговую строку профиля; служебные индексы, "
            "флаги и итоговые статусы выгрузки не входят"
        ),
    }

    checksum["content_sha256"] = sha256_bytes(
        dumps_json({key: checksum[key] for key in ("events", "profile")}).encode("utf-8")
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
    def open(corpus_dir: Path, canonical_dir: Path, processed_dir: Path | None = None,
             allow_unusable: bool = False, allow_short_horizon: bool = False) -> "TrainCorpus":
        """
        allow_unusable — режим диагностики: снимает проверки
        пригодности и свежести. Обучение открывает корпус без него.

        allow_short_horizon — согласиться на короткий горизонт.
        Технически такое разделение исправно, но договорённый
        горизонт оно не выполняет, и молчаливое согласие на это
        принимать нельзя.

        """

        corpus_dir = Path(corpus_dir)

        # Каталог набора: рядом с ним лежат маркеры этапов.
        processed = Path(processed_dir) if processed_dir is not None else corpus_dir.parent

        manifest = json.loads((corpus_dir / CORPUS_MANIFEST_FILE).read_text(encoding="utf-8"))

        corpus = manifest.get("train_corpus")

        if not allow_unusable:

            if not manifest.get("usable"):
                reasons = manifest.get("errors") or manifest.get("input_dependencies") or []
                raise CorpusError(
                    f"разделение со статусом {manifest.get('status')} непригодно для обучения: "
                    f"{'; '.join(reasons) or 'причина не названа'}. "
                    "Для диагностики откройте с allow_unusable=True"
                )

            actual = canonical_fingerprint(processed)

            if actual is None or actual != manifest["groups"][TRAIN_GROUP]["canonical_fingerprint"]:
                raise CorpusError(
                    "canonical группы train не тот, на котором построено разделение: "
                    "выполните этап split заново"
                )

            # Отпечаток говорит о согласии маркера с манифестом, а
            # корпус читает ФАЙЛЫ. Сверяются именно они.
            changed = edited_canonical_files(processed, canonical_dir)

            if changed:
                raise CorpusError(
                    "файлы canonical изменились после разделения ("
                    + ", ".join(changed[:5])
                    + "): выполните этапы canonical и split заново"
                )

            if not allow_short_horizon and not manifest.get("contract_met", True):
                reasons = (manifest.get("contract") or {}).get("reasons") or []
                raise CorpusError(
                    "разделение технически исправно, но договорённый горизонт не выполнен: "
                    + ("; ".join(reasons) or "причина не названа")
                    + ". Откройте с allow_short_horizon=True, если это осознанное решение"
                )

        if corpus is None:
            raise CorpusError("разрешённого train-корпуса в этом разделении нет")

        index_path = corpus_dir / TRAIN_INDEX_FILE

        index = pq.read_table(index_path)

        if index.num_rows != corpus["rows"]:
            raise CorpusError(
                f"индекс корпуса разошёлся с манифестом: строк в файле {index.num_rows}, "
                f"в манифесте {corpus['rows']}. Выполните этап split заново"
            )

        if not allow_unusable:

            # Число строк правленый индекс сохраняет, поэтому
            # сверяется сам файл: разрешение на строки обучения
            # выдаёт разделение, а не тот, кто правил parquet.
            expected = corpus.get("index_sha256")

            if not expected or sha256_file(index_path) != expected:
                raise CorpusError(
                    "индекс корпуса изменён после разделения или построен прежней версией "
                    "этапа: выполните этап split заново"
                )

        return TrainCorpus(
            store=CanonicalStore(canonical_dir),
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
            raise CorpusError(
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

    working = sorted(row["client_id"] for row in clients)
    excluded: list[str] = []

    times = visible.index.column("event_time").to_pylist()

    echo = store.report["raw"]

    return {
        "directory": source.group,
        "raw_period_start": echo["period_start"],
        "period_end": echo["period_end"],
        "window": window.as_dict(),
        "clients_total": len(clients),
        "clients_working": len(working),
        "clients": working,
        "clients_sha256": sha256_bytes("\n".join(working).encode("utf-8")),
        "excluded": {},
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


def build_corpus(sources: list[GroupInput], config: PreprocessingConfig, target: Path) -> CorpusResult:
    """
    Закрепляет группы, проверяет их независимость и общий мир,
    собирает разрешённый train-интерфейс и пишет артефакты.
    """

    target = Path(target)

    by_group = {source.group: source for source in sources}

    errors: list[str] = []
    input_dependencies: list[str] = []
    limitations: list[str] = []

    # Группы генерируются по одной, поэтому отсутствие соседней
    # не ошибка: разделение описывает то, что есть. Исключение
    # одно — train: без него не из чего собрать разрешённый
    # корпус, ради которого этап и существует.
    missing = sorted(set(config.windows) - set(by_group))

    if TRAIN_GROUP in missing:
        errors.append(
            f"группа {TRAIN_GROUP} отсутствует: без неё нет разрешённого корпуса, "
            "а он единственный вход будущего fit"
        )

    for name in missing:
        if name != TRAIN_GROUP:
            limitations.append(
                f"группа {name} отсутствует: разделение описывает только "
                + ", ".join(sorted(by_group))
                + "; оценка на ней невозможна, пока выгрузка не сделана"
            )

    stores: dict[str, CanonicalStore] = {}
    clients: dict[str, list[dict]] = {}
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

        if window.final_cutoff > store.period_end:
            errors.append(
                f"группа {name}: конечный cutoff {window.final_cutoff.isoformat()} позже границы выгрузки "
                f"{store.period_end.isoformat()}"
            )
            continue

        contract_groups[name] = {
            "period_start": store.period_start.isoformat(),
            "horizon_ok": store.period_start <= config.required_history_start,
            "passport_status": source.passport_status,
        }

        if store.period_start > config.required_history_start:
            limitations.append(
                f"группа {name}: история начинается {store.period_start.date()}, "
                f"а согласованный горизонт с {config.required_history_start.date()}"
            )

        if source.diagnostic_mode:
            limitations.append(f"группа {name}: canonical собран в режиме диагностики (паспорт contract_mismatch)")

        working = [row["client_id"] for row in clients[name]]

        visible[name] = collect_visible(store, working, window)

    population = population_check({name: [row["client_id"] for row in rows] for name, rows in clients.items()})

    for item in population["overlaps"]:
        errors.append(
            f"группы {item['groups'][0]} и {item['groups'][1]} делят {item['clients']} клиентов: "
            f"популяции не независимы (например {', '.join(item['examples'])})"
        )

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
                f"группа {name}: история с {item['period_start'][:10]}, "
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
        "groups_present": sorted(by_group),
        "groups_missing": missing,
        "groups_rule": (
            "группы генерируются по одной; отсутствие val или test это ограничение, "
            "а не ошибка, и потребитель обязан называть группу явно"
        ),
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

    manifest_path = target / CORPUS_MANIFEST_FILE
    write_json(manifest_path, report)
    outputs.append(manifest_path)

    md_path = target / REPORT_MD_FILE
    write_text(md_path, render_corpus_md(report))
    outputs.append(md_path)

    return CorpusResult(report=report, outputs=outputs)


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


def render_corpus_md(report: dict) -> str:

    out: list[str] = []

    out.append("# Разделение train / validation / test\n")
    out.append(f"Статус: **{report['status']}**.\n")
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
                    item["clients_total"],
                    item["clients_working"],
                    item["window"]["history_start"][:10],
                    item["raw_period_start"][:10],
                    item["window"]["final_cutoff"][:10],
                    f"{item['window']['target_start'][:10]} … {item['window']['target_end'][:10]}",
                ]
                for name, item in _ordered(report["groups"])
            ],
            [
                "группа",
                "клиентов",
                "рабочих",
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
                ["общих client_id", population["shared_clients"]],
            ],
            ["проверка", "результат"],
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
    "CORPUS_MANIFEST_FILE",
    "STAGE",
    "STAGE_VERSION",
    "STATUS_BLOCKED",
    "STATUS_BLOCKED_BY_INPUT",
    "STATUS_OK",
    "TRAIN_GROUP",
    "TRAIN_INDEX_FILE",
    "TRAIN_INDEX_SCHEMA",
    "GroupInput",
    "CorpusError",
    "CorpusResult",
    "TrainCorpus",
    "VisibleSet",
    "build_corpus",
    "canonical_fingerprint",
    "collect_visible",
    "group_summary",
    "mlm_target_eligible",
    "population_check",
    "render_corpus_md",
]
