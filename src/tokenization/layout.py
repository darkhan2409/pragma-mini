from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.preprocessing.artifacts import dumps_json, read_json, sha256_bytes, sha256_file, write_json, write_text

from .categorical import CATALOG_FILE
from .contract import CONFIG_FILE, FIT_MANIFEST_FILE
from .numeric import REGISTRY_FILE as NUMERIC_REGISTRY_FILE
from .numeric import FittedEncoder, load_encoders
from .report import render_vocab_md
from .settings import TokenizerConfig
from .text import BPE_FILE, PROBES, BpeModel, TextError, check_roundtrip, corpus_rows, load_bpe, train_bpe
from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


# ============================================================
# ИДЕЯ
# ============================================================
#
# Этап 4 назначает номера. Все виды токенов живут в одном
# пространстве ID непересекающимися диапазонами:
#
#   специальные -> ключи -> смысловые значения -> куски BPE
#
# Логически словари ключей и значений раздельны: ключ отвечает
# на вопрос «что это», значение — «чему равно». Физически их ID
# могут лежать в одной таблице embedding, и ровно поэтому
# диапазоны обязаны не пересекаться.
#
# Третьего словаря, который переучивал бы значения заново, нет:
# итог этапа это замороженный комплект и карта диапазонов.
#
# Назначение ID детерминировано: ключи в порядке имени, значения
# в порядке домена и типизированного значения, диапазоны в
# порядке своего номера, куски BPE в порядке модели. Новый fit
# вправе переставить номера и обязан получить новую версию.
# ============================================================


STAGE = "freeze"

SPECIAL_FILE = "special_tokens.json"
KEY_VOCAB_FILE = "key_vocab.json"
VALUE_VOCAB_FILE = "value_vocab.json"
KEY_VALUE_IDS_FILE = "key_value_ids.json"
LAYOUT_FILE = "vocab_layout.json"
MANIFEST_FILE = "tokenizer_manifest.json"
REPORT_FILE = "vocab_report.md"

PAD = "[PAD]"
UNK = "[UNK]"
MASK = "[MASK]"
EVT = "[EVT]"
USR = "[USR]"
MISSING = "[MISSING]"
INVALID = "[INVALID]"
EMPTY = "[EMPTY]"

# Порядок первых шести повторяет прежний словарь намеренно: это
# ничего не стоит и снимает один источник путаницы при чтении
# старого кода рядом с новым.
SPECIAL_TOKENS: tuple[str, ...] = (PAD, UNK, MASK, EVT, USR, MISSING, INVALID, EMPTY)

SPECIAL_ROLES: dict[str, str] = {
    PAD: "выравнивание batch; токенизатор его не пишет никогда",
    UNK: "значение, которого на train не было, и число без шкалы",
    MASK: "зарезервирован для Masker; в сохранённых данных не встречается никогда",
    EVT: "начало события, один на событие, ставит токенизатор",
    USR: "начало представления профиля, один на профиль, ставит токенизатор",
    MISSING: "объявленный у этого типа события ключ без значения",
    INVALID: "невозможное число: NaN, бесконечность или запрещённый доменом знак",
    EMPTY: "текст, в котором после нормализации не осталось ни одного символа",
}

KIND_CATEGORICAL = "categorical"
KIND_BUCKET = "bucket"

# Файлы, которые и есть словарь. Отпечаток по ним отвечает на
# вопрос «тот же ли это словарь», отдельно от вопроса «те же ли
# у него входы».
VOCABULARY_FILES: frozenset[str] = frozenset(
    {
        SPECIAL_FILE,
        KEY_VOCAB_FILE,
        VALUE_VOCAB_FILE,
        KEY_VALUE_IDS_FILE,
        LAYOUT_FILE,
        BPE_FILE,
        NUMERIC_REGISTRY_FILE,
        CATALOG_FILE,
        CONFIG_FILE,
    }
)


class LayoutError(ValueError):
    """
    Пространство ID собрать нельзя.
    """


def manifest_id(manifest: dict) -> str:
    """
    Тождество замороженного комплекта: отпечаток всего манифеста
    без самого номера.

    Считается и при заморозке, и при загрузке одной функцией:
    второго правила «что такое этот словарь» быть не должно.
    """

    core = {name: value for name, value in manifest.items() if name != "artifact_id"}

    return sha256_bytes(dumps_json(core).encode("utf-8"))[:12]


@dataclass
class LayoutResult:
    report: dict
    outputs: list[Path] = field(default_factory=list)


# ------------------------------------------------------------
# СБОРКА
# ------------------------------------------------------------


def _key_rows(catalog: dict) -> list[dict]:
    """
    Ключи, получающие код, в устойчивом порядке.
    """

    return sorted(catalog["keys"], key=lambda row: row["key"])


def _value_rows(catalog: dict, registry: dict) -> list[dict]:
    """
    Значения в порядке: сначала категории по доменам, затем
    числовые диапазоны по ключам.
    """

    rows: list[dict] = []

    for domain in sorted(catalog["domains"], key=lambda item: item["name"]):

        for value in domain["values"]:
            rows.append(
                {
                    "kind": KIND_CATEGORICAL,
                    "domain": domain["name"],
                    "key": None,
                    "value_type": value["value_type"],
                    "value": value["value"],
                    "label": f"{domain['name']}={value['value']}",
                    "lower": None,
                    "upper": None,
                    "train_count": value["count"],
                    "rare": value["rare"],
                }
            )

    for key in sorted(registry["encoders"]):

        entry = registry["encoders"][key]

        for bucket, count in zip(entry["buckets"], entry["distribution"]["buckets"]):
            rows.append(
                {
                    "kind": KIND_BUCKET,
                    "domain": None,
                    "key": key,
                    "value_type": "number",
                    "value": str(bucket["index"]),
                    "label": bucket["label"],
                    "lower": bucket["lower"],
                    "upper": bucket["upper"],
                    "train_count": count,
                    "rare": False,
                }
            )

    return rows


def build_layout(
    target: Path,
    processed_dir: Path,
    config: TokenizerConfig,
    group: str = "train",
) -> LayoutResult:
    """
    Замораживает словари, разбиение текста и карту диапазонов.
    """

    target = Path(target)

    catalog_path = target / CATALOG_FILE
    registry_path = target / NUMERIC_REGISTRY_FILE

    for path in (catalog_path, registry_path):
        if not path.exists():
            raise LayoutError(f"нет {path}: заморозке предшествуют этапы values и buckets")

    catalog = read_json(catalog_path)
    registry = read_json(registry_path)
    fit = read_json(target / FIT_MANIFEST_FILE)

    for name, artifact in (("каталог значений", catalog), ("числовые границы", registry)):
        if artifact["config_sha256"] != config.sha256():
            raise LayoutError(f"{name} собран другой конфигурацией: выполните этапы заново")

    # --- BPE ---

    text_keys = tuple(
        row["key"]
        for row in catalog["keys"]
        if row["value_kind"] == "text" and row["key"] not in config.text_keys_as_categorical
    )

    bpe = train_bpe(target, config.bpe, text_keys)

    if bpe.enabled:

        texts = sorted({text for _key, text, _count in corpus_rows(target, text_keys)})

        broken = check_roundtrip(bpe, [*PROBES, *texts])

        if broken:
            raise LayoutError(
                "разбиение текста не обратимо на "
                + ", ".join(repr(item) for item in broken[:3])
                + ": байтовый алфавит обязан кодировать любой текст без потерь"
            )

    # --- пространство ID ---

    keys = _key_rows(catalog)
    values = _value_rows(catalog, registry)

    first_key_id = len(SPECIAL_TOKENS)
    first_value_id = first_key_id + len(keys)
    bpe_offset = first_value_id + len(values)
    size = bpe_offset + bpe.size

    key_ids = {row["key"]: first_key_id + index for index, row in enumerate(keys)}

    value_ids: dict[tuple, int] = {}

    for index, row in enumerate(values):

        row["id"] = first_value_id + index

        if row["kind"] == KIND_CATEGORICAL:
            value_ids[(row["domain"], row["value_type"], row["value"])] = row["id"]
        else:
            value_ids[(row["key"], KIND_BUCKET, row["value"])] = row["id"]

    # --- кандидаты по ключам ---

    candidates: dict[str, dict] = {}

    for row in keys:

        key = row["key"]

        if row["value_kind"] == KIND_CATEGORICAL:
            ids = [
                value_ids[(row["domain"], item["value_type"], item["value"])]
                for item in row["candidates"]
            ]
            candidates[key] = {"kind": KIND_CATEGORICAL, "domain": row["domain"], "value_ids": ids}

        elif row["value_kind"] == "numeric":
            entry = registry["encoders"][key]
            ids = [value_ids[(key, KIND_BUCKET, str(bucket["index"]))] for bucket in entry["buckets"]]
            candidates[key] = {"kind": KIND_BUCKET, "domain": None, "value_ids": ids}

        else:
            candidates[key] = {"kind": "text", "domain": None, "value_ids": [], "bpe": bpe.enabled}

    layout = {
        "format_version": FORMAT_VERSION,
        "ranges": {
            "special": [0, first_key_id],
            "keys": [first_key_id, first_value_id],
            "values": [first_value_id, bpe_offset],
            "bpe": [bpe_offset, size],
        },
        "bpe_offset": bpe_offset,
        "size": size,
        "sizes": {
            "special": len(SPECIAL_TOKENS),
            "keys": len(keys),
            "categorical": sum(1 for row in values if row["kind"] == KIND_CATEGORICAL),
            "buckets": sum(1 for row in values if row["kind"] == KIND_BUCKET),
            "values": len(values),
            "bpe": bpe.size,
            "total": size,
        },
        "invariants": [
            "диапазоны не пересекаются и покрывают всё пространство без дыр",
            "ID куска BPE это bpe_offset плюс его номер в модели разбиения",
            "специальный токен в слоте значения означает не значение, а его отсутствие или невозможность",
            "ключи и значения логически раздельны, физически делят одну таблицу embedding",
        ],
        "rule": "специальные -> ключи -> смысловые значения -> куски BPE",
    }

    # --- запись ---

    outputs: list[Path] = []

    def write(name: str, payload: dict) -> None:
        path = target / name
        write_json(path, payload)
        outputs.append(path)

    write(
        SPECIAL_FILE,
        {
            "ids": {name: index for index, name in enumerate(SPECIAL_TOKENS)},
            "roles": SPECIAL_ROLES,
            "never_written": [PAD, MASK],
            "marker_owner": "tokenizer",
        },
    )

    write(
        KEY_VOCAB_FILE,
        {
            "first_key_id": first_key_id,
            "size": len(keys),
            "keys": [
                {
                    "key": row["key"],
                    "id": key_ids[row["key"]],
                    "value_kind": row["value_kind"],
                    "unit": row["unit"],
                    "origin": row["origin"],
                    "weight_rule": row["weight_rule"],
                    "domain": row["domain"],
                    "observed_in_train": row["observed_in_train"],
                    "physical_fields": row["physical_fields"],
                }
                for row in keys
            ],
            # Ссылки перечислены рядом намеренно: кода у них нет,
            # но потребитель обязан знать, что они существуют и
            # приходят метаданными.
            "link_keys": sorted(row["key"] for row in fit["keys"]["rows"] if row["role"] == "link"),
        },
    )

    write(
        VALUE_VOCAB_FILE,
        {
            "first_value_id": first_value_id,
            "size": len(values),
            "values": values,
        },
    )

    write(KEY_VALUE_IDS_FILE, {"keys": candidates, "rule": "кандидаты ключа в порядке его домена"})

    write(LAYOUT_FILE, layout)

    if bpe.enabled:
        outputs.append(target / BPE_FILE)

    # --- манифест ---

    artifacts = {
        path.relative_to(target).as_posix(): sha256_file(path)
        for path in [*outputs, target / CONFIG_FILE, catalog_path, registry_path, target / FIT_MANIFEST_FILE]
    }

    for name, digest in sorted((fit.get("artifacts") or {}).items()):
        artifacts.setdefault(name, digest)

    manifest = {
        "stage": STAGE,
        "schema_version": SCHEMA_VERSION,
        "format_version": FORMAT_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "group": group,
        "fit_end": fit["fit_end"],
        "fit_content_sha256": fit["fit_content_sha256"],
        "config_sha256": config.sha256(),
        "readiness": fit["readiness"],
        "layout": layout,
        "bpe": bpe.info,
        "artifacts": dict(sorted(artifacts.items())),
        "input_files": fit["input_files"],
        "versions": fit["versions"],
        "sources": _sources(),
        "libraries": _libraries(),
        "limitations": fit["limitations"],
        "rules": {
            "frozen": "словарь после fit неизменен: transform не добавляет ни одного токена",
            "refit": "новый fit вправе переставить ID и обязан получить новую версию артефактов; "
                     "совместимость со старым checkpoint автоматически не обещается",
            "markers": "маркеры события и профиля ставит токенизатор, marker_owner = tokenizer",
        },
    }

    # Отпечаток САМОГО словаря, без происхождения входов.
    #
    # artifact_id меняется и тогда, когда изменилось окружение
    # набора: например, другими стали validation и test, и вместе
    # с ними манифест разделения. Словарь при этом обязан
    # остаться прежним, и доказывает это отдельное число.
    manifest["vocab_sha256"] = sha256_bytes(
        dumps_json(
            {
                name: digest
                for name, digest in manifest["artifacts"].items()
                if name in VOCABULARY_FILES
            }
        ).encode("utf-8")
    )

    # Тождество комплекта считается по ВСЕМУ манифесту, а не по
    # одному списку файлов: версии, вердикт готовности,
    # происхождение входов и отпечаток словаря тоже входят в
    # то, чем этот комплект является. Иначе подмена любого поля
    # манифеста оставляла бы прежний номер.
    manifest["artifact_id"] = manifest_id(manifest)

    path = target / MANIFEST_FILE
    write_json(path, manifest)
    outputs.append(path)

    path = target / REPORT_FILE
    write_text(path, render_vocab_md(manifest, values, bpe))
    outputs.append(path)

    return LayoutResult(report=manifest, outputs=outputs)


def _sources() -> dict[str, str]:
    """
    sha256 модулей, которыми собран этот словарь.
    """

    from src.tokenization import categorical, contract, corpus, layout, numeric, report, scan, schema, settings, text

    modules = {
        "categorical": categorical,
        "contract": contract,
        "corpus": corpus,
        "layout": layout,
        "numeric": numeric,
        "report": report,
        "scan": scan,
        "schema": schema,
        "settings": settings,
        "text": text,
    }

    return {
        name: sha256_file(Path(module.__file__))
        for name, module in sorted(modules.items())
    }


def _libraries() -> dict[str, str]:

    import sys

    import numpy
    import pyarrow

    out = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "pyarrow": pyarrow.__version__,
    }

    try:
        import tokenizers

        out["tokenizers"] = tokenizers.__version__
    except ImportError:  # pragma: no cover - библиотека объявлена зависимостью
        pass

    return out


# ------------------------------------------------------------
# ЗАМОРОЖЕННЫЙ КОМПЛЕКТ
# ------------------------------------------------------------


@dataclass
class FrozenArtifacts:
    """
    Готовые словари, которыми кодируют и расшифровывают.

    Ничего не обучает и не меняет: при загрузке проверяет, что
    каждый артефакт тот же самый, каким его заморозили.
    """

    directory: Path
    manifest: dict
    layout: dict
    specials: dict[str, int]
    key_ids: dict[str, int]
    key_info: dict[str, dict]
    domain_of: dict[str, str]
    value_ids: dict[tuple, int]
    value_rows: list[dict]
    candidates: dict[str, dict]
    encoders: dict[str, FittedEncoder]
    declared_by_event_type: dict[str, tuple[str, ...]]
    link_keys: frozenset[str]
    profile_keys: tuple[str, ...]
    bpe: BpeModel

    @property
    def first_value_id(self) -> int:
        return self.layout["ranges"]["values"][0]

    @property
    def bpe_offset(self) -> int:
        return self.layout["bpe_offset"]

    @property
    def size(self) -> int:
        return self.layout["size"]

    def special(self, name: str) -> int:
        return self.specials[name]

    def key_id(self, key: str) -> int | None:
        return self.key_ids.get(key)

    def categorical_id(self, key: str, value_type: str, value: str) -> int | None:
        return self.value_ids.get((self.domain_of[key], value_type, value))

    def bucket_id(self, key: str, index: int) -> int:
        return self.value_ids[(key, KIND_BUCKET, str(index))]

    def describe(self, token_id: int) -> dict:
        """
        Человекочитаемая расшифровка одного ID.
        """

        if token_id < len(SPECIAL_TOKENS):
            name = SPECIAL_TOKENS[token_id]
            return {"id": token_id, "kind": "special", "label": name, "role": SPECIAL_ROLES[name]}

        keys_range = self.layout["ranges"]["keys"]

        if keys_range[0] <= token_id < keys_range[1]:
            key = next(name for name, value in self.key_ids.items() if value == token_id)
            return {"id": token_id, "kind": "key", "label": key}

        values_range = self.layout["ranges"]["values"]

        if values_range[0] <= token_id < values_range[1]:
            row = self.value_rows[token_id - values_range[0]]
            return {"id": token_id, "kind": row["kind"], "label": row["label"]}

        piece = self.bpe.piece(token_id - self.bpe_offset)

        return {"id": token_id, "kind": "bpe", "label": piece}

    def verify(self) -> None:
        """
        Пересчитывает тождество комплекта и отпечатки артефактов.

        Сначала манифест: он сам называет, что именно проверять,
        и доверять непроверенному списку нельзя. Правленый
        манифест с прежним номером — это и есть подмена словаря.
        """

        actual = manifest_id(self.manifest)

        if actual != self.manifest.get("artifact_id"):
            raise LayoutError(
                f"манифест словаря изменён после заморозки: номер комплекта "
                f"{self.manifest.get('artifact_id')!r}, а по содержимому {actual!r}. "
                "Соберите словарь заново"
            )

        for name, digest in sorted(self.manifest["artifacts"].items()):

            path = self.directory / name

            if not path.exists():
                raise LayoutError(f"артефакт {name} пропал из {self.directory}")

            if sha256_file(path) != digest:
                raise LayoutError(
                    f"артефакт {name} изменился после заморозки: кодировать им нельзя, "
                    "соберите словарь заново"
                )

    @staticmethod
    def load(directory: Path) -> "FrozenArtifacts":

        directory = Path(directory)

        path = directory / MANIFEST_FILE

        if not path.exists():
            raise LayoutError(f"нет {path}: словарь не заморожен, выполните freeze")

        manifest = read_json(path)

        specials = read_json(directory / SPECIAL_FILE)["ids"]
        key_vocab = read_json(directory / KEY_VOCAB_FILE)
        value_vocab = read_json(directory / VALUE_VOCAB_FILE)
        candidates = read_json(directory / KEY_VALUE_IDS_FILE)["keys"]
        registry = read_json(directory / NUMERIC_REGISTRY_FILE)
        catalog = read_json(directory / CATALOG_FILE)

        value_ids: dict[tuple, int] = {}

        for row in value_vocab["values"]:
            if row["kind"] == KIND_CATEGORICAL:
                value_ids[(row["domain"], row["value_type"], row["value"])] = row["id"]
            else:
                value_ids[(row["key"], KIND_BUCKET, row["value"])] = row["id"]

        bpe = (
            load_bpe(directory / BPE_FILE)
            if manifest["bpe"].get("enabled")
            else BpeModel(enabled=False, keys=(), info=manifest["bpe"])
        )

        artifacts = FrozenArtifacts(
            directory=directory,
            manifest=manifest,
            layout=manifest["layout"],
            specials=specials,
            key_ids={row["key"]: row["id"] for row in key_vocab["keys"]},
            key_info={row["key"]: row for row in key_vocab["keys"]},
            domain_of={row["key"]: row["domain"] for row in key_vocab["keys"] if row["domain"]},
            value_ids=value_ids,
            value_rows=value_vocab["values"],
            candidates=candidates,
            encoders=load_encoders(registry),
            declared_by_event_type={
                event_type: tuple(keys)
                for event_type, keys in catalog["declared_by_event_type"].items()
            },
            link_keys=frozenset(key_vocab["link_keys"]),
            profile_keys=tuple(
                row["key"] for row in key_vocab["keys"] if row["origin"] == "profile"
            ),
            bpe=bpe,
        )

        artifacts.verify()

        return artifacts


__all__ = [
    "EMPTY",
    "EVT",
    "INVALID",
    "KEY_VALUE_IDS_FILE",
    "KEY_VOCAB_FILE",
    "KIND_BUCKET",
    "KIND_CATEGORICAL",
    "LAYOUT_FILE",
    "MANIFEST_FILE",
    "MASK",
    "MISSING",
    "PAD",
    "REPORT_FILE",
    "SPECIAL_FILE",
    "SPECIAL_ROLES",
    "SPECIAL_TOKENS",
    "STAGE",
    "UNK",
    "USR",
    "VALUE_VOCAB_FILE",
    "FrozenArtifacts",
    "LayoutError",
    "LayoutResult",
    "build_layout",
]
