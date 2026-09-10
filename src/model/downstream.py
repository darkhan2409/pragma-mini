from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from src.generator.config import RUNS_DIR
from src.preprocessing.artifacts import write_json, write_text
from src.preprocessing.config import processed_dir as prep_processed_dir
from src.preprocessing.raw import read_manifest
from src.tokenizer.build import client_runs, iter_client_blocks
from src.tokenizer.config import artifacts_dir, tokenized_dir, vocab_dir
from src.tokenizer.dataset import collate

from .checkpoint import load_checkpoint
from .data import ClientStore, session_keys_from_examples
from .history_batching import metadata_from_examples, prepare_history_batch, to_model_inputs
from .trainer import TrainConfig, Trainer, autocast_for, load_environment


# ============================================================
# ИДЕЯ
# ============================================================
#
# MLM меряет, насколько модель предсказывает скрытое поле.
# Downstream меряет другое: помогает ли вектор клиента решать
# задачу, которую никто не тренировал.
#
# Задача одна: product_open_90d, открыл ли клиент хоть один
# продукт за 90 дней после среза признаков. Метка живёт только
# в RAW и в processed не попадает намеренно, поэтому читается
# отдельно и никогда не участвует в признаках.
#
# Cutoff здесь ровно один и выбран не нами: окно метки
# начинается в feature_end, значит признаки это вся история
# строго раньше feature_end. Один пример на клиента.
#
# Три модели на одних и тех же клиентах и сплитах:
#
#   1. признаки            -> boosting
#   2. замороженный [USR]  -> логистическая регрессия
#   3. признаки и [USR]    -> тот же boosting
#
# Всё, что учится преобразованиям (кодировщик категорий,
# нормировка), обучается ТОЛЬКО на train.
# ============================================================


LABEL = "product_open_90d"

GROUPS: tuple[str, ...] = ("train", "val", "test")

CUTOFF_INDEX = "cutoff_index.parquet"

DOWNSTREAM_DATASET = "downstream"

EVENT_TYPES: tuple[str, ...] = (
    "profile_snapshot",
    "product_event",
    "transaction",
    "communication",
    "banner",
    "app_screen",
    "app_operation",
)

PRODUCT_TYPES: tuple[str, ...] = (
    "debit_card",
    "credit_card",
    "cash_loan",
    "deposit",
    "insurance",
)

FUNNEL_STAGES: tuple[str, ...] = ("view", "application", "kyc", "approved", "rejected")

OPERATION_STATUSES: tuple[str, ...] = ("success", "failed", "cancelled")

OPERATION_DOMAINS: tuple[str, ...] = (
    "auth",
    "cards",
    "deposits",
    "loans",
    "market",
    "payments",
    "support",
    "transfers",
)

DIRECTIONS: tuple[str, ...] = ("debit", "credit")

BANNER_ACTIONS: tuple[str, ...] = ("shown", "clicked")

WINDOWS: tuple[int, ...] = (30, 90)

# Категориальные поля профиля: их кодирует OrdinalEncoder,
# обученный на train.
PROFILE_CATEGORICAL: tuple[str, ...] = (
    "gender",
    "family_status",
    "education",
    "region",
    "housing_type",
    "income_type",
    "industry",
)

PROFILE_NUMERIC: tuple[str, ...] = (
    "age",
    "children",
    "pensioner",
    "declared_income",
    "salary_day",
    "relationship_months",
    "contracts_count",
    "active_contracts",
    "holds_credit_card",
    "holds_debit_card",
    "holds_deposit",
    "credit_limit",
    "credit_utilization",
)

EVENT_COLUMNS: tuple[str, ...] = (
    "client_id",
    "seq",
    "ts",
    "event_type",
    "transaction__amount",
    "transaction__direction",
    "transaction__mcc",
    "transaction__is_online",
    "transaction__is_subscription",
    "product_event__product_type",
    "app_screen__funnel_stage",
    "app_operation__status",
    "app_operation__domain",
    "banner__action",
    "communication__delivered",
)


class DownstreamError(RuntimeError):
    """Downstream нельзя собрать на этих данных."""


# ============================================================
# ПРИМЕРЫ И МЕТКИ
# ============================================================


@dataclass(frozen=True)
class Examples:
    """
    По одному примеру на клиента на срезе признаков.
    """

    cutoff: np.datetime64
    rows: dict[str, list[dict]]
    labels: dict[int, int]

    # Клиенты, у которых на этом срезе примера нет. Считаются
    # и называются, а не подставляются.
    skipped: dict[str, int]

    @property
    def clients(self) -> dict[str, list[int]]:
        return {
            group: [int(row["client_id"]) for row in rows]
            for group, rows in self.rows.items()
        }

    def summary(self) -> dict:
        return {
            "cutoff": str(self.cutoff),
            "label": LABEL,
            "groups": {
                group: {
                    "clients": len(rows),
                    "positives": int(
                        sum(self.labels[int(row["client_id"])] for row in rows)
                    ),
                    "positive_rate": round(
                        sum(self.labels[int(row["client_id"])] for row in rows) / len(rows),
                        4,
                    )
                    if rows
                    else 0.0,
                }
                for group, rows in self.rows.items()
            },
        }


def build_examples(processed: Path, raw: Path) -> Examples:
    """
    Строки примеров на срезе признаков и метки к ним.

    Строки берутся из cutoff_index, а не из каталогов датасетов:
    на этом срезе примеры есть только у train-клиентов (это
    test_time), а нужны все три группы.
    """

    processed = Path(processed)

    manifest = read_manifest(Path(raw))

    cutoff = np.datetime64(manifest.feature_end, "us")

    index = pq.read_table(processed / CUTOFF_INDEX)

    stamps = index.column("cutoff").to_numpy().astype("datetime64[us]")

    at_cutoff = index.filter(stamps == cutoff)

    if at_cutoff.num_rows == 0:
        raise DownstreamError(
            f"в {CUTOFF_INDEX} нет строк на срезе признаков {cutoff}"
        )

    labels_table = pq.read_table(Path(raw) / "labels.parquet")

    starts = labels_table.column("label_start").to_numpy().astype("datetime64[us]")

    if not bool((starts == cutoff).all()):
        raise DownstreamError(
            "окно метки начинается не на срезе признаков: метка и признаки разъехались"
        )

    labels = {
        int(client): int(value)
        for client, value in zip(
            labels_table.column("client_id").to_pylist(),
            labels_table.column(LABEL).to_pylist(),
        )
    }

    rows: dict[str, list[dict]] = {group: [] for group in GROUPS}

    skipped: dict[str, int] = {}

    for row in at_cutoff.to_pylist():

        if not row["valid"]:
            reason = str(row["skip_reason"])
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        client_id = int(row["client_id"])

        if client_id not in labels:
            raise DownstreamError(f"нет метки клиента {client_id}")

        rows[str(row["client_group"])].append(
            {**row, "dataset": DOWNSTREAM_DATASET}
        )

    empty = [group for group, items in rows.items() if not items]

    if empty:
        raise DownstreamError(f"на срезе признаков нет клиентов групп {empty}")

    return Examples(cutoff=cutoff, rows=rows, labels=labels, skipped=skipped)


# ============================================================
# ПРИЗНАКИ
# ============================================================


def feature_names() -> list[str]:
    """
    Имена признаков в фиксированном порядке.
    """

    names = [f"profile__{name}" for name in (*PROFILE_NUMERIC, *PROFILE_CATEGORICAL)]

    names.append("history__n_events")
    names.append("history__span_days")

    for event_type in EVENT_TYPES:
        names.append(f"count__{event_type}")
        names.append(f"days_since__{event_type}")
        for window in WINDOWS:
            names.append(f"count__{event_type}__{window}d")

    for direction in DIRECTIONS:
        names.append(f"tx__{direction}__sum")
        names.append(f"tx__{direction}__mean")
        names.append(f"tx__{direction}__max")
        names.append(f"tx__{direction}__count")

    names.append("tx__distinct_mcc")
    names.append("tx__online_share")
    names.append("tx__subscription_count")

    for product in PRODUCT_TYPES:
        names.append(f"product__{product}")

    for stage in FUNNEL_STAGES:
        names.append(f"funnel__{stage}")

    for status in OPERATION_STATUSES:
        names.append(f"operation__{status}")

    for domain in OPERATION_DOMAINS:
        names.append(f"domain__{domain}")

    for action in BANNER_ACTIONS:
        names.append(f"banner__{action}")

    names.append("communication__delivered_share")

    return names


FEATURE_NAMES: tuple[str, ...] = tuple(feature_names())

N_CATEGORICAL = len(PROFILE_CATEGORICAL)

# Категориальные признаки идут подряд сразу за числовыми
# профиля: их позиции нужны кодировщику и boosting.
CATEGORICAL_SLICE = slice(len(PROFILE_NUMERIC), len(PROFILE_NUMERIC) + N_CATEGORICAL)


def _share(part, whole) -> float:
    return float(part) / float(whole) if whole else 0.0


def client_aggregates(
    block, lo: int, hi: int, cutoff: np.datetime64, seq_end: int
) -> dict[str, float]:
    """
    Агрегаты истории одного примера.

    Берётся строго префикс seq < seq_end, и ни одно событие
    в нём не может быть позже cutoff: это проверяется, а не
    предполагается.
    """

    take = slice(lo, lo + int(seq_end))

    ts = block["ts"][take]

    if ts.size and ts.max() >= cutoff:
        raise DownstreamError(
            f"событие {ts.max()} не раньше среза {cutoff}: признак смотрит в будущее"
        )

    event_type = block["event_type"][take]

    out: dict[str, float] = {}

    out["history__n_events"] = float(ts.size)
    out["history__span_days"] = (
        float((ts.max() - ts.min()) / np.timedelta64(1, "D")) if ts.size else 0.0
    )

    age_days = (
        (cutoff - ts) / np.timedelta64(1, "D") if ts.size else np.zeros(0, dtype=float)
    )

    for name in EVENT_TYPES:

        mine = event_type == name

        out[f"count__{name}"] = float(mine.sum())

        out[f"days_since__{name}"] = float(age_days[mine].min()) if mine.any() else -1.0

        for window in WINDOWS:
            out[f"count__{name}__{window}d"] = float((mine & (age_days <= window)).sum())

    # --- транзакции ---------------------------------------
    amount = block["transaction__amount"][take]
    direction = block["transaction__direction"][take]

    is_transaction = event_type == "transaction"

    for name in DIRECTIONS:

        mine = is_transaction & (direction == name)

        values = amount[mine]
        values = values[~np.isnan(values)]

        out[f"tx__{name}__sum"] = float(values.sum()) if values.size else 0.0
        out[f"tx__{name}__mean"] = float(values.mean()) if values.size else 0.0
        out[f"tx__{name}__max"] = float(values.max()) if values.size else 0.0
        out[f"tx__{name}__count"] = float(mine.sum())

    mcc = block["transaction__mcc"][take][is_transaction]

    out["tx__distinct_mcc"] = float(len({value for value in mcc if value is not None}))

    online = block["transaction__is_online"][take][is_transaction]

    out["tx__online_share"] = _share(
        sum(1 for value in online if value is True), is_transaction.sum()
    )

    subscription = block["transaction__is_subscription"][take][is_transaction]

    out["tx__subscription_count"] = float(
        sum(1 for value in subscription if value is True)
    )

    # --- продукты, воронка, операции, баннеры -------------
    product = block["product_event__product_type"][take]

    for name in PRODUCT_TYPES:
        out[f"product__{name}"] = float((product == name).sum())

    funnel = block["app_screen__funnel_stage"][take]

    for name in FUNNEL_STAGES:
        out[f"funnel__{name}"] = float((funnel == name).sum())

    status = block["app_operation__status"][take]

    for name in OPERATION_STATUSES:
        out[f"operation__{name}"] = float((status == name).sum())

    domain = block["app_operation__domain"][take]

    for name in OPERATION_DOMAINS:
        out[f"domain__{name}"] = float((domain == name).sum())

    action = block["banner__action"][take]

    for name in BANNER_ACTIONS:
        out[f"banner__{name}"] = float((action == name).sum())

    delivered = block["communication__delivered"][take]

    is_communication = event_type == "communication"

    out["communication__delivered_share"] = _share(
        sum(1 for value in delivered if value is True), is_communication.sum()
    )

    return out


def _block_arrays(table) -> dict:

    out = {
        "client_id": table.column("client_id").to_numpy(),
        "ts": table.column("ts").to_numpy().astype("datetime64[us]"),
        "event_type": np.asarray(table.column("event_type").to_pylist(), dtype=object),
        "transaction__amount": table.column("transaction__amount")
        .to_numpy(zero_copy_only=False)
        .astype(np.float64),
    }

    for name in EVENT_COLUMNS:
        if name in out or name == "seq":
            continue
        out[name] = np.asarray(table.column(name).to_pylist(), dtype=object)

    return out


def build_features(
    processed: Path, group: str, rows: list[dict], cutoff: np.datetime64
) -> np.ndarray:
    """
    Матрица признаков группы в порядке строк.
    """

    processed = Path(processed)

    source = processed / "clients" / f"{group}_clients" / "events.parquet"

    wanted = {int(row["client_id"]): index for index, row in enumerate(rows)}

    seq_end = {int(row["client_id"]): int(row["seq_end"]) for row in rows}

    aggregates: dict[int, dict[str, float]] = {}

    for table in iter_client_blocks(source, columns=list(EVENT_COLUMNS)):

        column = table.column("client_id").to_numpy()

        if column.size == 0:
            continue

        block = _block_arrays(table)

        for value, lo, hi in client_runs(column):

            client_id = int(value)

            if client_id not in wanted:
                continue

            if seq_end[client_id] > hi - lo:
                raise DownstreamError(
                    f"клиент {client_id}: seq_end {seq_end[client_id]} больше "
                    f"числа событий {hi - lo}"
                )

            aggregates[client_id] = client_aggregates(
                block, lo, hi, cutoff, seq_end[client_id]
            )

    missing = sorted(set(wanted) - set(aggregates))

    if missing:
        raise DownstreamError(f"{group}: нет событий клиентов {missing[:5]}")

    # --- профиль as-of ------------------------------------
    profile = pq.read_table(
        processed / "clients" / f"{group}_clients" / "profile.parquet"
    )

    stamps = profile.column("ts").to_numpy().astype("datetime64[us]")
    owners = profile.column("client_id").to_numpy().astype(np.int64)

    wanted_snapshot = {
        int(row["client_id"]): np.datetime64(row["snapshot_ts"], "us") for row in rows
    }

    position = {
        (int(owner), stamp): index
        for index, (owner, stamp) in enumerate(zip(owners, stamps))
    }

    columns = {
        name: profile.column(name).to_pylist()
        for name in (*PROFILE_NUMERIC, *PROFILE_CATEGORICAL)
    }

    matrix = np.zeros((len(rows), len(FEATURE_NAMES)), dtype=object)

    for index, row in enumerate(rows):

        client_id = int(row["client_id"])

        snapshot = wanted_snapshot[client_id]

        if snapshot >= cutoff:
            raise DownstreamError(
                f"клиент {client_id}: снимок профиля {snapshot} не раньше среза {cutoff}"
            )

        place = position.get((client_id, snapshot))

        if place is None:
            raise DownstreamError(f"клиент {client_id}: нет снимка профиля на {snapshot}")

        values: list = []

        for name in PROFILE_NUMERIC:
            value = columns[name][place]
            values.append(np.nan if value is None else float(value))

        for name in PROFILE_CATEGORICAL:
            value = columns[name][place]
            values.append("" if value is None else str(value))

        history = aggregates[client_id]

        values.extend(history[name] for name in FEATURE_NAMES[len(values):])

        matrix[index] = values

    return matrix


# ============================================================
# ЭМБЕДДИНГИ
# ============================================================


def build_embeddings(
    env,
    trainer: Trainer,
    root: Path,
    group: str,
    rows: list[dict],
    batch_size: int = 8,
) -> np.ndarray:
    """
    Замороженный вектор клиента на том же срезе.

    gather не передаётся: голова здесь не нужна, нужен только
    выход History Encoder в позиции 0.
    """

    store = ClientStore.from_rows(
        root, group, env.vocab_dir, rows, sessions=trainer.config.uses_sessions
    )

    order = {int(row["client_id"]): index for index, row in enumerate(rows)}

    out = np.zeros((len(rows), trainer.model_config.d_model), dtype=np.float32)

    trainer.eval_mode()

    with torch.inference_mode():

        for start in range(0, len(store), batch_size):

            chunk = range(start, min(start + batch_size, len(store)))

            examples = store.examples(chunk)

            keys = session_keys_from_examples(examples)

            history = prepare_history_batch(
                collate(examples),
                metadata_from_examples(examples),
                trainer.config.max_events_per_history,
                masker=None,
                session_keys=keys if trainer.config.uses_sessions else None,
                structure=trainer.model_config.structure,
            )

            inputs = to_model_inputs(history, trainer.model_config, trainer.device)

            with autocast_for(trainer.precision, trainer.device):
                result = trainer.backbone(inputs, trainer.config.event_microbatch)

            vectors = result.client_embedding.detach().float().cpu().numpy()

            for position, example in enumerate(examples):
                out[order[int(example.client_id)]] = vectors[position]

    return out


# ============================================================
# МОДЕЛИ
# ============================================================


BOOSTING = {
    "max_iter": 200,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "l2_regularization": 1.0,
    "early_stopping": False,
}


def fit_encoder(train_features: np.ndarray):
    """
    Кодировщик категорий, обученный ТОЛЬКО на train.
    """

    from sklearn.preprocessing import OrdinalEncoder

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    encoder.fit(train_features[:, CATEGORICAL_SLICE].astype(str))

    return encoder


def encode(features: np.ndarray, encoder) -> np.ndarray:

    numeric = np.delete(features, np.s_[CATEGORICAL_SLICE], axis=1).astype(np.float64)

    categorical = encoder.transform(features[:, CATEGORICAL_SLICE].astype(str))

    # Категории уходят в конец: их позиции знает boosting.
    return np.concatenate([numeric, categorical.astype(np.float64)], axis=1)


def categorical_mask(width: int, n_features: int | None = None) -> np.ndarray:
    """
    Категории это последние колонки БЛОКА ПРИЗНАКОВ, а не матрицы.

    Когда к признакам приписаны эмбеддинги, конец матрицы это уже
    размерности вектора клиента: пометить их категориальными значит
    отдать boosting'у 128 непрерывных величин как метки классов.
    """

    n_features = width if n_features is None else n_features

    mask = np.zeros(width, dtype=bool)
    mask[n_features - N_CATEGORICAL : n_features] = True
    return mask


def scores_of(model, features: np.ndarray) -> np.ndarray:
    return model.predict_proba(features)[:, 1]


def metrics_of(labels: np.ndarray, scores: np.ndarray) -> dict:

    from sklearn.metrics import average_precision_score, roc_auc_score

    positives = int(labels.sum())

    if positives == 0 or positives == labels.size:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "positive_rate": _share(positives, labels.size),
            "n": int(labels.size),
        }

    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "positive_rate": _share(positives, labels.size),
        "n": int(labels.size),
    }


def bootstrap_interval(
    labels: np.ndarray, scores: np.ndarray, draws: int, seed: int
) -> dict:
    """
    Интервал по клиентам: единица пересэмплирования это клиент,
    и у downstream клиент это ровно один пример.
    """

    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(seed)

    roc: list[float] = []
    pr: list[float] = []

    for _ in range(draws):

        picked = rng.integers(0, labels.size, labels.size)

        drawn = labels[picked]

        if drawn.sum() == 0 or drawn.sum() == drawn.size:
            continue

        roc.append(float(roc_auc_score(drawn, scores[picked])))
        pr.append(float(average_precision_score(drawn, scores[picked])))

    if not roc:
        return {"roc_auc": None, "pr_auc": None, "draws": 0}

    return {
        "roc_auc": [
            round(float(np.percentile(roc, 2.5)), 4),
            round(float(np.percentile(roc, 97.5)), 4),
        ],
        "pr_auc": [
            round(float(np.percentile(pr, 2.5)), 4),
            round(float(np.percentile(pr, 97.5)), 4),
        ],
        "draws": len(roc),
    }


def evaluate_model(model, data: dict, labels: dict, draws: int, seed: int) -> dict:

    out: dict = {}

    for group in ("val", "test"):

        scores = scores_of(model, data[group])

        out[group] = {
            **metrics_of(labels[group], scores),
            "interval": bootstrap_interval(labels[group], scores, draws, seed),
        }

    out["train"] = metrics_of(labels["train"], scores_of(model, data["train"]))

    return out


# ============================================================
# ПРОГОН
# ============================================================


def run_downstream(
    env,
    processed: Path,
    raw: Path,
    checkpoint: Path,
    out_dir: Path,
    device: str = "cpu",
    batch_size: int = 8,
    seed: int = 42,
    draws: int = 1000,
) -> dict:

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    started = time.perf_counter()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = build_examples(processed, raw)

    labels = {
        group: np.array(
            [examples.labels[int(row["client_id"])] for row in rows], dtype=np.int64
        )
        for group, rows in examples.rows.items()
    }

    # --- признаки ------------------------------------------
    raw_features = {
        group: build_features(processed, group, rows, examples.cutoff)
        for group, rows in examples.rows.items()
    }

    encoder = fit_encoder(raw_features["train"])

    features = {group: encode(value, encoder) for group, value in raw_features.items()}

    # --- эмбеддинги ----------------------------------------
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)

    stored = TrainConfig.from_dict(payload["train_config"])

    trainer = Trainer(stored, env.tokenizer, env.table, env.unigram, device)

    load_checkpoint(
        Path(checkpoint),
        backbone=trainer.backbone,
        head=trainer.head,
        model_config=trainer.model_config.as_dict(),
        artifacts=env.hashes,
        restore_random=False,
    )

    vectors = {
        group: build_embeddings(env, trainer, env.root, group, rows, batch_size)
        for group, rows in examples.rows.items()
    }

    scaler = StandardScaler().fit(vectors["train"])

    scaled = {group: scaler.transform(value) for group, value in vectors.items()}

    both = {
        group: np.concatenate([features[group], vectors[group]], axis=1)
        for group in GROUPS
    }

    # --- три модели ----------------------------------------
    width = features["train"].shape[1]

    boosting = HistGradientBoostingClassifier(
        **BOOSTING, random_state=seed, categorical_features=categorical_mask(width)
    ).fit(features["train"], labels["train"])

    regression = LogisticRegression(max_iter=2000, random_state=seed).fit(
        scaled["train"], labels["train"]
    )

    combined = HistGradientBoostingClassifier(
        **BOOSTING,
        random_state=seed,
        categorical_features=categorical_mask(both["train"].shape[1], width),
    ).fit(both["train"], labels["train"])

    report = {
        "label": LABEL,
        "cutoff": str(examples.cutoff),
        "checkpoint": str(checkpoint),
        "seed": seed,
        "bootstrap_draws": draws,
        "examples": examples.summary(),
        "skipped": dict(examples.skipped),
        "features": {
            "n": len(FEATURE_NAMES),
            "categorical": list(PROFILE_CATEGORICAL),
            "names": list(FEATURE_NAMES),
            "fit_on": "train",
        },
        "embedding": {
            "d_model": int(trainer.model_config.d_model),
            "structure": trainer.model_config.structure,
            "max_events_per_history": stored.max_events_per_history,
        },
        "models": {
            "features_boosting": evaluate_model(boosting, features, labels, draws, seed),
            "embedding_logistic": evaluate_model(regression, scaled, labels, draws, seed),
            "both_boosting": evaluate_model(combined, both, labels, draws, seed),
        },
        "seconds": round(time.perf_counter() - started, 1),
        "caveat": (
            "downstream мерит переносимость вектора клиента на одной задаче и "
            "одном срезе; это не общая полезность представления"
        ),
    }

    write_json(out_dir / "downstream.json", report)
    write_text(out_dir / "downstream.md", render_downstream(report))

    np.save(out_dir / "embeddings_test.npy", vectors["test"])

    return report


# ============================================================
# ОТЧЁТ
# ============================================================


MODEL_TITLES = {
    "features_boosting": "признаки -> boosting",
    "embedding_logistic": "[USR] -> логистическая регрессия",
    "both_boosting": "признаки и [USR] -> boosting",
}


def _text(value, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _interval(values) -> str:
    return "—" if not values else f"[{values[0]:.4f}, {values[1]:.4f}]"


def render_downstream(report: dict) -> str:

    lines: list[str] = []

    lines.append(f"# Downstream: {report['label']}")
    lines.append("")
    lines.append(
        f"Срез признаков {report['cutoff']}, один пример на клиента. "
        f"Метка из RAW, в признаки не входит."
    )
    lines.append("")

    lines.append("## Клиенты")
    lines.append("")
    lines.append("| группа | клиентов | положительных | доля |")
    lines.append("|---|---:|---:|---:|")

    for group, item in report["examples"]["groups"].items():
        lines.append(
            f"| {group} | {item['clients']} | {item['positives']} | {item['positive_rate']} |"
        )

    lines.append("")

    lines.append(
        f"Признаков {report['features']['n']}, все преобразования обучены "
        f"на {report['features']['fit_on']}. Вектор клиента d={report['embedding']['d_model']}, "
        f"структура {report['embedding']['structure']}."
    )
    lines.append("")

    for split in ("val", "test"):

        lines.append(f"## {split}")
        lines.append("")
        lines.append("| модель | ROC-AUC | интервал | PR-AUC | интервал |")
        lines.append("|---|---:|---|---:|---|")

        for name, title in MODEL_TITLES.items():

            item = report["models"][name][split]
            interval = item["interval"]

            lines.append(
                f"| {title} | {_text(item['roc_auc'])} | {_interval(interval['roc_auc'])} | "
                f"{_text(item['pr_auc'])} | {_interval(interval['pr_auc'])} |"
            )

        lines.append("")
        lines.append(
            f"Доля положительных {report['examples']['groups'][split_group(split)]['positive_rate']}: "
            "это опора для PR-AUC, ниже неё модель бесполезна."
        )
        lines.append("")

    lines.append("## Чего этот отчёт не утверждает")
    lines.append("")
    lines.append(report["caveat"] + ".")
    lines.append("")

    return "\n".join(lines)


def split_group(split: str) -> str:
    return {"val": "val", "test": "test", "train": "train"}[split]


# ============================================================
# CLI
# ============================================================


def main() -> None:

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Downstream: product_open_90d")

    parser.add_argument("--name", default="dev")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processed", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--vocab", type=Path, default=None)
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)

    args = parser.parse_args()

    env = load_environment(
        args.root or tokenized_dir(args.name),
        args.vocab or vocab_dir(args.name),
        args.artifacts or artifacts_dir(args.name),
    )

    out_dir = args.out or RUNS_DIR / args.name / "downstream"

    report = run_downstream(
        env=env,
        processed=args.processed or prep_processed_dir(args.name),
        raw=args.raw,
        checkpoint=args.checkpoint,
        out_dir=out_dir,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        draws=args.bootstrap,
    )

    for name, title in MODEL_TITLES.items():
        item = report["models"][name]["test"]
        print(f"{title:<40s} ROC-AUC {_text(item['roc_auc'])}  PR-AUC {_text(item['pr_auc'])}")

    print(f"записано: {out_dir}")


if __name__ == "__main__":
    main()


__all__ = [
    "FEATURE_NAMES",
    "build_examples",
    "build_features",
    "render_downstream",
    "run_downstream",
]
