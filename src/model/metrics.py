from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from src.tokenizer.vocab import Vocab

from .mlm_head import FieldLogits, FieldTable


# ============================================================
# ИДЕЯ
# ============================================================
#
# Loss падает у любой модели, которая выучила частоты. Чтобы
# отличить представление от запомненного распределения, каждое
# поле сравнивается с unigram-baseline preprocessing НА ТЕХ ЖЕ
# позициях:
#
#   CE_unigram = среднее −ln p_train(target)
#   NCE gain   = (CE_unigram − CE_model) / CE_unigram
#
# Baseline берётся только из artifacts и не пересчитывается по
# оцениваемым данным: иначе он подглядывал бы в val.
#
# Метрики копятся по всему набору и агрегируются один раз.
# Среднее готовых batch-F1 или batch-NCE было бы другой
# величиной: у batch'ей разное число целей, а F1 нелинейна.
# ============================================================


EPSILON = 1e-8

STATUS_OK = "ok"
STATUS_NO_TARGETS = "no_targets"
# Целей нет по решению, а не по бедности данных: политика
# целей вывела поле из задачи. Без отдельного статуса это
# читалось бы как дефект набора.
STATUS_EXCLUDED = "excluded"
STATUS_DEGENERATE = "degenerate"
STATUS_NO_UNIGRAM = "no_unigram"
STATUS_UNDEFINED = "undefined"


# ============================================================
# UNIGRAM
# ============================================================


class UnigramTable:
    """
    Распределение train-частот каждого поля в локальных индексах.

    Исходный artifact не меняется: из него только читают.
    """

    def __init__(
        self,
        artifact: dict,
        vocab: Vocab,
        table: FieldTable,
        epsilon: float = EPSILON,
    ):

        self.epsilon = float(epsilon)

        self.log_probs: dict[int, np.ndarray] = {}
        self.mode: dict[int, int] = {}

        self.unmatched: dict[str, list[str]] = {}
        self.coverage: dict[str, float] = {}
        self.missing: list[str] = []

        fields = artifact.get("fields", {})

        for key_id in table.trainable_key_ids:

            entry = vocab.key_entry_by_id(key_id)

            item = fields.get(entry.namespace, {}).get(entry.field)

            if item is None or not item.get("distribution"):
                self.missing.append(entry.key)
                continue

            size = table.size_of(key_id)

            weights = np.zeros(size, dtype=np.float64)

            unmatched: list[str] = []

            for value, share in item["distribution"]:

                local = self._local(item.get("encoding"), value, key_id, size, vocab, table)

                if local is None:
                    unmatched.append(str(value))
                    continue

                weights[local] += float(share)

            if unmatched:
                self.unmatched[entry.key] = unmatched

            self.coverage[entry.key] = float(weights.sum())

            if weights.sum() <= 0.0:
                self.missing.append(entry.key)
                continue

            # Пол снизу и нормировка по кандидатам поля: масса
            # несопоставленных значений не должна утекать.
            weights = np.clip(weights, self.epsilon, None)
            weights = weights / weights.sum()

            self.log_probs[key_id] = np.log(weights)
            self.mode[key_id] = int(np.argmax(weights))

    # --------------------------------------------------------

    @staticmethod
    def _local(encoding, value, key_id: int, size: int, vocab: Vocab, table: FieldTable) -> int | None:
        """
        Значение artifact в локальный индекс кандидата.
        """

        if encoding == "bucket":

            index = int(value)

            return index if 0 <= index < size else None

        found = vocab.value_id(key_id, str(value))

        if found is None:
            return None

        local = int(found) - int(table.value_start[key_id])

        return local if 0 <= local < size else None

    # --------------------------------------------------------

    def has(self, key_id: int) -> bool:
        return int(key_id) in self.log_probs

    def as_dict(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "fields": len(self.log_probs),
            "missing": sorted(self.missing),
            "unmatched": {name: sorted(values) for name, values in sorted(self.unmatched.items())},
            "min_coverage": min(self.coverage.values()) if self.coverage else None,
            "rule": (
                "распределение берётся из unigram_baselines.json preprocessing и не "
                "пересчитывается по оцениваемым данным; вероятность ограничена снизу epsilon "
                "и нормирована по кандидатам поля"
            ),
        }


# ============================================================
# НАКОПИТЕЛЬ
# ============================================================


def aggregate_fields(fields: list[dict], exclude: frozenset[str] = frozenset()) -> dict:
    """
    Агрегаты по набору полей, возможно по подмножеству.

    Считается одним кодом и для полного отчёта, и для среза без
    event_type и дублирующих профиль полей: иначе две цифры в
    отчёте могли бы расходиться по причине, не связанной с
    данными.
    """

    kept = [item for item in fields if item["field"] not in exclude]

    with_targets = [item for item in kept if item["n_targets"] > 0]

    known = [item for item in with_targets if item["ce_unigram"] is not None]

    gains = [item["nce_gain"] for item in with_targets if item["status"] == STATUS_OK]

    def weighted(name: str, source: list[dict]) -> float | None:
        if not source:
            return None
        total = sum(item["n_targets"] for item in source)
        return sum(item[name] * item["n_targets"] for item in source) / total

    def balanced(name: str, source: list[dict]) -> float | None:
        return sum(item[name] for item in source) / len(source) if source else None

    return {
        "n_targets": sum(item["n_targets"] for item in with_targets),
        "n_fields_with_targets": len(with_targets),
        "n_fields_considered": len(kept),
        "n_fields_excluded": len(fields) - len(kept),
        "excluded": sorted(item["field"] for item in fields if item["field"] in exclude),
        "field_balanced_ce": balanced("ce_model", with_targets),
        "token_weighted_ce": weighted("ce_model", with_targets),
        "field_balanced_ce_unigram": balanced("ce_unigram", known),
        "token_weighted_ce_unigram": weighted("ce_unigram", known),
        "mean_nce_gain": (sum(gains) / len(gains)) if gains else None,
        "accuracy": weighted("accuracy", with_targets),
        "unigram_accuracy": weighted("unigram_accuracy", known),
        "macro_f1_mean": balanced("macro_f1", with_targets),
    }


@dataclass
class _FieldState:
    nll_model: float = 0.0
    nll_unigram: float = 0.0
    correct: int = 0
    unigram_correct: int = 0
    topk_correct: int = 0
    n: int = 0
    true: list[np.ndarray] = field(default_factory=list)
    pred: list[np.ndarray] = field(default_factory=list)


class MetricAccumulator:
    """
    Суммы по полям за весь набор, агрегация один раз в конце.
    """

    def __init__(
        self,
        table: FieldTable,
        unigram: UnigramTable,
        top_k: int = 5,
        epsilon: float = EPSILON,
        keep_units: bool = False,
    ):

        self.table = table
        self.unigram = unigram
        self.top_k = int(top_k)
        self.epsilon = float(epsilon)

        self.state: dict[int, _FieldState] = {}

        self.n_degenerate = 0
        self.n_masked = 0

        # Разбивка по примерам нужна только парному bootstrap:
        # хранить её всегда значило бы платить памятью за то,
        # что обычному отчёту не нужно.
        self.keep_units = bool(keep_units)

        self.units: dict[tuple[int, int], list[float]] = {}

    # --------------------------------------------------------

    def note_batch(self, n_masked: int, n_degenerate: int) -> None:
        """
        Сколько позиций скрыл masker и сколько из них отброшено
        как вырожденные. Это не то же, что число целей.
        """

        self.n_masked += int(n_masked)
        self.n_degenerate += int(n_degenerate)

    def _note_units(self, key_id: int, units: torch.Tensor, nll: torch.Tensor) -> None:
        """
        Суммы по примерам, сгруппированные векторно.

        Цикл по каждой цели здесь стоил бы миллионы итераций
        Python: на полных историях у одного набора несколько
        миллионов целей.
        """

        owners = units.detach().to("cpu").numpy().astype(np.int64)

        losses = nll.numpy().astype(np.float64)

        keys, index = np.unique(owners, return_inverse=True)

        index = np.asarray(index).ravel()

        sums = np.bincount(index, weights=losses, minlength=keys.size)
        counts = np.bincount(index, minlength=keys.size)

        for owner, total, count in zip(keys.tolist(), sums.tolist(), counts.tolist()):
            cell = self.units.setdefault((int(owner), int(key_id)), [0.0, 0.0])
            cell[0] += total
            cell[1] += float(count)

    def update(
        self,
        field_logits: list[FieldLogits],
        local_targets: torch.Tensor,
        units: torch.Tensor | None = None,
    ) -> None:

        for item in field_logits:

            targets = local_targets[item.index].detach().to("cpu")

            if targets.numel() == 0:
                continue

            logits = item.logits.detach().float().to("cpu")

            cell = self.state.setdefault(item.key_id, _FieldState())

            log_probs = torch.log_softmax(logits, dim=-1)

            nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

            prediction = logits.argmax(dim=-1)

            cell.nll_model += float(nll.sum())
            cell.correct += int((prediction == targets).sum())
            cell.n += int(targets.numel())

            if self.keep_units:

                if units is None:
                    raise ValueError("keep_units требует номеров примеров для каждой позиции")

                self._note_units(item.key_id, units[item.index], nll)

            width = min(self.top_k, item.n_candidates)

            best = logits.topk(width, dim=-1).indices

            cell.topk_correct += int((best == targets.unsqueeze(1)).any(dim=1).sum())

            true = targets.numpy().astype(np.int64)

            cell.true.append(true)
            cell.pred.append(prediction.numpy().astype(np.int64))

            if self.unigram.has(item.key_id):

                log_p = self.unigram.log_probs[item.key_id]

                cell.nll_unigram += float(-log_p[true].sum())
                cell.unigram_correct += int((true == self.unigram.mode[item.key_id]).sum())

    # --------------------------------------------------------

    @staticmethod
    def _macro_f1(true: np.ndarray, pred: np.ndarray, size: int) -> float:
        """
        Macro F1 по ПОЛНОМУ набору кандидатов поля.

        Класс без предсказаний и без истинных примеров даёт 0,
        как zero_division=0: иначе редкие значения исчезали бы
        из знаменателя и метрика росла бы от сужения набора.
        """

        hit = true == pred

        tp = np.bincount(true[hit], minlength=size).astype(np.float64)

        predicted = np.bincount(pred, minlength=size).astype(np.float64)
        actual = np.bincount(true, minlength=size).astype(np.float64)

        precision = np.divide(tp, predicted, out=np.zeros(size), where=predicted > 0)
        recall = np.divide(tp, actual, out=np.zeros(size), where=actual > 0)

        total = precision + recall

        f1 = np.divide(2 * precision * recall, total, out=np.zeros(size), where=total > 0)

        return float(f1.mean())

    def _field_report(self, key_id: int, excluded: frozenset[str] = frozenset()) -> dict:

        size = self.table.size_of(key_id)

        base = {
            "field": self.table.name(key_id),
            "key_id": int(key_id),
            "kind": self.table.kind(key_id),
            "n_candidates": size,
        }

        cell = self.state.get(key_id)

        if cell is None or cell.n == 0:

            if self.table.degenerate[key_id]:
                status = STATUS_DEGENERATE
            elif base["field"] in excluded:
                status = STATUS_EXCLUDED
            else:
                status = STATUS_NO_TARGETS

            return {
                **base,
                "status": status,
                "n_targets": 0,
                "n_classes_with_support": 0,
                "ce_model": None,
                "ce_unigram": None,
                "nce_gain": None,
                "accuracy": None,
                "unigram_accuracy": None,
                "macro_f1": None,
                "top_k_accuracy": None,
            }

        true = np.concatenate(cell.true)
        pred = np.concatenate(cell.pred)

        ce_model = cell.nll_model / cell.n

        known = self.unigram.has(key_id)

        ce_unigram = cell.nll_unigram / cell.n if known else None

        if not known:
            status, gain = STATUS_NO_UNIGRAM, None
        elif ce_unigram < self.epsilon:
            status, gain = STATUS_UNDEFINED, None
        else:
            status, gain = STATUS_OK, (ce_unigram - ce_model) / ce_unigram

        return {
            **base,
            "status": status,
            "n_targets": cell.n,
            "n_classes_with_support": int(np.unique(true).size),
            "ce_model": ce_model,
            "ce_unigram": ce_unigram,
            "nce_gain": gain,
            "accuracy": cell.correct / cell.n,
            "unigram_accuracy": (cell.unigram_correct / cell.n) if known else None,
            "macro_f1": self._macro_f1(true, pred, size),
            # Top-K осмысленна только когда кандидатов больше K.
            "top_k_accuracy": (cell.topk_correct / cell.n) if size > self.top_k else None,
        }

    # --------------------------------------------------------

    def finalize(self, exclude: frozenset[str] = frozenset()) -> dict:

        fields = [
            self._field_report(key_id, exclude) for key_id in self.table.trainable_key_ids
        ]
        fields += [
            self._field_report(key_id, exclude) for key_id in self.table.degenerate_key_ids
        ]

        excluded_names = sorted(
            item["field"] for item in fields if item["status"] == STATUS_EXCLUDED
        )

        return {
            "top_k": self.top_k,
            "excluded_fields": excluded_names,
            "epsilon": self.epsilon,
            "n_masked_positions": self.n_masked,
            "n_degenerate_skipped": self.n_degenerate,
            "n_fields_trainable": len(self.table.trainable_key_ids),
            **aggregate_fields(fields),
            "subset": aggregate_fields(fields, exclude) if exclude else None,
            "fields": fields,
        }

    # --------------------------------------------------------

    def unit_losses(
        self,
        exclude: frozenset[str] = frozenset(),
        universe: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        По каждому примеру и полю: сумма NLL и число целей.

        Возвращает номера примеров, матрицу сумм и матрицу
        счётчиков формы [примеры, поля]. Поля берутся общие для
        всего набора, чтобы bootstrap считал field-balanced CE
        так же, как finalize.

        universe задаёт полный список единиц. Без него в строки
        попадают только примеры, у которых нашлась хотя бы одна
        цель, и пример без целей не смог бы быть выбранным в
        bootstrap. Это занижало бы разброс.
        """

        if not self.keep_units:
            raise ValueError("накопитель собран без keep_units")

        if universe is not None:
            owners = [int(value) for value in np.asarray(universe, dtype=np.int64)]
        else:
            owners = sorted({owner for owner, _ in self.units})

        keys = [
            key_id
            for key_id in self.table.trainable_key_ids
            if key_id in self.state and self.table.name(key_id) not in exclude
        ]

        row = {owner: index for index, owner in enumerate(owners)}
        column = {key_id: index for index, key_id in enumerate(keys)}

        total = np.zeros((len(owners), len(keys)), dtype=np.float64)
        counts = np.zeros((len(owners), len(keys)), dtype=np.float64)

        for (owner, key_id), (value, count) in self.units.items():

            if key_id not in column or owner not in row:
                continue

            total[row[owner], column[key_id]] += value
            counts[row[owner], column[key_id]] += count

        return np.asarray(owners, dtype=np.int64), total, counts

    def clients_with_targets(self, clusters: np.ndarray) -> dict[int, int]:
        """
        Сколько РАЗНЫХ клиентов дало цели каждому полю.

        У редкого поля может быть тридцать целей и один клиент;
        число целей об этом не говорит, а интервал зависит
        именно от числа клиентов.
        """

        clusters = np.asarray(clusters, dtype=np.int64)

        seen: dict[int, set[int]] = {}

        for (owner, key_id), (_, count) in self.units.items():

            if count <= 0 or owner >= clusters.size:
                continue

            seen.setdefault(int(key_id), set()).add(int(clusters[owner]))

        return {key_id: len(owners) for key_id, owners in seen.items()}


# ============================================================
# ПАРНЫЙ BOOTSTRAP
# ============================================================


def _balanced_ce(total: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    Field-balanced CE построчно: среднее по полям с целями.
    """

    with np.errstate(invalid="ignore", divide="ignore"):
        by_field = np.where(counts > 0, total / np.maximum(counts, 1e-12), np.nan)

    return np.nanmean(by_field, axis=-1)


def cluster_draws(n_units: int, n_boot: int = 2000, seed: int = 20240608) -> np.ndarray:
    """
    Кратности единиц в bootstrap-выборках: [n_boot, n_units].

    Одна матрица на набор, а не свой розыгрыш в каждом вызове:
    сравниваются модели и правила на ОДНОЙ выборке клиентов, и
    полагаться на то, что seed и размер случайно совпали, нельзя.
    """

    if n_units < 1:
        raise ValueError("нечего пересэмплировать: единиц нет")

    rng = np.random.default_rng(seed)

    return rng.multinomial(
        n_units, np.full(n_units, 1.0 / n_units), size=int(n_boot)
    ).astype(np.float64)


def group_sample(
    sample: tuple[np.ndarray, np.ndarray, np.ndarray], clusters: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Складывает строки одной единицы кластеризации.

    Field-balanced CE суммирует NLL и counts по полю, а потом
    усредняет по полям. Сумма столбца не зависит от того,
    сгруппированы строки или нет, поэтому «взять клиента с
    кратностью k вместе со всеми его примерами» это то же, что
    «взять строку клиента с весом k». Формула сохраняется, а не
    подменяется средним по клиентским CE.
    """

    owners, total, counts = sample

    clusters = np.asarray(clusters, dtype=np.int64)

    if clusters.shape[0] != owners.shape[0]:
        raise ValueError(
            f"кластеров {clusters.shape[0]}, а строк {owners.shape[0]}"
        )

    keys, index = np.unique(clusters, return_inverse=True)

    index = np.asarray(index).ravel()

    grouped_total = np.zeros((keys.size, total.shape[1]), dtype=np.float64)
    grouped_counts = np.zeros((keys.size, counts.shape[1]), dtype=np.float64)

    np.add.at(grouped_total, index, total)
    np.add.at(grouped_counts, index, counts)

    return keys, grouped_total, grouped_counts


def bootstrap_combination(
    terms: list[tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    n_boot: int = 2000,
    seed: int = 20240608,
    alpha: float = 0.05,
    draws: np.ndarray | None = None,
) -> dict:
    """
    Линейная комбинация field-balanced CE нескольких оценок.

    Пересэмплируются ПРИМЕРЫ, а не позиции: позиции одного
    клиента зависимы, и bootstrap по ним дал бы интервал уже
    настоящего. Все слагаемые пересэмплируются одними и теми же
    примерами, поэтому разность двух разностей считается на
    общей выборке, а не на двух независимых.
    """

    if not terms:
        raise ValueError("нечего комбинировать")

    owners = terms[0][1][0]
    shape = terms[0][1][1].shape

    for _, (other, total, _) in terms:

        if not np.array_equal(other, owners):
            raise ValueError("bootstrap требует одних и тех же примеров во всех слагаемых")

        if total.shape != shape:
            raise ValueError(f"наборы полей различаются: {total.shape} против {shape}")

    units = int(owners.size)

    if units == 0:
        raise ValueError("нечего пересэмплировать: примеров нет")

    estimate = float(
        sum(
            weight * _balanced_ce(total.sum(axis=0), counts.sum(axis=0))
            for weight, (_, total, counts) in terms
        )
    )

    # Кратности единиц вместо явной выборки индексов: то же
    # распределение, но одним умножением матриц.
    draw = cluster_draws(units, n_boot, seed) if draws is None else np.asarray(draws, dtype=np.float64)

    if draw.shape[1] != units:
        raise ValueError(f"матрица кратностей на {draw.shape[1]} единиц, а их {units}")

    values = sum(
        weight * _balanced_ce(draw @ total, draw @ counts) for weight, (_, total, counts) in terms
    )

    low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "share_below_zero": float((values < 0).mean()),
        "n_boot": int(draw.shape[0]),
        "n_units": units,
        "alpha": alpha,
        "significant": bool(low > 0 or high < 0),
    }


def bootstrap_contrast(
    left: tuple[np.ndarray, np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_boot: int = 2000,
    seed: int = 20240608,
    alpha: float = 0.05,
    draws: np.ndarray | None = None,
) -> dict:
    """
    Разница field-balanced CE (right − left) с интервалом.
    """

    return bootstrap_combination(
        [(-1.0, left), (1.0, right)], n_boot=n_boot, seed=seed, alpha=alpha, draws=draws
    )


# ============================================================
# ОТЧЁТ
# ============================================================


def _cell(value, digits: int = 3, width: int = 9) -> str:

    if value is None:
        return f"{'—':>{width}s}"

    return f"{value:>{width}.{digits}f}"


def render_metrics(report: dict, title: str = "МЕТРИКИ") -> str:

    lines: list[str] = []

    lines.append("=" * 108)
    lines.append(title)
    lines.append("=" * 108)

    header = (
        f"  {'поле':<38s}{'целей':>8s}{'канд':>6s}{'класс':>7s}"
        f"{'CE':>9s}{'CE uni':>9s}{'NCE':>9s}{'Acc':>9s}{'Acc uni':>9s}{'MacroF1':>9s}{'Top-K':>9s}"
    )

    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for item in report["fields"]:

        if item["n_targets"] == 0:
            lines.append(f"  {item['field']:<38s}{0:>8}{item['n_candidates']:>6}   {item['status']}")
            continue

        lines.append(
            f"  {item['field']:<38s}{item['n_targets']:>8,}{item['n_candidates']:>6}"
            f"{item['n_classes_with_support']:>7}"
            f"{_cell(item['ce_model'])}{_cell(item['ce_unigram'])}{_cell(item['nce_gain'])}"
            f"{_cell(item['accuracy'])}{_cell(item['unigram_accuracy'])}"
            f"{_cell(item['macro_f1'])}{_cell(item['top_k_accuracy'])}".replace(",", " ")
        )

    lines.append("")

    def row(label: str, value, digits: int = 4) -> None:
        text = "—" if value is None else f"{value:.{digits}f}"
        lines.append(f"  {label:<38s}{text:>12s}")

    lines.append(f"  целей всего: {report['n_targets']:,}".replace(",", " "))
    lines.append(
        f"  полей с целями: {report['n_fields_with_targets']} из {report['n_fields_trainable']}; "
        f"пропущено вырожденных позиций: {report['n_degenerate_skipped']:,}".replace(",", " ")
    )

    excluded = report.get("excluded_fields") or []

    if excluded:
        lines.append(
            f"  выведено из задачи политикой целей: {', '.join(excluded)}"
        )

    lines.append("")

    row("field-balanced CE модели", report["field_balanced_ce"])
    row("field-balanced CE unigram", report["field_balanced_ce_unigram"])
    row("token-weighted CE модели", report["token_weighted_ce"])
    row("token-weighted CE unigram", report["token_weighted_ce_unigram"])
    row("средний NCE gain по полям", report["mean_nce_gain"])
    row("accuracy модели", report["accuracy"])
    row("accuracy unigram", report["unigram_accuracy"])
    row("средний macro F1", report["macro_f1_mean"])

    return "\n".join(lines)
