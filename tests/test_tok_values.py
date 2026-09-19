from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.preprocessing.artifacts import read_json, sha256_file
from src.tokenization.categorical import CATALOG_FILE, ValuesError, build_values, sort_key, typed_value
from src.tokenization.contract import CATEGORICAL_FILE, STATISTICS_DIR, build_contract
from src.tokenization.numeric import (
    FOUND_BUCKET,
    FOUND_INVALID,
    FOUND_UNKNOWN,
    REGISTRY_FILE,
    SOURCE_CONFIG,
    SOURCE_FALLBACK,
    SOURCE_TRAIN,
    BucketsError,
    FittedEncoder,
    build_buckets,
    build_buckets_list,
    quantile_boundaries,
)
from src.tokenization.scan import TYPE_BOOL, TYPE_INT, TYPE_STR
from src.tokenization.settings import (
    METHOD_QUANTILE,
    NEGATIVE_ALLOWED,
    NEGATIVE_INVALID,
    ZERO_IN_RANGE,
    ZERO_SEPARATE,
    TokenizerConfig,
)

from tests.tok_fixtures import prepared


# ============================================================
# ОБЩЕЕ
# ============================================================


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> tuple[Path, Path]:
    return prepared(tmp_path_factory.mktemp("tok_values"))


@pytest.fixture(scope="module")
def built(dataset, tmp_path_factory) -> tuple[Path, dict, dict]:

    root, out = dataset

    target = tmp_path_factory.mktemp("vocab_values")

    config = TokenizerConfig()

    build_contract(out, root / "train", target, config)
    values = build_values(target, out, config).report
    buckets = build_buckets(target, out, config).report

    return target, values, buckets


def _encoder(key: str, boundaries: tuple[float, ...], unit: str | None = "KZT",
             zero: str = ZERO_SEPARATE, negative: str = NEGATIVE_INVALID) -> FittedEncoder:

    return FittedEncoder(
        key=key,
        unit=unit,
        method=METHOD_QUANTILE,
        boundaries=boundaries,
        zero_policy=zero,
        negative_policy=negative,
        buckets=build_buckets_list(key, unit, boundaries, zero),
    )


# ============================================================
# ЭТАП 2: ЗНАЧЕНИЯ
# ============================================================


def test_declared_domains_merge_and_everything_else_stays_apart(built):
    """
    Объединение это решение с причиной, а не догадка по
    написанию.
    """

    _target, values, _buckets = built

    domains = {item["name"]: item for item in values["domains"]}

    assert domains["event_type_domain"]["keys"] == ["event_type", "related_event_type"]
    assert domains["event_type_domain"]["shared"] is True
    assert domains["event_type_domain"]["reason"]

    # У каналов операции и заявки домены разные: реестр объявил
    # их несовместимыми.
    assert domains["operation_channel"]["keys"] == ["operation_channel"]
    assert domains["application_channel"]["keys"] == ["application_channel"]

    assert values["counts"]["domains_shared"] == 2


def test_shared_domain_really_shares_values(built):
    """
    Тип события-причины берёт значения из того же перечня, что и
    сам тип события.
    """

    _target, values, _buckets = built

    domain = next(item for item in values["domains"] if item["name"] == "event_type_domain")

    texts = {item["value"] for item in domain["values"]}

    assert "purchase" in texts

    rows = {row["key"]: row for row in values["keys"]}

    assert rows["related_event_type"]["domain"] == "event_type_domain"
    assert rows["event_type"]["domain"] == "event_type_domain"


def test_digit_codes_stay_categories(built):
    """
    MCC и версия продукта это коды, а не величины: складывать их
    бессмысленно, и в диапазоны они не режутся.
    """

    _target, values, buckets = built

    rows = {row["key"]: row for row in values["keys"]}

    for key in ("mcc", "product_version", "tariff_version", "profile_salary_day"):
        assert rows[key]["value_kind"] == "categorical"
        assert key not in buckets["encoders"]


def test_value_order_is_typed_not_textual():
    """
    Порядок значений это свойство значения, а не его записи:
    иначе 10 встало бы перед 9.
    """

    assert typed_value(TYPE_INT, "10") == 10
    assert typed_value(TYPE_BOOL, "true") is True

    ordered = sorted(
        [(TYPE_STR, "9"), (TYPE_INT, "10"), (TYPE_INT, "9"), (TYPE_BOOL, "true")],
        key=lambda item: sort_key(*item),
    )

    assert ordered == [(TYPE_BOOL, "true"), (TYPE_INT, "9"), (TYPE_INT, "10"), (TYPE_STR, "9")]


def test_values_refuse_another_configuration(dataset, tmp_path):
    """
    Каталог значений строится по тем же правилам, по которым
    считалась статистика.
    """

    root, out = dataset

    target = tmp_path / "vocab"

    build_contract(out, root / "train", target, TokenizerConfig())

    other = replace(TokenizerConfig(), rare_min_count=5)

    with pytest.raises(ValuesError, match="конфигурация изменилась"):
        build_values(target, out, other)


def test_edited_statistics_are_refused(dataset, tmp_path):
    """
    Правленая статистика не становится входом словаря.
    """

    root, out = dataset

    target = tmp_path / "vocab"

    build_contract(out, root / "train", target, TokenizerConfig())

    path = target / STATISTICS_DIR / CATEGORICAL_FILE

    table = pq.read_table(path)

    counts = table.column("count").to_pylist()

    changed = table.set_column(
        table.column_names.index("count"),
        "count",
        pa.array([value * 2 for value in counts], pa.int64()),
    )

    pq.write_table(changed, path)

    with pytest.raises(ValuesError, match="изменилась после контракта"):
        build_values(target, out, TokenizerConfig())


def test_rare_threshold_is_off_by_default(built):
    """
    Все наблюдавшиеся категории сохраняются целиком, а поиска
    «лучшего порога» здесь нет вовсе.
    """

    _target, values, _buckets = built

    assert values["counts"]["rare"] == 0
    assert "сохраняются целиком" in values["rules"]["rare"]


# ============================================================
# ЭТАП 3: ДИАПАЗОНЫ
# ============================================================


def test_value_on_the_boundary_belongs_to_the_next_range():
    """
    Интервал это [lower, upper). 200000 при последней границе
    200000 уходит в открытый сверху диапазон вместе с 350000 и
    пятью миллионами, а 199999 остаётся в предыдущем.
    """

    encoder = _encoder("transaction_amount", (1_000.0, 200_000.0))

    last = len(encoder.buckets) - 1

    assert encoder.locate(199_999) == (FOUND_BUCKET, last - 1)
    assert encoder.locate(200_000) == (FOUND_BUCKET, last)
    assert encoder.locate(350_000) == (FOUND_BUCKET, last)
    assert encoder.locate(5_000_000) == (FOUND_BUCKET, last)

    assert encoder.buckets[last].label == "transaction_amount[200000,inf)KZT"
    assert encoder.buckets[last].upper is None


def test_zero_negative_invalid_and_unknown_are_four_different_answers():
    """
    Ноль, запрещённый минус, невозможное число и отсутствие
    шкалы кодируются по-разному.
    """

    encoder = _encoder("transaction_amount", (1_000.0,))

    assert encoder.locate(0) == (FOUND_BUCKET, 0)
    assert encoder.buckets[0].zero is True
    assert encoder.buckets[0].label == "transaction_amount=0KZT"

    assert encoder.locate(-5) == (FOUND_INVALID, None)
    assert encoder.locate(float("nan")) == (FOUND_INVALID, None)
    assert encoder.locate(math.inf) == (FOUND_INVALID, None)

    # Минус там, где домен его допускает, это обычное значение.
    allowed = _encoder("balance_after", (1_000.0,), negative=NEGATIVE_ALLOWED)

    assert allowed.locate(-5) == (FOUND_BUCKET, 1)

    nothing = FittedEncoder(
        key="original_amount", unit="original_currency", method="unfitted",
        boundaries=(), zero_policy=ZERO_IN_RANGE, negative_policy=NEGATIVE_INVALID, buckets=(),
    )

    assert nothing.locate(19) == (FOUND_UNKNOWN, None)


def test_quantiles_leave_no_empty_bucket():
    """
    Каждая граница это наблюдавшееся значение, поэтому диапазон,
    который с неё начинается, не бывает пустым.
    """

    values = [float(index % 37) for index in range(500)]

    boundaries = quantile_boundaries(values, 8, "inverted_cdf")

    encoder = _encoder("key", boundaries, unit=None, zero=ZERO_IN_RANGE)

    counts = [0] * len(encoder.buckets)

    for value in values:
        _found, index = encoder.locate(value)
        counts[index] += 1

    assert min(counts) > 0


def test_repeated_values_collapse_boundaries():
    """
    Совпавшие квантили схлопываются: пустых и дублирующих
    интервалов не создаётся.
    """

    # Константа: все квантили равны минимуму, границ не остаётся.
    assert quantile_boundaries([5.0] * 100, 8, "inverted_cdf") == ()

    # Два значения пополам: восемь запрошенных корзин дают одну
    # границу, а не семь совпавших.
    assert quantile_boundaries([1.0] * 50 + [9.0] * 50, 8, "inverted_cdf") == (9.0,)


def test_constant_key_gets_a_single_range():

    encoder = _encoder("key", (), unit=None, zero=ZERO_IN_RANGE)

    assert len(encoder.buckets) == 1
    assert encoder.buckets[0].label == "key(-inf,inf)"
    assert encoder.locate(42) == (FOUND_BUCKET, 0)


def test_scarce_key_falls_back_to_the_declared_scale(built):
    """
    Мало наблюдений — берётся заранее объявленная шкала. По
    validation и test границы не считаются никогда.
    """

    _target, _values, buckets = built

    entry = buckets["encoders"]["approved_amount"]

    assert entry["requested_method"] == METHOD_QUANTILE
    assert entry["boundary_source"] == SOURCE_FALLBACK
    assert entry["boundaries"] == list(TokenizerConfig().numeric_encoders["approved_amount"].fallback)
    assert any("approved_amount" in item for item in buckets["warnings"])


def test_business_scales_do_not_depend_on_the_sample(built):
    """
    Просрочка, срок и ставка заданы бизнесом: их границы от
    выборки не зависят.
    """

    _target, _values, buckets = built

    for key in ("days_past_due", "term", "rate", "profile_age"):
        assert buckets["encoders"][key]["boundary_source"] == SOURCE_CONFIG

    assert buckets["encoders"]["days_past_due"]["zero_policy"] == ZERO_SEPARATE


def test_key_with_own_scale_learns_from_train(dataset, tmp_path):
    """
    Сумма проводки режется по train: её разброс это свойство
    популяции.

    Порог наблюдений здесь опущен намеренно: на маленьком наборе
    он иначе уводит ключ на объявленную шкалу, и квантильный
    путь остался бы непроверенным.
    """

    root, out = dataset

    config = replace(TokenizerConfig(), numeric_min_values=5, numeric_min_clients=1)

    target = tmp_path / "vocab"

    build_contract(out, root / "train", target, config)
    build_values(target, out, config)

    entry = build_buckets(target, out, config).report["encoders"]["transaction_amount"]

    assert entry["boundary_source"] == SOURCE_TRAIN
    assert entry["fit"]["algorithm"] == "inverted_cdf"
    assert entry["actual_bins"] > 2

    # Метка однозначна: ключ, границы и единица.
    assert all(item["label"].startswith("transaction_amount") for item in entry["buckets"])
    assert all(item["label"].endswith("KZT") for item in entry["buckets"])

    # Ноль остаётся отдельным состоянием, а не маленькой суммой.
    assert entry["buckets"][0]["label"] == "transaction_amount=0KZT"


def test_old_and_new_profile_values_share_the_scale_of_their_field(built):
    """
    Прежний и новый доход делят шкалу с самим доходом: иначе
    одно и то же число попало бы в разные корзины.
    """

    _target, _values, buckets = built

    base = buckets["encoders"]["profile_declared_income"]

    for key in ("profile_declared_income_old", "profile_declared_income_new"):
        entry = buckets["encoders"][key]
        assert entry["fit_source"] == "profile_declared_income"
        assert entry["boundaries"] == base["boundaries"]


def test_unfitted_key_stays_without_a_scale(built):
    """
    Единица original_amount лежит в соседнем ключе, поэтому шкалы
    у него нет и значения станут [UNK].
    """

    _target, _values, buckets = built

    entry = buckets["encoders"]["original_amount"]

    assert entry["method"] == "unfitted"
    assert entry["buckets"] == []


def test_buckets_are_reproducible(dataset, tmp_path):

    root, out = dataset

    reports = []

    for name in ("one", "two"):

        target = tmp_path / name

        build_contract(out, root / "train", target, TokenizerConfig())
        build_values(target, out, TokenizerConfig())
        reports.append(build_buckets(target, out, TokenizerConfig()).report)

    assert reports[0] == reports[1]
    assert sha256_file(tmp_path / "one" / REGISTRY_FILE) == sha256_file(tmp_path / "two" / REGISTRY_FILE)


def test_buckets_need_the_value_catalogue(dataset, tmp_path):

    root, out = dataset

    target = tmp_path / "vocab"

    build_contract(out, root / "train", target, TokenizerConfig())

    with pytest.raises(BucketsError, match="выполните values"):
        build_buckets(target, out, TokenizerConfig())


def test_catalogue_records_what_each_event_type_declares(built):
    """
    Карта «тип события -> объявленные ключи» это часть словаря:
    по ней этап 5 решает, где поставить [MISSING].
    """

    target, values, _buckets = built

    declared = values["declared_by_event_type"]

    assert "decline_reason" in declared["purchase"]
    assert "event_type" in declared["purchase"]

    # Расчётного ключа среди объявленных полей события нет.
    assert "amount_to_limit" not in declared["purchase"]

    # Каталог на диске тот же, что в памяти.
    assert read_json(target / CATALOG_FILE)["declared_by_event_type"] == declared
