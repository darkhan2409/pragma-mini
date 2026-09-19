from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.preprocessing.artifacts import read_json, sha256_file, write_json
from src.preprocessing.semantic.build import REGISTRY_FILE as SEMANTIC_REGISTRY_FILE
from src.tokenization.contract import (
    CATEGORICAL_FILE,
    COMPATIBILITY_REPORT_FILE,
    CONFIG_FILE,
    CONTRACT_REPORT_FILE,
    FIT_MANIFEST_FILE,
    MISSING_FILE,
    NUMERIC_SUMMARY_FILE,
    STATISTICS_DIR,
    TEXT_FILE,
    ContractError,
    build_contract,
)
from src.tokenization.corpus import CorpusError, FitCorpus
from src.tokenization.scan import TYPE_BOOL, TYPE_INT, TYPE_STR, FitStatistics, TextEntry, value_text, value_type
from src.tokenization.schema import WEIGHT_PER_CLIENT, WEIGHT_PER_EVENT, SemanticSchema
from src.tokenization.settings import ConfigError, TokenizerConfig, ValueDomain

from tests.tok_fixtures import FIT_END, prepared


# ============================================================
# ОБЩЕЕ
# ============================================================


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> tuple[Path, Path]:
    return prepared(tmp_path_factory.mktemp("tok_base"))


@pytest.fixture(scope="module")
def built(dataset, tmp_path_factory) -> tuple[dict, Path]:

    root, out = dataset

    target = tmp_path_factory.mktemp("tok_vocab")

    result = build_contract(out, root / "train", target, TokenizerConfig())

    return result.report, target


def _contract(root: Path, out: Path, target: Path, config: TokenizerConfig | None = None, **kwargs):
    return build_contract(out, root / "train", target, config or TokenizerConfig(), **kwargs)


# ============================================================
# РАЗРЕШЁННЫЙ КОРПУС
# ============================================================


def test_contract_reads_only_the_allowed_train_corpus(built):
    """
    В fit идёт train на fit_end и ничего больше.
    """

    report, target = built

    assert report["group"] == "train"
    assert report["fit_end"] == FIT_END.isoformat()

    corpus = report["corpus"]
    declared = report["declared_by_split"]

    # Событий прочитано ровно столько, сколько объявило
    # разделение: второй реализации видимости нет.
    assert corpus["events"] == declared["events_rows"]

    # Молчащий клиент остаётся в группе: молчание это факт.
    assert corpus["clients"] == 2
    assert corpus["clients_without_profile"] == declared["clients_without_profile"] == 1

    # Версия профиля считается по клиентам, а не по событиям, и
    # это ДВА разных числа: у клиента с анкетой две её версии
    # известны к fit_end, а признаком становится одна
    # действующая. Проверять только второе — значит не проверять
    # ничего: при совпадении чисел разница была бы не видна.
    assert corpus["profiles_as_of"] == 1
    assert declared["source_profile_version_rows"] == 2

    for name in (CONFIG_FILE, FIT_MANIFEST_FILE, CONTRACT_REPORT_FILE, COMPATIBILITY_REPORT_FILE):
        assert (target / name).exists()

    for name in (CATEGORICAL_FILE, TEXT_FILE, MISSING_FILE, NUMERIC_SUMMARY_FILE):
        assert (target / STATISTICS_DIR / name).exists()


def test_horizon_and_readiness_are_two_different_verdicts(tmp_path):
    """
    Технически исправное разделение с коротким горизонтом
    открывается только по явному разрешению, и результат
    помечается диагностикой.
    """

    root, out = prepared(tmp_path, history_start=datetime(2025, 1, 1))

    with pytest.raises(CorpusError, match="горизонт"):
        FitCorpus.open(out, root / "train")

    corpus = FitCorpus.open(out, root / "train", allow_short_horizon=True)

    assert corpus.readiness.status == "diagnostic"
    assert corpus.readiness.reasons

    result = _contract(root, out, tmp_path / "vocab", allow_short_horizon=True)

    assert result.report["readiness"]["status"] == "diagnostic"


def test_full_horizon_dataset_is_ready(built):
    """
    У набора без оговорок вердикт чистый: диагностику не ставят
    на всякий случай.
    """

    report, _target = built

    assert report["readiness"]["status"] == "ready"
    assert report["readiness"]["reasons"] == []


def test_catalog_must_be_the_one_split_was_built_on(dataset, tmp_path):
    """
    Справочник другого мира это не мелочь: из него придут другие
    названия продуктов и другая расшифровка точек.
    """

    root, out = dataset

    path = root / "train" / "catalog" / "products.parquet"

    table = pq.read_table(path)

    changed = table.set_column(
        table.column_names.index("product_name"),
        "product_name",
        pa.array(["другое имя"] * table.num_rows, pa.string()),
    )

    backup = path.with_suffix(".backup")
    path.replace(backup)

    try:
        pq.write_table(changed, path)

        with pytest.raises(CorpusError, match="не тот, на котором построено разделение"):
            FitCorpus.open(out, root / "train")

    finally:
        path.unlink()
        backup.replace(path)

    # Восстановленный справочник снова принимается.
    FitCorpus.open(out, root / "train")


# ============================================================
# НЕЗАВИСИМОСТЬ FIT ОТ ОСТАЛЬНЫХ ДАННЫХ
# ============================================================


def _statistics_digests(target: Path) -> dict[str, str]:
    directory = target / STATISTICS_DIR
    return {path.name: sha256_file(path) for path in sorted(directory.iterdir())}


def test_other_groups_and_future_records_do_not_change_fit(tmp_path):
    """
    Замена validation и test другими клиентами и запись после
    fit_end не меняют ни отпечатка содержимого, ни статистики.

    Правка видимой суммы train — меняет. Без этого контроля
    проверка доказывала бы только то, что отпечаток вообще
    считается.
    """

    base_root, base_out = prepared(tmp_path / "base")
    base = _contract(base_root, base_out, tmp_path / "base_vocab").report

    other_root, other_out = prepared(tmp_path / "other", future=True, variant=1)
    other = _contract(other_root, other_out, tmp_path / "other_vocab").report

    assert other["fit_content_sha256"] == base["fit_content_sha256"]
    assert other["corpus"]["events"] == base["corpus"]["events"]
    assert _statistics_digests(tmp_path / "other_vocab") == _statistics_digests(tmp_path / "base_vocab")

    edited_root, edited_out = prepared(tmp_path / "edited", amount_shift=25)
    edited = _contract(edited_root, edited_out, tmp_path / "edited_vocab").report

    assert edited["fit_content_sha256"] != base["fit_content_sha256"]


def test_fit_is_reproducible(dataset, tmp_path):
    """
    Тот же вход даёт тот же результат: статистика не зависит ни
    от порядка чтения, ни от случайности.
    """

    root, out = dataset

    first = _contract(root, out, tmp_path / "one").report
    second = _contract(root, out, tmp_path / "two").report

    assert first == second
    assert _statistics_digests(tmp_path / "one") == _statistics_digests(tmp_path / "two")


def test_artifacts_are_byte_identical_in_another_process(dataset, tmp_path):
    """
    Другой процесс с другим PYTHONHASHSEED даёт те же байты.

    Порядок обхода словарей Python зависит от хэшей строк, и
    статистика, собранная в таком словаре, легко начинает
    зависеть от запуска. Проверяется это только из отдельного
    процесса: внутри одного seed уже выбран.
    """

    import os
    import subprocess
    import sys

    root, out = dataset

    here = _contract(root, out, tmp_path / "here")

    script = (
        "from pathlib import Path;"
        "from src.tokenization.contract import build_contract;"
        "from src.tokenization.settings import TokenizerConfig;"
        f"build_contract(Path({str(out)!r}), Path({str(root / 'train')!r}),"
        f" Path({str(tmp_path / 'there')!r}), TokenizerConfig())"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONHASHSEED": "12345", "PYTHONPATH": str(Path.cwd())},
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")

    assert _statistics_digests(tmp_path / "there") == _statistics_digests(tmp_path / "here")

    assert sha256_file(tmp_path / "there" / FIT_MANIFEST_FILE) == sha256_file(
        tmp_path / "here" / FIT_MANIFEST_FILE
    )

    assert here.report["fit_content_sha256"]


# ============================================================
# КЛЮЧИ, ВЕС И ПРОПУСКИ
# ============================================================


def test_reference_keys_stay_links(built):
    """
    Восемь ключей-ссылок не получают кода: они связывают события
    клиента и значением модели не становятся.
    """

    report, _target = built

    rows = {row["key"]: row for row in report["keys"]["rows"]}

    assert report["keys"]["link"] == 8

    for key in ("account_ref", "card_ref", "contract_ref", "outlet_ref"):
        assert rows[key]["role"] == "link"
        assert rows[key]["value_kind"] == "reference"

    # Но сами связи наблюдаются и попадают в метаданные.
    assert report["references"]["account_ref"] > 0


def test_profile_key_is_counted_once_per_client(built):
    """
    Доход клиента это один факт, а не сто повторов по числу его
    покупок.
    """

    report, _target = built

    rows = {row["key"]: row for row in report["keys"]["rows"]}

    income = rows["profile_declared_income"]

    assert income["weight_rule"] == WEIGHT_PER_CLIENT
    assert income["observations"] == 1

    assert rows["transaction_amount"]["weight_rule"] == WEIGHT_PER_EVENT
    assert rows["transaction_amount"]["observations"] > 1

    # Прежнее и новое значение приходят событием, а не профилем.
    assert rows["profile_declared_income_new"]["weight_rule"] == WEIGHT_PER_EVENT


def test_declared_but_absent_payload_key_is_counted(built):
    """
    Пропуск считается только у того, что у этого типа события
    объявлено. Расчётному ключу пропуск не приписывается: у него
    есть причина.
    """

    report, _target = built

    missing = {(item["event_type"], item["key"]) for item in report["missing"]["top"]}

    # Причины отказа у прошедшей покупки нет, и поле объявлено.
    assert ("purchase", "decline_reason") in missing

    # Отношение к лимиту не «пропущено»: оно не посчитано, и
    # причина названа.
    assert not any(key == "amount_to_limit" for _event_type, key in missing)
    assert "amount_to_limit" in report["absent_reasons"]


def test_day_precision_event_is_visible_in_the_corpus(built):
    """
    У записи дневной точности час суток не наблюдался, и знать
    об этом обязан уже контракт входа.
    """

    report, _target = built

    assert report["corpus"]["day_precision_events"] >= 1


def test_unobserved_keys_stay_declared(built):
    """
    Ключ без наблюдений на train не ошибка и не повод заглянуть
    в validation.
    """

    report, _target = built

    unobserved = set(report["keys"]["unobserved_in_train"])

    assert unobserved
    assert "merchant_brand" in unobserved

    rows = {row["key"]: row for row in report["keys"]["rows"]}

    assert rows["merchant_brand"]["observed_in_train"] is False


# ============================================================
# ТИПЫ И ЗНАЧЕНИЯ
# ============================================================


def test_bool_int_and_string_are_different_values():
    """
    True, 1 и "1" в Python сравниваются между собой. Для словаря
    это разные значения, и склеить их нельзя.
    """

    assert value_type(True) == TYPE_BOOL
    assert value_type(1) == TYPE_INT
    assert value_type("1") == TYPE_STR

    assert value_text(True) == "true"
    assert value_text(1) == "1"

    # Ключ счётчика несёт тип, поэтому совпадения записи мало.
    assert (("k", TYPE_BOOL, "true") != ("k", TYPE_STR, "true"))


def test_categorical_values_carry_their_type(built):
    """
    В каталоге категорий у каждого значения записан его тип.
    """

    _report, target = built

    rows = pq.read_table(target / STATISTICS_DIR / CATEGORICAL_FILE).to_pylist()

    by_key = {}
    for row in rows:
        by_key.setdefault(row["key"], set()).add(row["value_type"])

    assert by_key["is_online"] == {TYPE_BOOL}
    assert by_key["event_type"] == {TYPE_STR}

    # mcc это код, а не величина: он остаётся категорией.
    assert "mcc" in by_key


# ============================================================
# ПРОТИВОРЕЧИЯ
# ============================================================


def test_text_that_looks_like_code_is_reported_not_fixed(dataset, tmp_path):
    """
    Реестр ведёт препроцессинг. Токенизатор называет
    несоответствие и следует реестру, а не переименовывает смысл
    у себя внутри.
    """

    from src.tokenization.contract import _text_contradictions

    root, out = dataset

    schema = SemanticSchema.open(out, "train")

    # Ключ offer был таким случаем на живых данных: реестр
    # объявлял его текстом, а на train это оказался закрытый
    # перечень кодов. Решение принято в смысловом слое, и теперь
    # он категория. Сама проверка нужна по-прежнему: следующий
    # такой ключ обязан быть назван так же громко.
    assert "offer" not in schema.text_keys

    stats = FitStatistics()
    stats.text["merchant_name"] = {
        name: TextEntry(count=5, clients=3, example_raw=name)
        for name in ("cash_loan", "deposit", "insurance", "market_promo")
    }

    found = _text_contradictions(stats, schema)

    assert [item["kind"] for item in found] == ["text_looks_like_code"]
    assert found[0]["key"] == "merchant_name"
    assert "cash_loan" in found[0]["examples"]

    # Свободный текст противоречием не считается.
    stats.text["merchant_name"] = {
        "европharma алматы": TextEntry(count=1, clients=1, example_raw="Европharma Алматы"),
        "магнум астана": TextEntry(count=1, clients=1, example_raw="Магнум Астана"),
    }

    assert _text_contradictions(stats, schema) == []


def test_identifiers_inside_text_are_reported(dataset):
    """
    Внутри устойчивого имени контрагента встречаются
    идентификаторы: BPE будет резать их как обычный текст, и
    молчать об этом нельзя.
    """

    from src.tokenization.contract import _text_contradictions

    _root, out = dataset

    schema = SemanticSchema.open(out, "train")

    stats = FitStatistics()
    stats.text["counterparty"] = {
        "payer_08916177": TextEntry(count=3, clients=1, example_raw="payer_08916177"),
        "o. sarsenbay": TextEntry(count=3, clients=1, example_raw="O. Sarsenbay"),
    }

    kinds = [item["kind"] for item in _text_contradictions(stats, schema)]

    assert "text_contains_identifiers" in kinds


def test_numeric_unit_pointing_at_another_key_is_an_accepted_limit(built):
    """
    Единица original_amount лежит в соседнем ключе, поэтому общей
    шкалы у него быть не может. Решение принято: шкалы нет, и это
    записано как ограничение V1, а не висит открытым вопросом.
    """

    report, target = built

    accepted = {item["key"]: item for item in report["accepted_limits"]}

    assert accepted["original_amount"]["kind"] == "unit_is_another_key"
    assert "transaction_amount" in accepted["original_amount"]["decision"]

    # Среди открытых противоречий его больше нет.
    assert "original_amount" not in {item["key"] for item in report["contradictions"]}

    # Ограничение названо там же, где остальные ограничения входа.
    assert any(item.startswith("original_amount:") for item in report["limitations"])

    # Решение записано в конфигурации, а не спрятано в коде:
    # шкалы у ключа нет, значение получит числовое [UNK].
    encoders = read_json(target / CONFIG_FILE)["numeric_encoders"]

    assert encoders["original_amount"]["method"] == "unfitted"
    assert encoders["original_amount"]["boundaries"] == []


def test_offer_is_a_category_and_keeps_its_own_domain(built):
    """
    Код предложения это закрытый перечень, а не свободный текст:
    разбивать его на куски нечего.

    Домен у него свой: предложение в баннере называет повод
    показа, а не продукт.
    """

    report, _target = built

    rows = {row["key"]: row for row in report["keys"]["rows"]}

    assert rows["offer"]["value_kind"] == "categorical"

    # И как текст он больше не выглядит: противоречия по нему нет.
    assert "offer" not in {item["key"] for item in report["contradictions"]}


# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================


def test_every_numeric_key_needs_an_encoder(dataset, tmp_path):
    """
    Число без объявленного способа кодирования в словарь не
    попадает молча.
    """

    root, out = dataset

    config = TokenizerConfig()

    encoders = dict(config.numeric_encoders)
    encoders.pop("transaction_amount")

    with pytest.raises(ContractError, match="без кодировщика"):
        _contract(root, out, tmp_path / "vocab", replace(config, numeric_encoders=encoders))


def test_domain_cannot_merge_keys_declared_incompatible(dataset, tmp_path):
    """
    Объединение, запрещённое смысловым реестром, не проходит
    через конфигурацию.
    """

    root, out = dataset

    config = replace(
        TokenizerConfig(),
        value_domains=(
            ValueDomain("channel_domain", ("operation_channel", "application_channel"), "похожи"),
        ),
    )

    with pytest.raises(ContractError, match="несовместимыми"):
        _contract(root, out, tmp_path / "vocab", config)


def test_domain_cannot_mix_value_kinds(dataset, tmp_path):
    """
    Домен это множество значений одного вида.
    """

    root, out = dataset

    config = replace(
        TokenizerConfig(),
        value_domains=(ValueDomain("mixed", ("event_type", "transaction_amount"), "нет"),),
    )

    with pytest.raises(ContractError, match="разных видов"):
        _contract(root, out, tmp_path / "vocab", config)


def test_unknown_config_key_is_refused():

    with pytest.raises(ConfigError, match="неизвестные ключи"):
        TokenizerConfig.from_dict({"fit_group": "train", "выдуманное": 1})


def test_written_configuration_reads_back(built, tmp_path):
    """
    Записанный нами же tokenizer_config.json обязан читаться
    обратно и давать ту же контрольную сумму.

    Иначе собственный артефакт нельзя ни передать команде, ни
    сверить с манифестом, а заметно это становится далеко от
    места ошибки.
    """

    _report, target = built

    restored = TokenizerConfig.load(target / CONFIG_FILE)

    assert restored.sha256() == TokenizerConfig().sha256()
    assert restored.as_dict() == TokenizerConfig().as_dict()


def test_recorded_decision_cannot_be_changed_through_the_file():
    """
    Список намеренно не объединённых доменов это запись решения,
    а не настройка.
    """

    payload = TokenizerConfig().as_dict()
    payload["declined_domains"] = []

    with pytest.raises(ConfigError, match="запись принятого решения"):
        TokenizerConfig.from_dict(payload)


def test_quantile_boundaries_cannot_be_set_by_hand():
    """
    Границы квантилей считает train. Заданные руками, они были бы
    не квантилями, а чужой шкалой под именем квантилей.
    """

    with pytest.raises(ConfigError, match="квантильные границы"):
        TokenizerConfig.from_dict(
            {"numeric_encoders": {"transaction_amount": {"method": "quantile", "bins": 8,
                                                         "boundaries": [1, 2, 3]}}}
        )


# ============================================================
# СОГЛАСИЕ РЕЕСТРА И КОДА
# ============================================================


def test_registry_of_another_code_version_is_refused(dataset, tmp_path, monkeypatch):
    """
    Реестр, собранный другой версией смыслового слоя, к fit не
    допускается: описание смысла и сами значения разошлись бы
    молча.

    Расходится именно ВЕРСИЯ, а не файл: правленый файл ловит
    проверка выходов этапа, и это отдельный случай.
    """

    root, out = dataset

    monkeypatch.setattr("src.tokenization.contract.KEYS_VERSION", "0.0.1")

    with pytest.raises(ContractError, match="другой версией кода"):
        _contract(root, out, tmp_path / "vocab")


def test_edited_semantic_output_is_refused(dataset, tmp_path):
    """
    Выходы этапа сверяются по файлам: правленый реестр не
    становится входом словаря.
    """

    root, out = dataset

    path = out / "semantic" / "train" / SEMANTIC_REGISTRY_FILE

    original = path.read_bytes()

    report = read_json(path)
    report["clients_checked"] = 999
    write_json(path, report)

    try:
        with pytest.raises(CorpusError, match="изменились после сборки"):
            _contract(root, out, tmp_path / "vocab")
    finally:
        path.write_bytes(original)


# ============================================================
# ВЕРСИЯ РЕАЛИЗАЦИИ
# ============================================================


def test_version_tracks_sources():
    """
    Правка любого модуля пакета без поднятия версии реализации
    не проходит: прежние артефакты считались бы актуальными.
    """

    from src.tokenization import (
        categorical,
        contract,
        corpus,
        encode,
        layout,
        numeric,
        report as report_module,
        run,
        scan,
        schema,
        settings,
        text,
        transform,
        version,
    )

    modules = {
        "categorical": categorical,
        "contract": contract,
        "corpus": corpus,
        "encode": encode,
        "layout": layout,
        "numeric": numeric,
        "report": report_module,
        "run": run,
        "scan": scan,
        "schema": schema,
        "settings": settings,
        "text": text,
        "transform": transform,
        "version": version,
    }

    stored = json.loads(Path("tests/tok_sources.json").read_text(encoding="utf-8"))

    assert stored["version"] == version.IMPLEMENTATION_VERSION

    actual = {
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in modules.items()
    }

    assert stored["modules"] == actual, (
        "модули токенизатора изменены: поднимите IMPLEMENTATION_VERSION "
        "и обновите tests/tok_sources.json"
    )
