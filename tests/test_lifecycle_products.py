from __future__ import annotations

import re
import sys
from pathlib import Path

from src.generator.world import products as catalog_module


# ============================================================
# ИДЕЯ
# ============================================================
#
# Стадии CAPP (churn_baseline/churn/lifecycle.py) берут семейство
# продукта из своего справочника: в событии его нет. Справочник обязан
# совпадать с каталогом генератора — иначе дебетовая или премиальная
# карта незаметно выпала бы из правил CORE и Loyal.
#
# Отдельно: стадии CAPP не использует ни PRAGMA, ни генератор: стадия —
# разрез результата, не вход и не цель. Что truth/ никто вне генератора
# не читает, проверяет tests/test_generator_isolation.py.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]


def churn_products():
    """
    Справочник churn_baseline: отдельный проект, импорт — из его каталога.
    """
    sys.path.insert(0, str(ROOT / "churn_baseline"))
    try:
        from churn import products
    finally:
        sys.path.remove(str(ROOT / "churn_baseline"))
    return products


def test_the_lifecycle_product_reference_matches_the_generator_catalog():

    catalog = catalog_module.catalog()
    generator = {
        view.record.product_id: view.family
        for family in catalog.families()
        for view in catalog.by_family(family)
    }

    products = churn_products()

    assert products.families() == generator
    assert products.PREMIUM <= set(generator)
    assert {generator[item] for item in products.PREMIUM} == {products.DEBIT_CARD}


def test_pragma_and_the_generator_never_use_the_lifecycle():
    """
    Стадии CAPP считает churn_baseline для анализа после прогноза. Ни вход
    PRAGMA, ни набор, ни downstream, ни генератор их не читают: генератор
    моделирует поведение, а не подгоняется под стадии.
    """

    imports = re.compile(r"from churn\b|import churn\b|churn\.lifecycle|churn/lifecycle")
    fields = re.compile(r"\bstage_at\b|current_stage|previous_stage|stage_since|at_risk_reason|transition_reason")

    offenders = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        name = str(path.relative_to(ROOT))
        if imports.search(text):
            offenders.append(name)
        # У генератора своё скрытое состояние клиента (behaviour/
        # engagement.py), с бизнес-стадиями CAPP оно не связано.
        if not path.is_relative_to(ROOT / "src" / "generator") and fields.search(text):
            offenders.append(name)

    assert offenders == []
