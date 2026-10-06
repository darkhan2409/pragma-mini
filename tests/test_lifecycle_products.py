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
# Отдельно: этапы PRAGMA не читают truth/ — служебную правду генератора
# для оценки (CLAUDE.md).
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


def test_pragma_stages_never_read_truth():

    pattern = re.compile(r"""["'/]truth\b""")

    offenders = [
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "src").rglob("*.py"))
        if not path.is_relative_to(ROOT / "src" / "generator") and pattern.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []


def test_pragma_never_sees_lifecycle_metadata():
    """
    Стадия, прежняя стадия и причины переходов — метаданные задач
    churn_baseline. Вход PRAGMA, набор и downstream их не читают.
    """

    names = re.compile(
        r"lifecycle_tasks|churn\.lifecycle|current_stage|previous_active_stage|at_risk_reason|"
        r"at_risk_previous_stage|first_at_risk_at"
    )

    offenders = [
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "src").rglob("*.py"))
        if not path.is_relative_to(ROOT / "src" / "generator") and names.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []
