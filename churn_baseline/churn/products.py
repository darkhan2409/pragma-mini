from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from .config import REPO


# ============================================================
# ПРОДУКТЫ БАНКА
# ============================================================
#
# Выгрузка называет продукт только его product_id: семейство в событие не
# копируется. Справочник продуктов у банка есть всегда, это не скрытое
# знание о клиенте, поэтому семейство берётся из него.
#
# Настоящие продукты — reference/home_product_timeline.json. Вымышленные
# SYNTH_* генератор добавляет сам (src/generator/params/products.py), их
# семейства перечислены здесь; сверку с каталогом генератора держит
# tests/test_lifecycle_products.py в корне репозитория.
#
# Премиальные карты — Тау и Алем. Excel CAPP, ячейка I7: «закрытие
# премиальных карт», коды DC_VIP*, DC_VRTV* (Тау) и DC_ALEM, DC_VRT_ALM,
# DC_ALM_NR (Алем).
# ============================================================


CATALOG = REPO / "reference" / "home_product_timeline.json"

SYNTHETIC: dict[str, str] = {
    "prd_synth_deposit_kids": "deposit",
    "prd_synth_card_youth": "debit_card",
    "prd_synth_loan_green": "cash_loan",
    "prd_synth_loan_green_2": "cash_loan",
}

DEBIT_CARD = "debit_card"

# Договоров по сервисным продуктам (приложение, тарифы, бонус) нет:
# продуктом клиента они не считаются.
SERVICE = "service"

PREMIUM: frozenset[str] = frozenset({"prd_tau", "prd_alem"})


@cache
def families(path: Path = CATALOG) -> dict[str, str]:
    """
    product_id → семейство продукта.
    """
    products = json.loads(path.read_text(encoding="utf-8"))["products"]
    out = {item["product_id"]: item["family"] for item in products}
    clash = set(out) & set(SYNTHETIC)
    if clash:
        raise ValueError(f"вымышленные продукты совпали с настоящими: {sorted(clash)}")
    return {**out, **SYNTHETIC}
