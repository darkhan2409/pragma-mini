from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .keys import MERCHANT_KEYS


# ============================================================
# ИДЕЯ
# ============================================================
#
# Расшифровка торговой точки по справочнику: сектор, категория,
# подкатегория, бренд, район, канал и ценовой сегмент. Само
# событие несёт только название в терминальной строке, MCC и
# город; остальное лежит в справочнике мерчантов.
#
# Сырой outlet_id наружу не выходит: он остаётся связью, а в
# признаки идут расшифрованные атрибуты. Происхождение значения
# указывает на справочник и на локальную ссылку точки.
#
# Точки, которой в справочнике нет, не выдумывается ничего:
# признаки остаются пустыми с причиной merchant_unknown.
#
# Частотных порогов и уровней редкости здесь нет: они относятся
# к обучению словарей на train и делаются позже.
# ============================================================


UNKNOWN_REASON = "merchant_unknown"


class MerchantCatalog:
    """
    Справочник мерчантов для расшифровки. Читается один раз.
    """

    COLUMNS: tuple[str, ...] = ("outlet_id", *MERCHANT_KEYS)

    def __init__(self, table: pa.Table | None):

        self._rows: dict[str, dict] = {}

        if table is None:
            return

        names = [name for name in self.COLUMNS if name in table.column_names]

        for row in table.select(names).to_pylist():
            self._rows[row["outlet_id"]] = row

    @staticmethod
    def open(raw_dir: Path | None) -> "MerchantCatalog":

        if raw_dir is None:
            return MerchantCatalog(None)

        path = Path(raw_dir) / "catalog" / "merchants.parquet"

        return MerchantCatalog(pq.read_table(path) if path.exists() else None)

    @property
    def size(self) -> int:
        return len(self._rows)

    def decode(self, outlet_id: str | None) -> tuple[dict[str, object], str | None]:
        """
        Атрибуты точки и причина, если расшифровать не удалось.
        """

        if not self._rows:
            return {}, "merchant_catalog_absent"

        if outlet_id is None:
            return {}, None

        row = self._rows.get(outlet_id)

        if row is None:
            return {}, UNKNOWN_REASON

        return (
            {
                key.key: row[column]
                for column, key in MERCHANT_KEYS.items()
                if row.get(column) is not None
            },
            None,
        )


__all__ = ["MERCHANT_KEYS", "UNKNOWN_REASON", "MerchantCatalog"]
