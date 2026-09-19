from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.preprocessing.run import EXIT_CONTRACT_MISMATCH, EXIT_OK
from src.preprocessing.run import main as run_main
from src.preprocessing.settings import PreprocessingConfig

from tests.prep_fixtures import MiniRaw, purchase_payload


# ============================================================
# ИДЕЯ
# ============================================================
#
# Один небольшой набор из трёх групп, прошедший все пять этапов
# препроцессинга, на котором проверяется токенизатор.
#
# Train-клиент сделан богатым нарочно: в нём есть ноль, крайние
# суммы, текст с казахскими буквами, запись дневной точности,
# вклад и кредитная карта, две версии профиля и изменение
# профиля. Без этого проверки границ, пропусков и трассировки
# были бы проверками пустоты.
#
# Отдельно лежат клиент без событий и клиент без профиля:
# молчание это факт, а не причина потерять человека.
#
# Группа val держит продукт, которого нет у train: так
# проверяется, что новое значение при transform не расширяет
# словарь.
# ============================================================


CONFIG = PreprocessingConfig()

FULL_HORIZON = datetime(2023, 1, 1)
FIT_END = CONFIG.windows["train"].final_cutoff

WORLD_SEED = 77
SEEDS = {"train": 1, "val": 2, "test": 3}

# Суммы покупок train-клиента: ноль, обычные и крайние.
TRAIN_AMOUNTS: tuple[int, ...] = (0, 12_500, 199_999, 200_000, 350_000, 5_000_000)

# Названия точек: казахский и русский сохраняются без
# транслитерации, а шумные варианты одного названия это
# нормальная жизнь терминальной строки.
MERCHANT_NAMES: tuple[str, ...] = (
    "Europharma Алматы",
    "Қазпошта",
    "EUROPHARMA ALMATY",
    "Europharma  Алматы",
    "Магнум Астана",
    "Europharma Алматы",
)

PRODUCTS: tuple[dict, ...] = (
    {"product_id": "prd_dep", "product_code": "DEP", "product_name": "Депозит «Сберегательный»",
     "product_family": "deposit"},
    {"product_id": "prd_cc", "product_code": "CC", "product_name": "Кредитная карта Tau",
     "product_family": "credit_card"},
    # Продукт, которого train не видел. Семейство у него то же,
    # что у известной карты: так при transform проверяется, что
    # неизвестный код становится [UNK], а известное семейство
    # сохраняет свой смысл.
    {"product_id": "prd_new", "product_code": "NEW", "product_name": "Кредитная карта Jaña",
     "product_family": "credit_card"},
)


def cli(*args: str) -> int:
    with pytest.raises(SystemExit) as result:
        run_main(list(args))
    return int(result.value.code)


def _catalog(mini: MiniRaw) -> None:
    """
    Общий мир трёх групп: справочник продуктов у всех один.
    """

    for item in PRODUCTS:
        mini.product(valid_from=FULL_HORIZON, **item)


def _product_payload(product: dict, contract_id: str, account_id: str, amount: int,
                     term: int | None = 12, rate: float = 0.14) -> dict:
    return {
        "product_id": product["product_id"],
        "product_code": product["product_code"],
        "product_version": 1,
        "tariff_version": 1,
        "product_family": product["product_family"],
        "contract_id": contract_id,
        "account_id": account_id,
        "card_id": None,
        "offer_id": None,
        "previous_product_id": None,
        "migration_reason": None,
        "amount_or_limit": amount,
        "term": term,
        "rate": rate,
        "reason": "application",
        "timestamp_quality": "exact",
    }


def _rich_client(mini: MiniRaw, client_id: str = "train_c1", amount_shift: int = 0,
                 future: bool = False) -> None:
    """
    Train-клиент со всем, что должен уметь кодировать V1.
    """

    mini.cover_all(client_id, first_seen="2023-01-01")

    mini.profile_version(
        client_id, 1, "2023-01-01",
        declared_income=300_000, age=34, children=1, city="Almaty", region="Almaty",
        education="higher", family_status="married", housing_type="own", income_type="salary",
        industry="it", pensioner=False, salary_day=10, gender="male",
        credit_limit=400_000, credit_utilization=0.35, contracts_count=2, active_contracts=2,
        relationship_months=48, holds_debit_card=True, holds_credit_card=True, holds_deposit=True,
    )

    mini.profile_version(
        client_id, 2, "2025-03-01",
        declared_income=450_000, age=36, children=2, city="Astana", region="Astana",
        education="higher", family_status="married", housing_type="own", income_type="salary",
        industry="it", pensioner=False, salary_day=10, gender="male",
        credit_limit=400_000, credit_utilization=0.41, contracts_count=2, active_contracts=2,
        relationship_months=72, holds_debit_card=True, holds_credit_card=True, holds_deposit=True,
    )

    # Вклад и кредитная карта: одно и то же поле означает разное.
    mini.event(
        client_id, "product_opened", "2024-02-01 10:00:00",
        correlation_id="ctr_dep", link_type="contract",
        payload=_product_payload(PRODUCTS[0], "ctr_dep", "acc_dep", 1_000_000),
    )

    mini.event(
        client_id, "product_opened", "2024-03-01 10:00:00",
        correlation_id="ctr_cc", link_type="contract",
        payload=_product_payload(PRODUCTS[1], "ctr_cc", "acc_cc", 400_000, term=None, rate=0.0),
    )

    # Доход: от него считается время до следующей траты.
    mini.event(
        client_id, "salary_credit", "2025-01-10 09:00:00",
        payload=purchase_payload(amount=450_000, direction="credit", reason="salary",
                                 account_id="acc_main", card_id=None, merchant_id=None,
                                 outlet_id=None, merchant_name=None, mcc=None,
                                 merchant_city=None, merchant_country=None,
                                 counterparty="Employer", balance_after=450_000),
    )

    purchases: list[str] = []

    for index, amount in enumerate(TRAIN_AMOUNTS):

        moment = datetime(2025, 2, 1 + index, 12, 0, 0)

        purchases.append(
            mini.event(
                client_id, "purchase", moment,
                payload=purchase_payload(
                    amount=amount + amount_shift,
                    account_id="acc_cc",
                    contract_id="ctr_cc",
                    merchant_name=MERCHANT_NAMES[index],
                    balance_after=500_000 - amount,
                ),
            )
        )

    # Возврат по конкретной покупке: отсюда берутся признаки
    # связи вместо идентификатора причины.
    mini.event(
        client_id, "refund", "2025-02-20 12:00:00",
        payload=purchase_payload(
            amount=12_500 + amount_shift,
            direction="credit",
            reason="refund",
            account_id="acc_cc",
            contract_id="ctr_cc",
            cause_event_id=purchases[1],
            merchant_name=MERCHANT_NAMES[1],
            balance_after=512_500,
        ),
    )

    # Запись дневной точности: час суток у неё не наблюдался.
    mini.event(
        client_id, "installment_due", "2025-04-15 00:00:00", precision="day",
        payload={
            "contract_id": "ctr_cc", "installment_no": 3, "amount_due": 45_000,
            "amount_paid": None, "principal_outstanding": 180_000, "days_past_due": 0,
            "due_date": "2025-04-15", "cause_event_id": None, "reason": "schedule",
        },
    )

    # Исправленная запись: версия 2 уточняет сумму. Обе версии
    # лежат на одном времени события, поэтому на срезе раньше их
    # нет вовсе, а на срезе позже действует вторая.
    corrected = mini.event(
        client_id, "purchase", "2025-03-10 12:00:00",
        payload=purchase_payload(amount=33_000, account_id="acc_cc", contract_id="ctr_cc",
                                 merchant_name="Magnum Astana", balance_after=467_000),
    )

    mini.event(
        client_id, "purchase", "2025-03-10 12:00:00", event_id=corrected, version=2,
        payload=purchase_payload(amount=44_000, account_id="acc_cc", contract_id="ctr_cc",
                                 merchant_name="Magnum Astana", balance_after=456_000),
    )

    # Изменение профиля: смысл old_value и new_value задаёт само
    # изменившееся поле.
    mini.event(
        client_id, "profile_change", "2025-03-01 08:00:00", initiator="bank",
        payload={
            "field_name": "declared_income",
            "old_value": "300000",
            "new_value": "450000",
            "change_source": "document",
            "confirmed": True,
        },
    )

    if future:
        # Событие после fit_end и его исправление: ни то, ни
        # другое в корпус попасть не должно.
        event_id = mini.event(
            client_id, "purchase", "2026-02-10 12:00:00",
            payload=purchase_payload(amount=777_000, account_id="acc_cc", contract_id="ctr_cc"),
        )
        mini.event(
            client_id, "purchase", "2026-02-10 12:00:00", event_id=event_id, version=2,
            payload=purchase_payload(amount=888_000, account_id="acc_cc", contract_id="ctr_cc"),
        )


def _quiet_client(mini: MiniRaw, client_id: str = "train_c2") -> None:
    """
    Клиент, о котором источники молчат: ни одного события и ни
    одной версии профиля.
    """

    mini.cover_all(client_id, first_seen="2023-01-01")


def _plain_client(mini: MiniRaw, client_id: str, amount: int = 12_500,
                  product: dict | None = None, at: str = "2026-03-05 10:00:00") -> None:
    """
    Клиент других групп: одна покупка и, если попросили,
    продукт, которого train не видел.
    """

    mini.cover_all(client_id, first_seen="2023-01-01")
    mini.profile_version(client_id, 1, "2023-01-01", declared_income=200_000, age=41, city="Shymkent")

    mini.event(client_id, "purchase", at, payload=purchase_payload(amount=amount,
                                                                   merchant_name="Sulpak Шымкент"))

    if product is not None:
        mini.event(
            client_id, "product_opened", at, correlation_id="ctr_new", link_type="contract",
            payload=_product_payload(product, "ctr_new", "acc_new", 600_000),
        )


def build_raw(root: Path, future: bool = False, amount_shift: int = 0,
              history_start: datetime = FULL_HORIZON, variant: int = 0) -> Path:
    """
    Три независимые группы одного мира.

    variant меняет содержимое ТОЛЬКО val и test: на fit это
    влиять не должно, и ровно это проверяется.
    """

    root = Path(root)

    for name in ("train", "val", "test"):

        mini = MiniRaw(root / name, history_start=history_start, seed=SEEDS[name], world_seed=WORLD_SEED)

        _catalog(mini)

        if name == "train":
            _rich_client(mini, amount_shift=amount_shift, future=future)
            _quiet_client(mini)
        elif name == "val":
            # Продукт, которого train не видел: при transform он
            # обязан стать [UNK], а не расширить словарь.
            _plain_client(
                mini, f"val_c{1 + variant}", amount=44_000 + 1_000 * variant,
                product=PRODUCTS[2], at="2026-03-05 10:00:00",
            )
        else:
            _plain_client(
                mini, f"test_c{1 + variant}", amount=61_000 + 1_000 * variant,
                at="2026-07-05 10:00:00",
            )

        mini.write()

    return root


def run_preprocessing(root: Path, out: Path, name: str = "tok") -> None:
    """
    Пять этапов препроцессинга на этом наборе.
    """

    root = Path(root)
    out = Path(out)

    assert cli("passport", "--raw-root", str(root), "--out", str(out), "--name", name) in (
        EXIT_OK,
        EXIT_CONTRACT_MISMATCH,
    )
    assert cli("canonical", "--raw-root", str(root), "--out", str(out), "--name", name) == EXIT_OK
    assert cli("split", "--raw-root", str(root), "--out", str(out), "--name", name) == EXIT_OK

    for group in ("train", "val", "test"):
        assert cli(
            "semantic", "--name", name, "--group", group,
            "--raw", str(root / group), "--out", str(out),
        ) == EXIT_OK


def prepared(tmp_path: Path, future: bool = False, amount_shift: int = 0,
             history_start: datetime = FULL_HORIZON, variant: int = 0) -> tuple[Path, Path]:
    """
    Готовый набор: RAW и каталог обработанных данных.
    """

    root = build_raw(tmp_path / "raw", future=future, amount_shift=amount_shift,
                     history_start=history_start, variant=variant)
    out = tmp_path / "processed"

    run_preprocessing(root, out)

    return root, out


def build_vocab(root: Path, out: Path, target: Path, config=None, group: str = "train",
                allow_short_horizon: bool = False):
    """
    Четыре этапа токенизатора до заморозки включительно.
    """

    from src.tokenization.categorical import build_values
    from src.tokenization.contract import build_contract
    from src.tokenization.layout import build_layout
    from src.tokenization.numeric import build_buckets
    from src.tokenization.settings import TokenizerConfig

    config = config or TokenizerConfig()

    build_contract(out, Path(root) / group, target, config, group=group,
                   allow_short_horizon=allow_short_horizon)
    build_values(target, out, config, group)
    build_buckets(target, out, config, group)

    return build_layout(target, out, config, group)
