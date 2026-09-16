from __future__ import annotations

from dataclasses import dataclass, field


# ============================================================
# СОЦИАЛЬНЫЙ ГРАФ
# ============================================================
#
# Граф строится ДО симуляции и детерминирован: состав сообщества
# определяется только client_ordinal и размером сообщества,
# рёбра внутри сообщества разыгрываются своим потоком.
#
# Сообщество это единица симуляции: внутрибанковский перевод
# должен дойти до получателя и повлиять на его решения, значит
# обе стороны обязаны жить в одной очереди событий.
# ============================================================


RELATION_TYPES = (
    "spouse",
    "relative",
    "friend",
    "colleague",
    "employer",
    "landlord",
    "own_account_other_bank",
    "regular_counterparty",
    "random_counterparty",
)


@dataclass(frozen=True)
class RelationshipParams:

    relation_types: tuple = RELATION_TYPES

    # Размер сообщества. Меняет состав связей, поэтому уезжает
    # в манифест и участвует в отпечатке параметров.
    community_size: int = 32

    # Доля клиентов сообщества, состоящих в домохозяйстве
    # с другим клиентом сообщества.
    household_pair_share: float = 0.22
    household_extra_member_share: float = 0.18

    # Сколько связей каждого типа у клиента.
    degree: dict = field(
        default_factory=lambda: {
            "relative": (0, 4),
            "friend": (0, 5),
            "colleague": (0, 3),
            "regular_counterparty": (1, 4),
            "random_counterparty": (0, 6),
        }
    )

    # Доля связей, ведущих к другому клиенту этого же банка
    # внутри сообщества (остальные внешние).
    internal_share: dict = field(
        default_factory=lambda: {
            "spouse": 0.55,
            "relative": 0.30,
            "friend": 0.28,
            "colleague": 0.22,
            "regular_counterparty": 0.18,
            "random_counterparty": 0.10,
        }
    )

    # Сила связи: она задаёт, насколько часто к контрагенту
    # возвращаются.
    strength_range: dict = field(
        default_factory=lambda: {
            "spouse": (0.75, 1.00),
            "relative": (0.35, 0.85),
            "friend": (0.25, 0.75),
            "colleague": (0.15, 0.55),
            "employer": (0.85, 1.00),
            "landlord": (0.60, 0.95),
            "own_account_other_bank": (0.55, 0.95),
            "regular_counterparty": (0.35, 0.80),
            "random_counterparty": (0.02, 0.20),
        }
    )

    # Сколько переводов в месяц ожидается по связи.
    frequency_per_month: dict = field(
        default_factory=lambda: {
            "spouse": (1.5, 8.0),
            "relative": (0.2, 2.5),
            "friend": (0.1, 2.0),
            "colleague": (0.05, 1.2),
            "employer": (0.0, 0.0),
            "landlord": (1.0, 1.0),
            "own_account_other_bank": (0.3, 4.0),
            "regular_counterparty": (0.2, 2.0),
            "random_counterparty": (0.0, 0.3),
        }
    )

    # Типичная сумма перевода как доля месячного дохода.
    amount_share_of_income: dict = field(
        default_factory=lambda: {
            "spouse": (0.05, 0.45),
            "relative": (0.03, 0.30),
            "friend": (0.01, 0.15),
            "colleague": (0.01, 0.10),
            "landlord": (0.18, 0.34),
            "own_account_other_bank": (0.08, 0.60),
            "regular_counterparty": (0.02, 0.22),
            "random_counterparty": (0.01, 0.12),
        }
    )

    # Доля переводов, уходящих знакомому контрагенту.
    known_counterparty_share: tuple = (0.62, 0.94)

    # Собственный счёт в другом банке есть почти у всех, кроме
    # клиентов, для которых этот банк единственный.
    own_other_bank_share: dict = field(
        default_factory=lambda: {
            "primary": 0.55,
            "secondary": 0.92,
            "credit_only": 0.88,
            "deposit_only": 0.85,
            "episodic": 0.80,
        }
    )

    # Связи появляются и прекращаются во времени.
    relation_end_share_per_year: dict = field(
        default_factory=lambda: {
            "friend": 0.18,
            "colleague": 0.28,
            "regular_counterparty": 0.22,
            "random_counterparty": 0.65,
        }
    )

    # Что происходит при нехватке средств для перевода.
    transfer_shortfall: dict = field(
        default_factory=lambda: {
            "topup_from_other_bank": 0.34,
            "reduce_amount": 0.30,
            "client_cancels": 0.22,
            "declined": 0.14,
        }
    )

    reduce_amount_factor: tuple = (0.25, 0.85)

    # Редкий сетевой сценарий: цепочка переводов через дропов.
    mule_ring_share: float = 0.004
    mule_ring_size: tuple = (2, 4)
