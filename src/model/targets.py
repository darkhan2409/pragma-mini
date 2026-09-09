from __future__ import annotations

from src.tokenizer.masking import excluded_field_names, resolve_excluded


# ============================================================
# ИДЕЯ
# ============================================================
#
# Не всякое предсказуемое поле стоит предсказывать.
#
# timeline__event_type восстанавливается из состава самого
# события: у транзакции есть mcc и сумма, у экрана есть
# firebase_screen. Модель угадывает тип по полям, которые
# лежат рядом в том же событии, и высокая точность здесь
# ничего не говорит об истории.
#
# profile_snapshot__* дублируют as-of профиль, который модель
# и так видит позицией 0. Это тоже не про историю, а про
# копию входа.
#
# Раньше оба множества исключались только из АГРЕГАТА отчёта:
# модель всё равно на них училась, и их лёгкость подмешивалась
# в loss. Политика целей убирает их из задачи целиком, оставляя
# входом: значения на месте, маски на них не ставятся.
#
# Исключение задаётся ПАТТЕРНАМИ имён, а не готовым множеством
# идентификаторов. Причина техническая и важная: MaskingConfig
# уезжает в checkpoint и восстанавливается из него там, где
# словаря под рукой нет (сравнение двух запусков, диагностика,
# ablation). Паттерн переживает такую поездку, разрешённый
# список id не пережил бы.
# ============================================================


POLICY_ALL = "all"
POLICY_HISTORY = "history"

TARGET_POLICIES: tuple[str, ...] = (POLICY_ALL, POLICY_HISTORY)

DEFAULT_POLICY = POLICY_ALL

# Паттерны в стиле имён файлов: точное имя или префикс с *.
HISTORY_EXCLUDES: tuple[str, ...] = (
    "timeline__event_type",
    "profile_snapshot__*",
)

POLICY_NOTES = {
    POLICY_ALL: "цели ставятся на все предсказуемые поля",
    POLICY_HISTORY: (
        "цели только про историю: тип события и снимок профиля остаются "
        "входом, но не предсказываются"
    ),
}


def exclude_patterns(policy: str = DEFAULT_POLICY) -> tuple[str, ...]:
    """
    Паттерны имён полей, которые политика убирает из целей.
    """

    if policy not in TARGET_POLICIES:
        raise ValueError(
            f"неизвестная политика целей {policy!r}, ожидалась одна из {TARGET_POLICIES}"
        )

    return HISTORY_EXCLUDES if policy == POLICY_HISTORY else ()


def excluded_fields(table, policy: str = POLICY_HISTORY) -> frozenset[str]:
    """
    Поля, которые сами по себе не про историю.

    Отвечает на вопрос отчёта «а что будет без них», поэтому
    по умолчанию это набор политики history, даже если запуск
    учился на всех полях.
    """

    names = [table.name(key_id) for key_id in table.trainable_key_ids]

    return resolve_excluded(names, exclude_patterns(policy)) if policy != POLICY_ALL else frozenset()


__all__ = [
    "DEFAULT_POLICY",
    "HISTORY_EXCLUDES",
    "POLICY_ALL",
    "POLICY_HISTORY",
    "POLICY_NOTES",
    "TARGET_POLICIES",
    "exclude_patterns",
    "excluded_field_names",
    "excluded_fields",
    "resolve_excluded",
]
