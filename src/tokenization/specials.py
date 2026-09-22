from __future__ import annotations

from pathlib import Path

from src.preprocessing.artifacts import read_json

from .settings import SPECIAL_TOKENS_FILE, vocab_path


# ============================================================
# ЭТАП 1: СПЕЦИАЛЬНЫЕ ТОКЕНЫ
# ============================================================
#
# Их пять, и каждый действительно нужен кодированию или обучению:
#
#   [PAD]   выравнивание batch; токенизатор его не пишет
#   [UNK]   значение известного вида, которого на train не было
#   [MASK]  скрытое значение при обучении; в данных не лежит
#   [EVT]   начало события
#   [USR]   начало профиля
#
# Токенов «нет значения», «пустой текст» и «невозможное число»
# здесь нет намеренно. Отсутствующее поле просто не попадает в
# последовательность: пара ключ/значение существует тогда, когда
# значение есть. Пустой после нормализации текст это то же
# отсутствие. Невозможное число отбрасывает препроцессинг, и до
# словаря оно не доходит.
#
# Состав задан форматом, а не данными: читать здесь нечего и
# учиться нечему. Номера занимают начало пространства ID, и
# ключи продолжают его сразу за ними.
# ============================================================


PAD = "[PAD]"
UNK = "[UNK]"
MASK = "[MASK]"
EVT = "[EVT]"
USR = "[USR]"

SPECIAL_TOKENS: tuple[str, ...] = (PAD, UNK, MASK, EVT, USR)

SPECIAL_IDS: dict[str, int] = {name: index for index, name in enumerate(SPECIAL_TOKENS)}


class SpecialsError(ValueError):
    """
    Специальные токены прочитать нельзя.
    """


def build_special_tokens() -> dict[str, int]:
    """
    Специальные токены и их ID.
    """

    return dict(SPECIAL_IDS)


def load_special_tokens(directory: Path | None = None) -> dict[str, int]:
    """
    Специальные токены первого этапа.
    """

    path = (Path(directory) / SPECIAL_TOKENS_FILE) if directory else vocab_path(SPECIAL_TOKENS_FILE)

    if not path.exists():
        raise SpecialsError(
            f"нет {path}: выполните python -m src.tokenization.run special-tokens"
        )

    stored = read_json(path)

    # Сравнивается соответствие, а не порядок строк в файле:
    # порядок в JSON решает запись, а смысл задают номера.
    if stored != SPECIAL_IDS:
        raise SpecialsError(
            f"{path} описывает другой набор специальных токенов, чем код: "
            "выполните special-tokens заново и пересоберите словарь"
        )

    return stored


__all__ = [
    "EVT",
    "MASK",
    "PAD",
    "SPECIAL_IDS",
    "SPECIAL_TOKENS",
    "UNK",
    "USR",
    "SpecialsError",
    "build_special_tokens",
    "load_special_tokens",
]
