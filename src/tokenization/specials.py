from __future__ import annotations

from pathlib import Path

from src.preprocessing.artifacts import read_json

from .settings import SPECIAL_TOKENS_FILE, tokenizer_path
from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


# ============================================================
# СПЕЦИАЛЬНЫЕ ТОКЕНЫ
# ============================================================
#
# Они занимают начало пространства ID и не зависят ни от одного
# этапа обучения словаря: их состав это решение формата, а не
# наблюдение по данным. Поэтому ключи могут получать номера
# сразу после них, ещё до того, как прочитан хоть один клиент.
#
# Специальный токен в слоте значения означает не значение, а
# его отсутствие, неизвестность или невозможность. Это четыре
# разные вещи, и одинаково они не кодируются.
#
# Конец события и конец профиля своих токенов НЕ имеют
# и иметь не должны: границы записей в этом формате
# задаются массивами начал и длин, а не закрывающим
# токеном. Два источника одной границы разошлись бы.
# ============================================================


PAD = "[PAD]"
UNK = "[UNK]"
MASK = "[MASK]"
EVT = "[EVT]"
USR = "[USR]"
MISSING = "[MISSING]"
INVALID = "[INVALID]"
EMPTY = "[EMPTY]"

# Порядок первых шести повторяет прежний словарь намеренно: это
# ничего не стоит и снимает один источник путаницы при чтении
# старого кода рядом с новым.
SPECIAL_TOKENS: tuple[str, ...] = (PAD, UNK, MASK, EVT, USR, MISSING, INVALID, EMPTY)

SPECIAL_ROLES: dict[str, str] = {
    PAD: "выравнивание batch; токенизатор его не пишет никогда",
    UNK: "значение, которого на train не было, и число без шкалы",
    MASK: "зарезервирован для Masker; в сохранённых данных не встречается никогда",
    EVT: "начало события, один на событие, ставит токенизатор",
    USR: "начало представления профиля, один на профиль, ставит токенизатор",
    MISSING: "объявленный у этого типа события ключ без значения",
    INVALID: "невозможное число: NaN, бесконечность или запрещённый доменом знак",
    EMPTY: "текст, в котором после нормализации не осталось ни одного символа",
}

SPECIAL_IDS: dict[str, int] = {name: index for index, name in enumerate(SPECIAL_TOKENS)}

# Первый ID, свободный от специальных токенов: с него начинаются
# ключи.
FIRST_KEY_ID = len(SPECIAL_TOKENS)

KIND_CATEGORICAL = "categorical"
KIND_BUCKET = "bucket"


class SpecialsError(ValueError):
    """
    Специальные токены прочитать нельзя.
    """


def build_special_tokens() -> dict:
    """
    Специальные токены, их ID и назначение.

    Состав задан форматом, а не данными: читать здесь
    нечего и учиться нечему.
    """

    return {
        "schema_version": SCHEMA_VERSION,
        "format_version": FORMAT_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "first_id": 0,
        "size": len(SPECIAL_TOKENS),
        "next_id": len(SPECIAL_TOKENS),
        "rule": "специальные токены занимают начало пространства ID",
        "boundaries": (
            "конец события и конец профиля токенами не обозначаются: "
            "границы записей едут массивами начал и длин"
        ),
        "never_written": [PAD, MASK],
        "marker_owner": "tokenizer",
        "tokens": [
            {
                "id": SPECIAL_IDS[name],
                "token": name,
                "type": "special",
                "role": SPECIAL_ROLES[name],
                "written_by_tokenizer": name not in (PAD, MASK),
            }
            for name in SPECIAL_TOKENS
        ],
    }


def load_special_tokens(directory: Path | None = None) -> dict:
    """
    Специальные токены первого этапа.
    """

    path = (Path(directory) / SPECIAL_TOKENS_FILE) if directory else tokenizer_path(SPECIAL_TOKENS_FILE)

    if not path.exists():
        raise SpecialsError(
            f"нет {path}: выполните python -m src.tokenization.run special-tokens"
        )

    stored = read_json(path)

    if [row["token"] for row in stored["tokens"]] != list(SPECIAL_TOKENS):
        raise SpecialsError(
            f"{path} описывает другой набор специальных токенов, чем код: "
            "выполните special-tokens заново и пересоберите словарь"
        )

    return stored


__all__ = [
    "EMPTY",
    "EVT",
    "FIRST_KEY_ID",
    "INVALID",
    "KIND_BUCKET",
    "KIND_CATEGORICAL",
    "MASK",
    "MISSING",
    "PAD",
    "SPECIAL_IDS",
    "SPECIAL_ROLES",
    "SPECIAL_TOKENS",
    "UNK",
    "USR",
    "SpecialsError",
    "build_special_tokens",
    "load_special_tokens",
]
