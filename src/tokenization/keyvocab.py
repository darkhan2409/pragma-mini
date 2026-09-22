from __future__ import annotations

from pathlib import Path

from src.preprocessing.artifacts import read_json
from src.preprocessing.keys import CATEGORICAL

from .schema import SemanticSchema
from .settings import KEY_VOCAB_FILE, TokenizerConfig, vocab_path


# ============================================================
# ЭТАП 2: КЛЮЧИ
# ============================================================
#
# Ключ отвечает на вопрос «что это», значение — «чему равно».
# Список ключей известен ДО чтения данных: его объявляет
# смысловой реестр, собранный из каталога полей payload.
# Поэтому этап не читает ни одного клиента и не может ничему
# научиться у test: он только называет поля и выдаёт им номера.
#
# Файл решает ровно одну задачу:
#
#   название ключа -> ID
#
# Ни вида значения, ни единицы, ни происхождения в нём нет: всё
# это живёт в реестре смыслов, и второй копии рядом с данными
# быть не должно.
#
# Номера идут сразу за специальными токенами в порядке имени
# ключа. Порядок имени выбран потому, что он не зависит ни от
# частот, ни от порядка чтения: тот же реестр даёт те же номера.
# ============================================================


class KeyVocabError(ValueError):
    """
    Словарь ключей собрать или прочитать нельзя.
    """


def domains_of(config: TokenizerConfig, schema: SemanticSchema) -> dict[str, str]:
    """
    Ключ -> имя домена значений.

    По умолчанию домен у ключа свой: одинаковое написание ничего
    не доказывает, и active у карты не то же самое, что active у
    обращения. Объединение делается только явным списком с
    причиной, и тогда ключи домена делят одни и те же номера
    значений.
    """

    domain_of: dict[str, str] = {}

    for domain in config.value_domains:

        for key in domain.keys:

            info = schema.info(key)

            if info.value_kind != CATEGORICAL:
                raise KeyVocabError(
                    f"домен {domain.name} объединяет {key}, а он {info.value_kind}: "
                    "общий домен бывает только у категорий"
                )

            domain_of[key] = domain.name

    for key in sorted(schema.categorical_keys):
        domain_of.setdefault(key, key)

    return domain_of


def build_key_vocab(specials: dict[str, int], schema: SemanticSchema) -> dict[str, int]:
    """
    Все поля, поступающие в модель, и их ID.

    Номера продолжают пространство сразу за специальными
    токенами: сколько их, говорит первый этап, а не константа
    здесь.
    """

    first_id = next_id(specials)

    keys = sorted(key for key, info in schema.keys.items() if info.is_model_feature)

    return {key: first_id + index for index, key in enumerate(keys)}


def next_id(mapping: dict) -> int:
    """
    Первый свободный номер после уже выданных.

    Считается по самим номерам, а не по отдельному полю рядом с
    ними: такое поле однажды разошлось бы с содержимым файла.
    """

    numbers = _ids(mapping)

    return max(numbers) + 1 if numbers else 0


def _ids(value) -> list[int]:
    """
    Все номера внутри вложенных словарей файла.
    """

    if isinstance(value, bool):
        return []

    if isinstance(value, int):
        return [value]

    if isinstance(value, dict):
        return [number for item in value.values() for number in _ids(item)]

    return []


def load_key_vocab(directory: Path | None = None) -> dict[str, int]:
    """
    Словарь ключей предыдущего этапа.
    """

    path = (Path(directory) / KEY_VOCAB_FILE) if directory else vocab_path(KEY_VOCAB_FILE)

    if not path.exists():
        raise KeyVocabError(f"нет {path}: выполните python -m src.tokenization.run key-vocab")

    return read_json(path)


__all__ = [
    "KeyVocabError",
    "build_key_vocab",
    "domains_of",
    "load_key_vocab",
    "next_id",
]
