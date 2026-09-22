from __future__ import annotations

from pathlib import Path

from src.preprocessing.artifacts import read_json
from src.preprocessing.keys import CATEGORICAL

from .schema import SemanticSchema
from .settings import KEY_VOCAB_FILE, TokenizerConfig, tokenizer_path
from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


# ============================================================
# ЭТАП 2: КЛЮЧИ
# ============================================================
#
# Ключ отвечает на вопрос «что это», значение — «чему равно».
# Список ключей известен ДО чтения данных: его объявляет
# смысловой реестр, собранный из каталога полей payload.
# Поэтому этап 1 не читает ни одного клиента и не может ничему
# научиться у test: он только называет поля и выдаёт им номера.
#
# Номера идут сразу за специальными токенами и назначаются в
# порядке имени ключа. Порядок имени выбран потому, что он не
# зависит ни от частот, ни от порядка чтения: тот же реестр
# даёт те же номера.
#
# Ссылки на сущности перечислены рядом, но кода не получают:
# это связь, а не значение, и в embedding она не входит.
# ============================================================


class KeyVocabError(ValueError):
    """
    Словарь ключей собрать или прочитать нельзя.
    """


def domains_of(config: TokenizerConfig, schema: SemanticSchema) -> tuple[dict[str, str], dict[str, dict]]:
    """
    Ключ -> имя домена и описание каждого домена.

    По умолчанию домен у ключа свой: одинаковое написание ничего
    не доказывает, и active у карты не то же самое, что active у
    обращения. Объединение делается только явным списком с
    причиной.
    """

    declared: dict[str, dict] = {}
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

        declared[domain.name] = {
            "name": domain.name,
            "keys": list(domain.keys),
            "shared": True,
            "reason": domain.reason,
        }

    for key in sorted(schema.categorical_keys):

        if key in domain_of:
            continue

        domain_of[key] = key
        declared[key] = {
            "name": key,
            "keys": [key],
            "shared": False,
            "reason": "домен по умолчанию: значения ключа ни с кем не делятся",
        }

    return domain_of, declared


def build_key_vocab(specials: dict, config: TokenizerConfig, schema: SemanticSchema) -> dict:
    """
    Все поля, поступающие в модель, и их ID.

    Номера продолжают пространство сразу за специальными
    токенами: сколько их, говорит первый этап, а не константа
    здесь.
    """

    first_key_id = int(specials["next_id"])

    domain_of, _domains = domains_of(config, schema)

    keys = sorted(key for key, info in schema.keys.items() if info.is_model_feature)

    rows: list[dict] = []

    for index, key in enumerate(keys):

        info = schema.info(key)

        rows.append(
            {
                "key": key,
                "id": first_key_id + index,
                "value_kind": info.value_kind,
                "unit": info.unit,
                "origin": info.origin,
                "weight_rule": info.weight_rule,
                "domain": domain_of.get(key),
                "physical_fields": list(info.physical_fields),
                "description": info.description,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "format_version": FORMAT_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "keys_version": schema.keys_version,
        "projection_version": schema.projection_version,
        "first_key_id": first_key_id,
        "size": len(rows),
        "next_id": first_key_id + len(rows),
        "rule": (
            "ключи идут сразу за специальными токенами в порядке имени; "
            "состав объявлен смысловым реестром, а не наблюдениями"
        ),
        "specials": int(specials["size"]),
        "keys": rows,
        # Ссылки названы намеренно: кода у них нет, но потребитель
        # обязан знать, что они существуют и приходят метаданными.
        "link_keys": sorted(schema.link_keys),
        "declared_by_event_type": {
            event_type: list(item) for event_type, item in schema.declared_payload.items()
        },
    }


def load_key_vocab(directory: Path | None = None) -> dict:
    """
    Словарь ключей предыдущего этапа.
    """

    path = (Path(directory) / KEY_VOCAB_FILE) if directory else tokenizer_path(KEY_VOCAB_FILE)

    if not path.exists():
        raise KeyVocabError(
            f"нет {path}: выполните python -m src.tokenization.run key-vocab"
        )

    return read_json(path)


__all__ = [
    "KeyVocabError",
    "build_key_vocab",
    "domains_of",
    "load_key_vocab",
]
