from __future__ import annotations

from pathlib import Path

from src.preprocessing.artifacts import read_json

from .fit import TrainCorpus
from .keyvocab import domains_of, next_id
from .scan import TYPE_BOOL, TYPE_FLOAT, TYPE_INT, TYPE_ORDER
from .schema import SemanticSchema
from .settings import VALUE_VOCAB_FILE, TokenizerConfig, vocab_path


# ============================================================
# ЭТАП 3: КАТЕГОРИАЛЬНЫЕ ЗНАЧЕНИЯ
# ============================================================
#
# Файл отвечает на один вопрос:
#
#   ключ + значение -> ID
#
# Именно пара, а не значение само по себе: active у карты и
# active у обращения это разные факты, и общий номер склеил бы
# их. Поэтому значения сгруппированы по ключу, а плоской таблицы
# «значение -> номер» здесь нет.
#
# Ключи, объединённые в общий домен явным списком с причиной,
# делят одни и те же номера: одинаковое значение у них и правда
# одно и то же.
#
# Ключ без наблюдений на train остаётся в файле с пустым
# набором: «категория, которой мы не видели» и «поле не
# категория» это разные вещи.
#
# Чисел и кусков текста здесь нет. Частот тоже: они нужны при
# построении и в готовом словаре ничего не кодируют.
#
# Значения, которых на train не было, номера не получают
# никогда: на val и test они кодируются как [UNK].
# ============================================================


class ValuesError(ValueError):
    """
    Каталог значений собрать или прочитать нельзя.
    """


def typed_value(value_type: str, text: str) -> object:
    """
    Значение в своём типе: по нему и сортируется домен.
    """

    if value_type == TYPE_BOOL:
        return text == "true"

    if value_type == TYPE_INT:
        return int(text)

    if value_type == TYPE_FLOAT:
        return float(text)

    return text


def sort_key(value_type: str, text: str) -> tuple:
    """
    Порядок значений внутри домена: сначала тип, потом само
    значение.

    Числа сортируются как числа: иначе 10 оказалось бы между 1 и
    2, и соседние ID достались бы далёким значениям.
    """

    return (TYPE_ORDER[value_type], typed_value(value_type, text))


def build_value_vocab(
    train: TrainCorpus,
    key_vocab: dict[str, int],
    config: TokenizerConfig,
    schema: SemanticSchema,
) -> dict[str, dict[str, int]]:
    """
    Категориальные значения train по ключам: ключ -> значение -> ID.
    """

    domain_of = domains_of(config, schema)

    # --- что встретилось на train ---

    per_domain: dict[str, dict[tuple[str, str], int]] = {}

    for (key, value_type, value), entry in sorted(train.statistics.categorical.items()):

        domain = domain_of.get(key)

        if domain is None:
            raise ValuesError(
                f"ключ {key} встретился в статистике, но категорией не объявлен: "
                "статистика и реестр разошлись"
            )

        slot = per_domain.setdefault(domain, {})

        item = (value_type, value)

        # Частота нужна здесь и только здесь: в файл она не едет.
        slot[item] = slot.get(item, 0) + entry.count

    # --- номера ---

    first_id = next_id(key_vocab)

    ids_of_domain: dict[str, dict[str, int]] = {}

    number = first_id

    for domain in sorted(set(domain_of.values()) | set(per_domain)):

        ordered = sorted(per_domain.get(domain, {}), key=lambda item: sort_key(*item))

        assigned: dict[str, int] = {}
        types: dict[str, str] = {}

        for value_type, text in ordered:

            # Значение в файле это запись, а не пара «тип, запись».
            # У одного ключа тип один — это проверяет fit; в общем
            # домене разные типы с одинаковой записью слились бы
            # молча, и это останавливает этап.
            if text in assigned:
                raise ValuesError(
                    f"домен {domain}: запись {text!r} встретилась и как {types[text]}, и как "
                    f"{value_type}: одним номером это разные значения не становятся"
                )

            assigned[text] = number
            types[text] = value_type
            number += 1

        ids_of_domain[domain] = assigned

    # --- по ключам ---

    return {
        key: dict(ids_of_domain.get(domain_of[key], {}))
        for key in sorted(schema.categorical_keys)
    }


def load_value_vocab(directory: Path | None = None) -> dict[str, dict[str, int]]:
    """
    Каталог значений предыдущего этапа.
    """

    path = (Path(directory) / VALUE_VOCAB_FILE) if directory else vocab_path(VALUE_VOCAB_FILE)

    if not path.exists():
        raise ValuesError(f"нет {path}: выполните python -m src.tokenization.run value-vocab")

    return read_json(path)


__all__ = [
    "ValuesError",
    "build_value_vocab",
    "load_value_vocab",
    "sort_key",
    "typed_value",
]
