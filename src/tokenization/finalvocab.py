from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.preprocessing.artifacts import read_json

from .categorical import load_value_vocab
from .keyvocab import load_key_vocab
from .numeric import Bucket, FOUND_BUCKET, load_buckets, locate, read_buckets
from .settings import BPE_FILE, FINAL_VOCAB_FILE, VOCAB_DIR, vocab_path
from .specials import SPECIAL_TOKENS, load_special_tokens
from .text import BpeModel, load_bpe


# ============================================================
# ЭТАП 6: ФИНАЛЬНЫЙ СЛОВАРЬ
# ============================================================
#
# Все виды токенов живут в одном пространстве ID, и финальный
# словарь отвечает на один вопрос:
#
#   уникальное имя токена -> глобальный ID
#
# Имя несёт префикс вида, потому что одинаковая строка у разных
# видов это разные токены: ключ `merchant_city`, категория
# `merchant_city=Almaty` и кусок текста `Almaty` не должны
# сталкиваться в одной таблице.
#
#   [PAD]                    специальные — без префикса
#   key:<ключ>               поле события или анкеты
#   value:<ключ>=<значение>  категория, всегда в паре с ключом
#   bucket:<имя диапазона>   числовой диапазон
#   bpe:<кусок>              кусок текста
#
# Содержимого остальных файлов здесь нет: границы диапазонов
# лежат в buckets.json, модель разбиения в bpe.json, и копии
# рядом им не нужны. Номера назначают сами этапы, а финальный
# словарь только складывает их вместе и проверяет, что ни один
# не повторился и ни один не потерялся.
# ============================================================


KEY_PREFIX = "key:"
VALUE_PREFIX = "value:"
BUCKET_PREFIX = "bucket:"
BPE_PREFIX = "bpe:"


class VocabError(ValueError):
    """
    Финальный словарь собрать или прочитать нельзя.
    """


def key_token(key: str) -> str:
    return f"{KEY_PREFIX}{key}"


def value_token(key: str, value: str) -> str:
    return f"{VALUE_PREFIX}{key}={value}"


def bucket_token(name: str) -> str:
    return f"{BUCKET_PREFIX}{name}"


def bpe_token(piece: str) -> str:
    return f"{BPE_PREFIX}{piece}"


def build_final_vocab(
    specials: dict[str, int],
    key_vocab: dict[str, int],
    value_vocab: dict[str, dict[str, int]],
    buckets: dict[str, dict[str, dict]],
    bpe: BpeModel,
) -> dict[str, int]:
    """
    Один словарь из пяти видов токенов.
    """

    out: dict[str, int] = {}

    for name, token_id in specials.items():
        _put(out, name, token_id)

    for key, token_id in key_vocab.items():
        _put(out, key_token(key), token_id)

    for key, values in value_vocab.items():
        for value, token_id in values.items():
            _put(out, value_token(key, value), token_id)

    for key, entries in buckets.items():
        for name, item in entries.items():
            _put(out, bucket_token(name), int(item["id"]))

    # Куски текста продолжают пространство последними: их номера
    # это сдвиг плюс номер внутри модели разбиения.
    offset = max(out.values()) + 1 if out else 0

    for piece, local in sorted(bpe.vocab().items(), key=lambda pair: pair[1]):
        _put(out, bpe_token(piece), offset + int(local))

    _check_space(out)

    return dict(sorted(out.items(), key=lambda pair: pair[1]))


def _put(out: dict[str, int], name: str, token_id: int) -> None:

    if name in out:
        raise VocabError(f"токен {name} объявлен дважды: имена обязаны быть уникальными")

    out[name] = token_id


def _check_space(vocab: dict[str, int]) -> None:
    """
    Номера уникальны, идут подряд и начинаются со специальных.

    Дыра в пространстве это мёртвая строка embedding, а
    повторившийся номер — два разных токена под одним ID.
    """

    ids = sorted(vocab.values())

    if len(set(ids)) != len(ids):
        repeated = sorted({value for value in ids if ids.count(value) > 1})
        raise VocabError(f"номера повторяются: {repeated[:5]}")

    if ids != list(range(len(ids))):
        missing = sorted(set(range(ids[-1] + 1)) - set(ids)) if ids else []
        raise VocabError(
            f"в пространстве ID дыра: не выдан {missing[:5]}. Этапы собраны не одной цепочкой"
        )

    for index, name in enumerate(SPECIAL_TOKENS):
        if vocab.get(name) != index:
            raise VocabError(
                f"специальный токен {name} обязан занимать номер {index}, а занимает "
                f"{vocab.get(name)}"
            )


def load_final_vocab(directory: Path | None = None) -> dict[str, int]:
    """
    Финальный словарь предыдущего этапа.
    """

    path = (Path(directory) / FINAL_VOCAB_FILE) if directory else vocab_path(FINAL_VOCAB_FILE)

    if not path.exists():
        raise VocabError(f"нет {path}: выполните python -m src.tokenization.run final-vocab")

    return read_json(path)


# ------------------------------------------------------------
# ЗАМОРОЖЕННЫЙ СЛОВАРЬ
# ------------------------------------------------------------


@dataclass
class FrozenArtifacts:
    """
    Файлы словаря из data/03_vocab, которыми кодируют и расшифровывают.

    Ничего не обучает и не меняет: каждый файл отвечает за свой
    вопрос, а финальный словарь переводит имя токена в номер.
    """

    directory: Path
    vocab: dict[str, int]
    keys: dict[str, int]
    values: dict[str, dict[str, int]]
    buckets: dict[str, tuple[Bucket, ...]]
    bpe: BpeModel
    bpe_ids: tuple[int, ...]
    name_of: dict[int, str]

    @property
    def size(self) -> int:
        return len(self.vocab)

    def special(self, name: str) -> int:

        token_id = self.vocab.get(name)

        if token_id is None:
            raise VocabError(f"специального токена {name} нет в словаре")

        return token_id

    def key_id(self, key: str) -> int | None:
        return self.keys.get(key)

    def kind(self, key: str) -> str:
        """
        Чем кодируется значение ключа: категорией, числом или
        текстом. Вид виден по тому, в каком файле лежит ключ.
        """

        if key in self.values:
            return "categorical"

        if key in self.buckets:
            return "numeric"

        return "text"

    def categorical_id(self, key: str, value: str) -> int | None:
        return self.values.get(key, {}).get(value)

    def bucket_id(self, key: str, value: float) -> int | None:
        """
        Номер диапазона, в который попало число, или None, если
        шкалы у ключа нет.
        """

        found, bucket = locate(self.buckets.get(key, ()), value)

        return bucket.token_id if found == FOUND_BUCKET and bucket is not None else None

    def piece_id(self, local: int) -> int:
        """
        Номер куска текста в общем пространстве.
        """

        if local < 0 or local >= len(self.bpe_ids):
            raise VocabError(f"кусок {local} вне модели разбиения")

        return self.bpe_ids[local]

    def describe(self, token_id: int) -> str:
        """
        Имя токена по его номеру.
        """

        name = self.name_of.get(token_id)

        if name is None:
            raise VocabError(f"номера {token_id} нет в словаре")

        return name

    def decode_text(self, ids: list[int]) -> str:
        """
        Собранный обратно текст по номерам его кусков.
        """

        local = [self.bpe_ids.index(token_id) for token_id in ids]

        return self.bpe.decode(local)

    @staticmethod
    def load(directory: Path | None = None) -> "FrozenArtifacts":
        """
        Словарь целиком: шесть файлов из data/03_vocab.
        """

        directory = Path(directory) if directory is not None else VOCAB_DIR

        vocab = load_final_vocab(directory)
        specials = load_special_tokens(directory)
        keys = load_key_vocab(directory)
        values = load_value_vocab(directory)
        buckets = read_buckets(load_buckets(directory))
        bpe = load_bpe(directory / BPE_FILE)

        _check_agreement(vocab, specials, keys, values, buckets)

        pieces = bpe.vocab()

        bpe_ids = [0] * len(pieces)

        for piece, local in pieces.items():

            token_id = vocab.get(bpe_token(piece))

            if token_id is None:
                raise VocabError(
                    f"кусок {piece!r} есть в bpe.json, но не в финальном словаре: "
                    "выполните final-vocab заново"
                )

            bpe_ids[int(local)] = token_id

        return FrozenArtifacts(
            directory=directory,
            vocab=vocab,
            keys={key: vocab[key_token(key)] for key in keys},
            values=values,
            buckets=buckets,
            bpe=bpe,
            bpe_ids=tuple(bpe_ids),
            name_of={token_id: name for name, token_id in vocab.items()},
        )


def _check_agreement(
    vocab: dict[str, int],
    specials: dict[str, int],
    keys: dict[str, int],
    values: dict[str, dict[str, int]],
    buckets: dict[str, tuple[Bucket, ...]],
) -> None:
    """
    Номера в файлах этапов и в финальном словаре обязаны
    совпадать.

    Иначе кодирование пошло бы по одному словарю, а расшифровка
    по другому, и разошлись бы они молча.
    """

    _check_space(vocab)

    problems: list[str] = []

    for name, token_id in specials.items():
        if vocab.get(name) != token_id:
            problems.append(f"{name}: {token_id} против {vocab.get(name)}")

    for key, token_id in keys.items():
        if vocab.get(key_token(key)) != token_id:
            problems.append(f"{key_token(key)}: {token_id} против {vocab.get(key_token(key))}")

    for key, items in values.items():
        for value, token_id in items.items():
            name = value_token(key, value)
            if vocab.get(name) != token_id:
                problems.append(f"{name}: {token_id} против {vocab.get(name)}")

    for key, items in buckets.items():
        for bucket in items:
            name = bucket_token(bucket.name)
            if vocab.get(name) != bucket.token_id:
                problems.append(f"{name}: {bucket.token_id} против {vocab.get(name)}")

    if problems:
        raise VocabError(
            "файлы словаря собраны не одной цепочкой (" + "; ".join(problems[:5])
            + "). Выполните python -m src.tokenization.run fit"
        )


__all__ = [
    "BPE_PREFIX",
    "BUCKET_PREFIX",
    "KEY_PREFIX",
    "VALUE_PREFIX",
    "FrozenArtifacts",
    "VocabError",
    "bpe_token",
    "bucket_token",
    "build_final_vocab",
    "key_token",
    "load_final_vocab",
    "value_token",
]
