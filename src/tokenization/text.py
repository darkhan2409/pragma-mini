from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from src.preprocessing.artifacts import read_json, sha256_bytes
from src.preprocessing.canonical.events import TEXT_NORMALIZATION
from src.preprocessing.keys import TEXT

from .fit import TrainCorpus
from .scan import FitStatistics
from .settings import BPE_FILE, BpeConfig, TokenizerConfig, tokenizer_path


# ============================================================
# ИДЕЯ
# ============================================================
#
# BPE применяется только к тем полям, которые смысловой реестр
# объявил свободным текстом. Новых текстовых полей ради BPE не
# придумывается, а структурированный код текстом не становится:
# если реестр в этом ошибается, этап 1 называет противоречие, а
# решение принимает человек.
#
# Алфавит байтовый и полный: 256 байт лежат в словаре с самого
# начала, поэтому невиданная казахская буква, эмодзи или
# китайский иероглиф кодируются и раскодируются без потерь и
# без переобучения.
#
# Нормализация не своя: берётся та же функция, что делает
# нормализованную копию текста в canonical. Двух правил
# нормализации у одного текста быть не должно.
#
# Пустая строка отличается от отсутствия значения: это отдельный
# служебный токен, а не пропуск.
# ============================================================


# Строки, на которых проверяется обратимость кодирования. Тут
# намеренно лежат пустая строка, пробельные символы, казахские и
# русские буквы, эмодзи и похожий на служебный токен текст: он
# обязан быть обычным текстом.
PROBES: tuple[str, ...] = (
    "",
    " ",
    "\t",
    "\n",
    "a  b",
    "Қазпошта",
    "Ёлка №1",
    "Europharma Алматы",
    "магазин 🌿 у дома",
    "[MASK]",
    "[PAD][UNK]",
    "北京市",
    "ÅÄÖ àéî",
    "x" * 300,
)


class TextError(ValueError):
    """
    Разбиение текста построить нельзя.
    """


@dataclass
class BpeModel:
    """
    Обученное разбиение вместе с тем, как оно получено.
    """

    enabled: bool
    keys: tuple[str, ...]
    info: dict
    tokenizer: object | None = None

    @property
    def size(self) -> int:
        return 0 if self.tokenizer is None else self.tokenizer.get_vocab_size()

    def pieces(self, text: str) -> list[int]:
        """
        Локальные номера кусков значения.
        """

        if self.tokenizer is None:
            raise TextError("BPE выключен: текстовых значений в этом словаре нет")

        return list(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int]) -> str:

        if self.tokenizer is None:
            raise TextError("BPE выключен: раскодировать нечего")

        return self.tokenizer.decode(list(ids))

    def piece(self, index: int) -> str:

        if self.tokenizer is None:
            raise TextError("BPE выключен")

        return self.tokenizer.id_to_token(index)


def corpus_rows(stats: FitStatistics, keys: tuple[str, ...]) -> list[tuple[str, str, int]]:
    """
    Нормализованные тексты разрешённых ключей с их частотой.

    Порядок задан данными: сначала ключ, потом сам текст.
    """

    allowed = set(keys)

    rows = [
        (key, text, entry.count)
        for key, bucket in stats.text.items()
        if key in allowed
        for text, entry in bucket.items()
    ]

    return sorted(rows)


def corpus_digest(rows: list[tuple[str, str, int]]) -> str:
    return sha256_bytes("\n".join(f"{key}\t{text}\t{count}" for key, text, count in rows).encode("utf-8"))


def _iterator(rows: list[tuple[str, str, int]]) -> Iterator[str]:
    """
    Частотное взвешивание: у тренера весов нет, поэтому текст
    повторяется столько раз, сколько он встретился.
    """

    for _key, text, count in rows:
        for _ in range(count):
            yield text


def train_bpe(stats: FitStatistics, config: BpeConfig, keys: tuple[str, ...]) -> BpeModel:
    """
    Обучает разбиение на разрешённых train-текстах.
    """

    if not keys:
        return BpeModel(
            enabled=False,
            keys=(),
            info={
                "enabled": False,
                "reason": "разрешённых текстовых ключей нет: текст ради BPE не придумывается",
            },
        )

    # Многопоточность тренера результат не меняет, но её
    # предупреждение в логе только мешает.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import tokenizers
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    rows = corpus_rows(stats, keys)

    model = Tokenizer(models.BPE(unk_token=None))

    model.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=config.add_prefix_space,
        use_regex=config.use_regex,
    )
    model.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
        show_progress=False,
    )

    model.train_from_iterator(_iterator(rows), trainer=trainer)

    texts = sorted({text for _key, text, _count in rows})

    lengths = sorted(len(model.encode(text, add_special_tokens=False).ids) for text in texts)

    info = {
        "enabled": True,
        "pieces_per_text": {
            "texts": len(lengths),
            "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0,
            "median": lengths[len(lengths) // 2] if lengths else 0,
            "max": lengths[-1] if lengths else 0,
            "note": (
                "на маленьком корпусе слияний мало, и текст остаётся почти побайтовым: "
                "это свойство корпуса, а не ошибка разбиения"
            ),
        },
        "keys": list(keys),
        "library": {"name": "tokenizers", "version": tokenizers.__version__},
        "model": "byte_level_bpe",
        "config": config.as_dict(),
        "normalization": dict(TEXT_NORMALIZATION),
        "corpus": {
            "values": len(rows),
            "texts": len(texts),
            "occurrences": sum(count for _key, _text, count in rows),
            "longest_bytes": max((len(text.encode("utf-8")) for text in texts), default=0),
            "sha256": corpus_digest(rows),
            "rule": "отсортированные нормализованные тексты, каждый повторён по числу вхождений",
        },
        "vocab_size": {"requested": config.vocab_size, "actual": model.get_vocab_size()},
        "trained_on_empty": not texts,
    }

    return BpeModel(enabled=True, keys=tuple(keys), info=info, tokenizer=model)


def check_roundtrip(model: BpeModel, texts: list[str]) -> list[str]:
    """
    Тексты, которые не переживают кодирование.

    Проверяется именно нормализованный текст: нормализация это
    решение препроцессинга, а не часть разбиения.
    """

    if not model.enabled:
        return []

    broken: list[str] = []

    for text in texts:
        if model.decode(model.pieces(text)) != text:
            broken.append(text)

    return broken


def dump_bpe(model: BpeModel) -> dict:
    """
    Разбиение в виде, пригодном для tokenizer.json.

    Отдельного файла разбиения больше нет: оно едет разделом
    внутри одного замороженного комплекта.
    """

    if model.tokenizer is None:
        raise TextError("BPE выключен: сохранять нечего")

    return json.loads(model.tokenizer.to_str())


def load_bpe(data: dict) -> BpeModel:
    """
    Замороженное разбиение из записи tokenizer.json.
    """

    from tokenizers import Tokenizer

    model = Tokenizer.from_str(json.dumps(data))

    return BpeModel(enabled=True, keys=(), info={"enabled": True}, tokenizer=model)


def build_bpe(train: TrainCorpus, key_vocab: dict, config: TokenizerConfig) -> dict:
    """
    Разбиение текста, обученное на train, со всем, что
    нужно для кодирования и раскодирования.

    Номеров в общем пространстве ID здесь нет: куски
    нумеруются внутри модели, а сдвиг назначает финальный
    словарь, когда известны все остальные виды токенов.
    """

    text_keys = tuple(
        row["key"]
        for row in key_vocab["keys"]
        if row["value_kind"] == TEXT and row["key"] not in config.text_keys_as_categorical
    )

    model = train_bpe(train.statistics, config.bpe, text_keys)

    if model.enabled:

        texts = sorted({text for _key, text, _count in corpus_rows(train.statistics, text_keys)})

        broken = check_roundtrip(model, [*PROBES, *texts])

        if broken:
            raise TextError(
                "разбиение текста не обратимо на "
                + ", ".join(repr(item) for item in broken[:3])
                + ": байтовый алфавит обязан кодировать любой текст без потерь"
            )

    return {
        **model.info,
        "fit": train.as_dict(),
        "config_sha256": config.sha256(),
        "size": model.size,
        "model": dump_bpe(model) if model.enabled else None,
    }


def load_bpe_file(directory: Path | None = None) -> dict:
    """
    Разбиение предыдущего этапа.
    """

    path = (Path(directory) / BPE_FILE) if directory else tokenizer_path(BPE_FILE)

    if not path.exists():
        raise TextError(f"нет {path}: выполните python -m src.tokenization.run bpe")

    return read_json(path)



__all__ = [
    "PROBES",
    "BpeModel",
    "TextError",
    "check_roundtrip",
    "corpus_digest",
    "build_bpe",
    "corpus_rows",
    "dump_bpe",
    "load_bpe",
    "load_bpe_file",
    "train_bpe",
]
