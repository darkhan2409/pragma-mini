from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from src.preprocessing.artifacts import sha256_bytes, sha256_file
from src.preprocessing.canonical.events import TEXT_NORMALIZATION, normalize_text

from .contract import STATISTICS_DIR, TEXT_FILE
from .settings import BpeConfig


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


BPE_FILE = "bpe.json"

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


def corpus_rows(target: Path, keys: tuple[str, ...]) -> list[tuple[str, str, int]]:
    """
    Нормализованные тексты разрешённых ключей с их частотой.

    Порядок задан данными, а не файловой системой: сначала ключ,
    потом сам текст.
    """

    path = Path(target) / STATISTICS_DIR / TEXT_FILE

    if not path.exists():
        raise TextError(f"нет {path}: тексты собирает этап 1")

    allowed = set(keys)

    rows = [
        (row["key"], row["text_norm"], int(row["count"]))
        for row in pq.read_table(path).to_pylist()
        if row["key"] in allowed
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


def train_bpe(target: Path, config: BpeConfig, keys: tuple[str, ...]) -> BpeModel:
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

    rows = corpus_rows(target, keys)

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

    path = Path(target) / BPE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path), pretty=True)

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
        "file_sha256": sha256_file(path),
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


def load_bpe(path: Path) -> BpeModel:
    """
    Замороженное разбиение с диска.
    """

    from tokenizers import Tokenizer

    model = Tokenizer.from_file(str(path))

    return BpeModel(enabled=True, keys=(), info={"enabled": True}, tokenizer=model)


__all__ = [
    "BPE_FILE",
    "PROBES",
    "BpeModel",
    "TextError",
    "check_roundtrip",
    "corpus_digest",
    "corpus_rows",
    "load_bpe",
    "train_bpe",
]
