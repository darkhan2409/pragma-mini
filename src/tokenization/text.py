from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from src.preprocessing.keys import TEXT

from .fit import TrainCorpus
from .scan import FitStatistics
from .settings import BPE_FILE, BpeConfig, TokenizerConfig, vocab_path


# ============================================================
# ЭТАП 5: РАЗБИЕНИЕ ТЕКСТА
# ============================================================
#
# BPE применяется только к тем полям, которые смысловой реестр
# объявил свободным текстом: сейчас это merchant_name и
# counterparty. Новых текстовых полей ради BPE не придумывается,
# а структурированный код текстом не становится.
#
# Алфавит байтовый и полный: 256 байт лежат в словаре с самого
# начала, поэтому невиданная казахская буква, эмодзи или
# китайский иероглиф кодируются и раскодируются без потерь и без
# переобучения.
#
# Нормализация не своя: берётся та же функция, что делает
# нормализованную копию текста в canonical. Двух правил
# нормализации у одного текста быть не должно.
#
# Файл bpe.json это СТАНДАРТНЫЙ файл библиотеки tokenizers,
# записанный её же Tokenizer.save: никакой своей обёртки, чтобы
# он читался обратно Tokenizer.from_file и не зависел от нашего
# формата. Настройки обучения живут в конфигурации, а не в
# выходном файле.
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
    Разбиение текста построить или прочитать нельзя.
    """


@dataclass
class BpeModel:
    """
    Обученное разбиение: тонкая обёртка над Tokenizer.
    """

    tokenizer: object | None = None

    @property
    def enabled(self) -> bool:
        return self.tokenizer is not None

    @property
    def size(self) -> int:
        return 0 if self.tokenizer is None else self.tokenizer.get_vocab_size()

    @property
    def merges(self) -> int:
        """
        Сколько слияний выучено сверх байтового алфавита.
        """

        return max(self.size - 256, 0)

    def vocab(self) -> dict[str, int]:
        """
        Кусок -> его номер внутри модели.
        """

        return {} if self.tokenizer is None else dict(self.tokenizer.get_vocab())

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

    def save(self, path: Path) -> None:

        if self.tokenizer is None:
            raise TextError("BPE выключен: сохранять нечего")

        path.parent.mkdir(parents=True, exist_ok=True)

        self.tokenizer.save(str(path), pretty=True)


def text_keys(key_vocab: dict[str, int], config: TokenizerConfig, schema) -> tuple[str, ...]:
    """
    Ключи, которые кодируются разбиением: объявленные текстом и
    не переведённые решением человека в категории.
    """

    return tuple(
        key
        for key in sorted(key_vocab)
        if schema.info(key).value_kind == TEXT and key not in config.text_keys_as_categorical
    )


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
        raise TextError(
            "разрешённых текстовых ключей нет: текст ради BPE не придумывается"
        )

    # Многопоточность тренера результат не меняет, но её
    # предупреждение в логе только мешает.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

    return BpeModel(tokenizer=model)


def build_bpe(train: TrainCorpus, key_vocab: dict[str, int], config: TokenizerConfig,
              schema) -> tuple[BpeModel, list[str]]:
    """
    Разбиение текста train и предупреждения для терминала.
    """

    keys = text_keys(key_vocab, config, schema)

    model = train_bpe(train.statistics, config.bpe, keys)

    texts = sorted({text for _key, text, _count in corpus_rows(train.statistics, keys)})

    broken = check_roundtrip(model, [*PROBES, *texts])

    if broken:
        raise TextError(
            "разбиение текста не обратимо на "
            + ", ".join(repr(item) for item in broken[:3])
            + ": байтовый алфавит обязан кодировать любой текст без потерь"
        )

    warnings: list[str] = []

    if not model.merges:
        warnings.append("BPE обучен на слишком маленьком корпусе: merges отсутствуют")

    return model, warnings


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


def load_bpe(path: Path | None = None) -> BpeModel:
    """
    Замороженное разбиение из стандартного bpe.json.
    """

    path = Path(path) if path is not None else vocab_path(BPE_FILE)

    if not path.exists():
        raise TextError(f"нет {path}: выполните python -m src.tokenization.run bpe")

    from tokenizers import Tokenizer

    return BpeModel(tokenizer=Tokenizer.from_file(str(path)))


__all__ = [
    "PROBES",
    "BpeModel",
    "TextError",
    "build_bpe",
    "check_roundtrip",
    "corpus_rows",
    "load_bpe",
    "text_keys",
    "train_bpe",
]
