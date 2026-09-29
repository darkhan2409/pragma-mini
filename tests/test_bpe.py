from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.tokenization.settings import BpeConfig, ConfigError
from src.tokenization.text import BYTES, PROBES, check_roundtrip, train_bpe


# ============================================================
# ИДЕЯ
# ============================================================
#
# Разбиение текста обязано быть обратимым при любом алфавите: и у
# байтового, и у символьного decode(pieces(x)) == x на любой строке,
# включая невиданные символы, пробельные последовательности и текст,
# похожий на служебный токен. Символьный алфавит — без подменённых
# символов в кусках (Ġ): пробел остаётся пробелом, а невиданный
# символ кодируется своими байтами, а не теряется.
# ============================================================


NAMES = {
    "merchant_name": {"Magnum Cash Carry": 40, "Small Coffee House": 25, "Кофе Хаус Алматы": 30,
                      "Coffee Point": 12},
    "counterparty": {"Айгерим К.": 9, "Own account": 20},
}


def stats() -> SimpleNamespace:
    return SimpleNamespace(text={
        key: {text: SimpleNamespace(count=count) for text, count in bucket.items()}
        for key, bucket in NAMES.items()
    })


@pytest.mark.parametrize("alphabet", ["bytes", "characters"])
def test_both_alphabets_are_lossless(alphabet: str):

    model = train_bpe(stats(), BpeConfig(vocab_size=600, alphabet=alphabet), tuple(NAMES))

    extra = ["Coffee Магнум", "Кофе 🌿 北京", "Қазпошта", "<0x41> literal", "a b"]

    assert check_roundtrip(model, [*PROBES, *extra, *NAMES["merchant_name"]]) == []


def test_character_pieces_keep_plain_text_and_fall_back_to_bytes():

    model = train_bpe(stats(), BpeConfig(vocab_size=600, alphabet="characters"), tuple(NAMES))

    vocab = model.vocab()

    assert all(name in vocab for name in BYTES)
    assert not any("Ġ" in piece for piece in vocab)

    known = [model.piece(index) for index in model.pieces("Small Coffee Point")]
    assert "".join(known) == "Small Coffee Point"
    assert " Coffee" in known, "пробел — обычный символ в начале куска"

    # Невиданная буква — своими байтами, а знакомые рядом — кусками.
    unseen = [model.piece(index) for index in model.pieces("Қофе")]
    assert unseen[:2] == ["<0xD2>", "<0x9A>"]
    assert "<0x41>" not in [model.piece(index) for index in model.pieces("<0x41>")]


def test_characters_alphabet_is_the_default_and_unknown_alphabets_are_refused():

    assert BpeConfig().alphabet == "characters"
    assert BpeConfig.from_dict({"alphabet": "bytes"}).alphabet == "bytes"

    with pytest.raises(ConfigError, match="alphabet"):
        BpeConfig.from_dict({"alphabet": "words"})
