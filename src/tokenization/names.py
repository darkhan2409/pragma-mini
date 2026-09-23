from __future__ import annotations

from .finalvocab import (
    BPE_PREFIX,
    BUCKET_PREFIX,
    KEY_PREFIX,
    VALUE_PREFIX,
    FrozenArtifacts,
)


# ============================================================
# ИМЕНА ТОКЕНОВ
# ============================================================
#
# Номер в словаре -> то, что можно прочитать. Нужно везде, где
# результат показывают человеку: в отчётах модели и в разборе
# целей.
#
# Живёт рядом со словарём, а не в потребителе: расшифровка это
# свойство словаря, и знать о ней должен тот, кто его собрал.
# ============================================================


class Names:
    """
    Номер -> то, что можно прочитать.
    """

    def __init__(self, vocab: FrozenArtifacts):

        self.vocab = vocab

        # Куски текста лежат в общем словаре со сдвигом, а
        # раскодировать их умеет только сама модель разбиения, по
        # своим местным номерам.
        self.local_of = {
            token_id: local for local, token_id in enumerate(vocab.bpe_ids)
        }

    def raw(self, token_id: int) -> str:
        return self.vocab.describe(int(token_id))

    def kind(self, token_id: int) -> str:

        name = self.raw(token_id)

        for prefix, kind in (
            (KEY_PREFIX, "key"),
            (VALUE_PREFIX, "value"),
            (BUCKET_PREFIX, "bucket"),
            (BPE_PREFIX, "bpe"),
        ):
            if name.startswith(prefix):
                return kind

        return "special"

    def short(self, token_id: int) -> str:
        """
        Короткое читаемое имя одного номера.
        """

        name = self.raw(token_id)
        kind = self.kind(token_id)

        if kind == "key":
            return name[len(KEY_PREFIX):]

        if kind == "value":
            return name[len(VALUE_PREFIX):].split("=", 1)[-1]

        if kind == "bucket":
            return name[len(BUCKET_PREFIX):]

        if kind == "bpe":
            return self.text([token_id])

        return name

    def text(self, token_ids: list[int]) -> str:
        """
        Значение целиком: куски BPE собираются обратно в строку.

        Байтовый алфавит в словаре записан служебными символами
        (Ġ вместо пробела и так далее), поэтому читать их напрямую
        нельзя — собирает строку та же модель, что её разбивала.
        """

        locals_ = [self.local_of.get(int(token_id)) for token_id in token_ids]

        if locals_ and all(local is not None for local in locals_):
            return self.vocab.bpe.decode(locals_)

        return " ".join(self.short(token_id) for token_id in token_ids)
