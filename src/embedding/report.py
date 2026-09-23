from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

import torch

from src.tokenization.finalvocab import (
    BPE_PREFIX,
    BUCKET_PREFIX,
    KEY_PREFIX,
    VALUE_PREFIX,
    FrozenArtifacts,
)

from .inputs import BATCH_COLUMNS, Checks, MASKED_COLUMNS
from .layer import InputEmbedding
from .select import Shown, ShownValue
from .settings import EmbeddingConfig


# ============================================================
# ВИДИМЫЙ РЕЗУЛЬТАТ
# ============================================================
#
# Отчёт читает человек, который проверяет вход эмбеддингов.
# Поэтому на главном экране одно: настоящая пара ключ-значение,
# номер куска и получившийся вектор. Всё, что нужно раз в жизни —
# пути, версии, формы, сверки — сложено в один закрытый блок.
#
# Ни одного придуманного числа здесь нет: показанные векторы
# посчитаны тем же слоем на тех же строках batches.parquet и
# masked.parquet.
#
# HTML собирается строками. Шаблонизатора в зависимостях проекта
# нет, и заводить его ради одной страницы незачем.
# ============================================================


# Сколько чисел вектора показать на главном экране. Полностью он
# лежит в закрытом блоке: 128 чисел в строке не читает никто.
PREVIEW_NUMBERS = 5


@dataclass(frozen=True)
class Shapes:
    """
    Форма выхода слоя, посчитанная по форме входа.

    Сам батч целиком через слой не прогоняется: отчёту нужны
    показанные позиции, а не гигабайт векторов.
    """

    clients: int
    width: int
    profile_width: int
    dim: int


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


class Page:
    """
    Сборка страницы: вся разметка знает про срез слоя.
    """

    def __init__(
        self,
        names: Names,
        shown: Shown,
        vectors: torch.Tensor,
        places: list[int],
        profile_vectors: torch.Tensor,
        profile_places: list[int],
        key_ids,
        visible,
        source,
        positions,
        profile_key_ids,
        profile_value_ids,
        profile_positions,
    ):
        self.names = names
        self.shown = shown

        self.vectors = vectors
        self.where = {place: number for number, place in enumerate(places)}

        self.profile_vectors = profile_vectors
        self.profile_where = {
            place: number for number, place in enumerate(profile_places)
        }

        self.key_ids = key_ids
        self.visible = visible
        self.source = source
        self.positions = positions

        self.profile_key_ids = profile_key_ids
        self.profile_value_ids = profile_value_ids
        self.profile_positions = profile_positions

    # --- доступ к посчитанным векторам ---

    def vector(self, place: int) -> torch.Tensor:
        return self.vectors[0, self.where[place]]

    def profile_vector(self, place: int) -> torch.Tensor:
        return self.profile_vectors[0, self.profile_where[place]]

    # --- строки ---

    def value_rows(self, values: tuple[ShownValue, ...]) -> str:
        return "".join(self.value_row(value) for value in values)

    def value_row(self, value: ShownValue) -> str:
        """
        Одно значение. Куски BPE раскрываются внутри него.
        """

        key = html.escape(self.names.short(value.key_id))

        seen = html.escape(
            self.names.text([int(self.visible[place]) for place in value.places])
        )

        # Что именно спрятано, видно по самому значению: там
        # написано [MASK] или [UNK]. Значок рядом лишь повторил бы
        # его, а пользы больше от исходного значения — по нему
        # видно, что модель обязана восстановить.
        was = ""

        if self._changed(value):
            was = (
                ' <span class="was">исходно '
                f'{html.escape(self.names.text([int(self.source[place]) for place in value.places]))}'
                "</span>"
            )

        if value.length == 1:
            return _row(key, seen + was, int(self.positions[value.start]),
                        self.vector(value.start))

        inside = "".join(
            _row(
                f"кусок {index}",
                html.escape(self.names.short(int(self.visible[place]))),
                int(self.positions[place]),
                self.vector(place),
            )
            for index, place in enumerate(value.places)
        )

        return f"""<details class="many"><summary>
<span class="k">{key}</span>
<span class="v">{seen}{was}</span>
<span class="p">0…{value.length - 1}</span>
<span class="n">{value.length} кусков</span>
<span class="hint">раскрыть куски BPE — у каждого свой position и свой вектор</span>
</summary><div class="inside">{inside}</div></details>"""

    def profile_rows(self) -> str:

        rows = []

        for place in self.shown.profile:

            key_id = int(self.profile_key_ids[place])
            value_id = int(self.profile_value_ids[place])

            name = self.names.raw(key_id)

            if name in ("[USR]", "[EVT]"):
                rows.append(
                    _row(
                        f'<span class="badge">{html.escape(name)}</span>',
                        "маркер: один номер в обоих слотах",
                        None,
                        self.profile_vector(place),
                    )
                )
                continue

            rows.append(
                _row(
                    html.escape(self.names.short(key_id)),
                    html.escape(self.names.short(value_id)),
                    int(self.profile_positions[place]),
                    self.profile_vector(place),
                )
            )

        return "".join(rows)

    def marker_row(self) -> str:

        place = self.shown.marker

        return _row(
            '<span class="badge">[EVT]</span>',
            "маркер события",
            None,
            self.vector(place),
        )

    def pad_row(self) -> str:

        place = self.shown.pad

        if place is None:
            return ""

        return _row(
            '<span class="badge pad">[PAD]</span>',
            "заполнитель",
            None,
            self.vector(place),
        )

    def _changed(self, value: ShownValue) -> bool:
        """
        Маскер тронул это значение.
        """

        return any(
            int(self.visible[place]) != int(self.source[place])
            for place in value.places
        )


def render(
    group: str,
    index: int,
    config: EmbeddingConfig,
    implementation: str,
    names: Names,
    checks: Checks,
    shapes: Shapes,
    shown: Shown,
    layer: InputEmbedding,
    page: Page,
    batches_path: Path,
    masked_path: Path,
    table_path: Path,
    weights_path: Path,
) -> str:
    """
    Страница целиком.
    """

    body = "\n".join(
        [
            _head(shown, config),
            _example(page, shown, names),
            _values(page, shown),
            _profile(page, shown),
            _markers(page),
            _technical(group, index, config, implementation, names, checks, shapes,
                       shown, layer, page, batches_path, masked_path, table_path,
                       weights_path),
        ]
    )

    return _frame(group, index, body)


def _frame(group: str, index: int, body: str) -> str:

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Вход эмбеддингов — {html.escape(group)}, батч {index}</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin: 0 auto; padding: 28px 24px 60px; max-width: 940px;
  font: 14px/1.55 -apple-system, Segoe UI, Roboto, sans-serif; color: #1f2328;
  background: #fff; }}
h1 {{ font-size: 21px; margin: 0 0 10px; }}
h2 {{ font-size: 15px; margin: 30px 0 8px; color: #59636e;
  text-transform: uppercase; letter-spacing: .04em; }}
p {{ margin: 8px 0; }}
.formula {{ font: 14px/1.6 Consolas, Menlo, monospace; background: #f6f8fa;
  border: 1px solid #d8dee4; border-radius: 6px; padding: 10px 14px;
  display: inline-block; }}
.who {{ color: #59636e; margin: 10px 0 0; }}
.card {{ border: 1px solid #d8dee4; border-radius: 10px; padding: 16px 18px;
  margin: 12px 0 4px; }}
.card table {{ border-collapse: collapse; }}
.card td {{ padding: 4px 16px 4px 0; vertical-align: baseline; }}
.card td:first-child {{ color: #59636e; white-space: nowrap; width: 150px; }}
.big {{ font-size: 16px; font-weight: 600; }}
.mono {{ font: 12.5px/1.6 Consolas, Menlo, monospace; }}
.head, .r, details.many > summary {{ display: grid;
  grid-template-columns: 190px minmax(110px, 1fr) 54px 74px 250px;
  gap: 12px; align-items: baseline; padding: 5px 8px; border-radius: 6px; }}
.head {{ color: #8c959f; font-size: 11.5px; text-transform: uppercase;
  letter-spacing: .04em; padding-bottom: 2px; }}
.r:nth-child(odd) {{ background: #f6f8fa; }}
.k {{ color: #1f2328; }}
.v {{ font-weight: 600; }}
.p, .n {{ font: 12px Consolas, Menlo, monospace; color: #59636e; }}
.num {{ font: 12px Consolas, Menlo, monospace; color: #59636e;
  white-space: nowrap; overflow: hidden; }}
details.many {{ margin: 0; }}
details.many > summary {{ cursor: pointer; list-style: none;
  background: #fff8f7; border: 1px solid #ffd7d5; }}
details.many > summary::-webkit-details-marker {{ display: none; }}
details.many[open] > summary {{ border-bottom-left-radius: 0;
  border-bottom-right-radius: 0; }}
details.many .hint {{ grid-column: 1 / -1; color: #8c959f; font-size: 12px; }}
.inside {{ border: 1px solid #ffd7d5; border-top: none; padding: 4px 0 6px 18px;
  margin-bottom: 4px; }}
.badge {{ display: inline-block; font: 11px Consolas, monospace; padding: 0 6px;
  border-radius: 10px; border: 1px solid #0969da; color: #0969da; }}
.badge.pad {{ border-color: #8c959f; color: #8c959f; }}
.was {{ color: #8c959f; font-weight: 400; font-size: 12px; }}
.zero {{ color: #1a7f37; }}
.note {{ background: #fff8c5; border: 1px solid #eac54f; border-radius: 6px;
  padding: 10px 12px; margin: 12px 0; }}
details.tech {{ margin: 36px 0 0; border-top: 1px solid #d8dee4;
  padding-top: 12px; }}
details.tech > summary {{ cursor: pointer; color: #59636e; }}
details.tech table {{ border-collapse: collapse; margin: 10px 0; }}
details.tech td {{ padding: 3px 16px 3px 0; vertical-align: top; }}
details.tech td:first-child {{ color: #59636e; white-space: nowrap; }}
details.tech ul {{ margin: 6px 0; padding-left: 20px; }}
details.tech li {{ margin: 3px 0; }}
code {{ font: 12px Consolas, Menlo, monospace; background: #f3f4f6;
  padding: 1px 5px; border-radius: 4px; }}
.full {{ font: 11px/1.7 Consolas, Menlo, monospace; background: #f6f8fa;
  padding: 8px 10px; border-radius: 6px; word-break: break-all; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def _head(shown: Shown, config: EmbeddingConfig) -> str:

    moment = "—" if shown.time is None else shown.time.isoformat(sep=" ")[:16]

    note = f'<div class="note">{html.escape(shown.note)}</div>' if shown.note else ""

    return f"""<h1>Вход эмбеддингов</h1>
<div class="formula">E[key_id] · scale &nbsp;+&nbsp; E[value_id] · scale
 &nbsp;+&nbsp; P[position] &nbsp;=&nbsp; вектор из {config.dim} чисел</div>
<p class="who">клиент {html.escape(shown.client_id)} · событие {shown.event}
 из {shown.n_events} · {html.escape(moment)} (время только для ориентира,
 в слой не подаётся)</p>
{note}"""


def _example(page: Page, shown: Shown, names: Names) -> str:
    """
    Главный экран: одна пара ключ-значение и её вектор.
    """

    value = shown.example

    key_id = int(page.key_ids[value.start])
    seen = int(page.visible[value.start])

    vector = page.vector(value.start)

    return f"""<h2>Один токен целиком</h2>
<div class="card">
<table>
<tr><td>ключ</td><td class="big">{html.escape(names.short(key_id))}</td></tr>
<tr><td>видит модель</td><td class="big">{html.escape(names.short(seen))}</td></tr>
<tr><td>position</td><td class="mono">{int(page.positions[value.start])}
 — номер куска внутри значения</td></tr>
<tr><td>вектор</td><td class="mono">{len(vector)} чисел,
 ‖v‖ {float(vector.norm()):.3f}<br>{_short(vector)}</td></tr>
</table>
</div>"""


def _values(page: Page, shown: Shown) -> str:

    return f"""<h2>Остальные значения этого события</h2>
{_header()}
{page.value_rows(tuple(value for value in shown.values if value is not shown.example))}"""


def _profile(page: Page, shown: Shown) -> str:

    return f"""<h2>Анкета — первые {len(shown.profile)} из {shown.profile_n_tokens}</h2>
{_header()}
{page.profile_rows()}"""


def _markers(page: Page) -> str:

    pad = page.pad_row()

    tail = ""

    if pad:
        tail = f"""<h2>Заполнитель</h2>
{pad}
<p>[PAD] стоит в обоих слотах и по содержимому неотличим от маркера события.
Отличает их только маска — она же зануляет вектор.</p>"""

    return f"""<h2>Маркер события</h2>
{page.marker_row()}
<p>[EVT] записан одним номером в оба слота, поэтому его эмбеддинг берётся
<b>один раз</b>: без удвоения и без позиции куска.</p>
{tail}"""


def _technical(group, index, config, implementation, names, checks, shapes, shown,
               layer, page, batches_path, masked_path, table_path,
               weights_path) -> str:

    with torch.no_grad():
        single = layer.table(
            torch.tensor([int(page.key_ids[shown.marker])], dtype=torch.int64)
        )[0] * layer.scale

    marker = page.vector(shown.marker)

    pad_zeros = "—"

    if shown.pad is not None:
        pad_zeros = f"{int((page.vector(shown.pad) != 0).sum())} из {config.dim}"

    checked = [
        f"batch_index совпал в обоих файлах на всех {checks.clients} строках",
        f"client_id совпал построчно и в том же порядке: {checks.clients} клиентов",
        f"value_ids_source поэлементно равен value_ids батча: {checks.compared_values} позиций",
        f"у маркеров [EVT] и [USR] один номер в обоих слотах: {checks.markers} позиций",
        f"[PAD] лежит ровно там, где маска False: {checks.pad_slots} мест событий "
        f"и {checks.profile_pad_slots} мест анкеты",
    ]

    items = "\n".join(f"<li>{line}</li>" for line in checked)

    return f"""<details class="tech">
<summary>Технические сведения</summary>

<table>
<tr><td>группа, батч</td><td>{html.escape(group)}, {index}</td></tr>
<tr><td>d</td><td>{config.dim}</td></tr>
<tr><td>seed</td><td>{config.seed}</td></tr>
<tr><td>scale</td><td>sqrt(d) = {config.dim ** 0.5:.4f}</td></tr>
<tr><td>словарь</td><td>{names.vocab.size} ID, одна общая таблица на ключи и значения</td></tr>
<tr><td>версия этапа</td><td>{html.escape(implementation)}</td></tr>
<tr><td>форма выхода</td><td><code>[{shapes.clients}, {shapes.width}, {shapes.dim}]</code>
 события, <code>[{shapes.clients}, {shapes.profile_width}, {shapes.dim}]</code> анкета
 — посчитана по форме входа; слой применён только к показанным позициям</td></tr>
<tr><td>ключи, позиции, анкета, маски</td><td><code>{html.escape(str(batches_path))}</code></td></tr>
<tr><td>видимые значения</td><td><code>{html.escape(str(masked_path))}</code></td></tr>
<tr><td>векторы всей группы</td><td><code>{html.escape(str(table_path))}</code>
 — эта же работа, но по всем батчам; страница показывает один</td></tr>
<tr><td>веса слоя</td><td><code>{html.escape(str(weights_path))}</code>
 — ими посчитан и выход, и эта страница</td></tr>
</table>

<p>Векторы в parquet посчитаны <b>начальным</b> розыгрышем весов. При обучении
веса меняются на каждом шаге, и модель считает эмбеддинги сама, в прямом
проходе: файл это снимок входа, а не замена forward.</p>

<p><b>Сверка двух входных файлов.</b> Файлы собираются разными командами и могут
разъехаться; любое расхождение останавливает этап.</p>
<ul>
{items}
</ul>

<p>Из масок прочитаны ровно колонки <code>{html.escape(", ".join(MASKED_COLUMNS))}</code>
— <b>labels не читались вовсе</b>. Из батчей прочитаны
<code>{html.escape(", ".join(BATCH_COLUMNS))}</code>: <b>calendar и event_time_log
в списке отсутствуют</b>, они понадобятся энкодеру события и TimeRoPE, а не входу.</p>

<table>
<tr><td>маркер против E[маркер]·scale</td>
<td class="zero">расхождение {float((marker - single).abs().max()):.2e}</td></tr>
<tr><td>маркер против 2·E[маркер]·scale</td>
<td>расхождение {float((marker - 2.0 * single).abs().max()):.4f} — удвоения нет</td></tr>
<tr><td>ненулевых компонент у показанного [PAD]</td><td class="zero">{pad_zeros}</td></tr>
<tr><td>показано позиций</td><td>{len(page.where)} событий и {len(page.profile_where)} анкеты
 — столько же и посчитано</td></tr>
</table>

<p><b>Полный вектор токена с главного экрана</b>, {config.dim} чисел:</p>
<div class="full">{_full(page.vector(shown.example.start))}</div>

</details>"""


def _header() -> str:

    return """<div class="head"><span>ключ</span><span>видит модель</span>
<span>pos</span><span>‖v‖</span><span>первые числа вектора</span></div>"""


def _row(key: str, value: str, position: int | None, vector: torch.Tensor) -> str:

    norm = float(vector.norm())

    numbers = _short(vector) if norm else f'<span class="zero">все нули</span>'

    return (
        f'<div class="r"><span class="k">{key}</span>'
        f'<span class="v">{value}</span>'
        f'<span class="p">{"—" if position is None else position}</span>'
        f'<span class="n">{norm:.2f}</span>'
        f'<span class="num">{numbers}</span></div>'
    )


def _short(vector: torch.Tensor) -> str:

    numbers = " ".join(
        f"{float(number):+.3f}" for number in vector[:PREVIEW_NUMBERS]
    )

    return f"{numbers} …" if len(vector) > PREVIEW_NUMBERS else numbers


def _full(vector: torch.Tensor) -> str:

    return " ".join(f"{float(number):+.4f}" for number in vector)


__all__ = [
    "Names",
    "Page",
    "Shapes",
    "render",
]
