from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime

from .settings import MlmConfig


# ============================================================
# ВИДИМЫЙ РЕЗУЛЬТАТ
# ============================================================
#
# Один пример на странице: что было замаскировано, что модель
# увидела вместо значения и что она предсказала. Полного словаря
# логитов и длинных векторов здесь нет.
#
# Наверху страницы стоит предупреждение, и оно не формальность:
# веса начальные, и любые предсказания на них — шум. Страница
# показывает, что механика собрана верно, а не что модель хороша.
# ============================================================


@dataclass(frozen=True)
class Piece:
    """
    Один кусок замаскированного значения.
    """

    position: int
    label: str
    loss: float
    correct: bool
    top: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class Shot:
    """
    Один показанный пример.
    """

    client_id: str
    event: int
    time: datetime | None
    key: str
    target: str
    pieces: tuple[Piece, ...]


def render(
    group: str,
    config: MlmConfig,
    implementation: str,
    shot: Shot,
    counts: dict,
    paths: dict,
) -> str:
    """
    Страница целиком.
    """

    body = "\n".join(
        [
            _warning(),
            _head(group, shot, counts),
            _example(shot),
            _pieces(shot, config),
            _totals(group, counts),
            _technical(group, config, implementation, counts, paths),
        ]
    )

    return _frame(group, body)


def _frame(group: str, body: str) -> str:

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>MLM-голова — {html.escape(group)}</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin: 0 auto; padding: 28px 24px 60px; max-width: 900px;
  font: 14px/1.55 -apple-system, Segoe UI, Roboto, sans-serif; color: #1f2328;
  background: #fff; }}
h1 {{ font-size: 21px; margin: 0 0 10px; }}
h2 {{ font-size: 15px; margin: 28px 0 8px; color: #59636e;
  text-transform: uppercase; letter-spacing: .04em; }}
p {{ margin: 8px 0; }}
.who {{ color: #59636e; margin: 10px 0 0; }}
.alarm {{ background: #ffebe9; border: 1px solid #ff8182; border-radius: 8px;
  padding: 14px 16px; margin: 0 0 20px; font-size: 15px; }}
.card {{ border: 1px solid #d8dee4; border-radius: 10px; padding: 14px 18px;
  margin: 10px 0; }}
.card table {{ border-collapse: collapse; }}
.card td {{ padding: 4px 18px 4px 0; vertical-align: baseline; }}
.card td:first-child {{ color: #59636e; white-space: nowrap; width: 170px; }}
.mono {{ font: 12.5px Consolas, Menlo, monospace; }}
.mask {{ color: #cf222e; font: 12.5px Consolas, Menlo, monospace; }}
.piece {{ border: 1px solid #d8dee4; border-radius: 10px; padding: 10px 14px;
  margin: 10px 0; }}
.piece h3 {{ font-size: 13px; margin: 0 0 8px; color: #59636e; }}
.r {{ display: grid; grid-template-columns: 30px 1fr 90px 70px; gap: 12px;
  padding: 4px 6px; border-radius: 6px; align-items: baseline; }}
.r:nth-child(odd) {{ background: #f6f8fa; }}
.r.hit {{ background: #dafbe1; }}
.bar {{ display: inline-block; height: 9px; background: #0969da; border-radius: 2px;
  vertical-align: middle; }}
.n {{ font: 12px Consolas, Menlo, monospace; color: #59636e; }}
.tot {{ border-collapse: collapse; }}
.tot td {{ padding: 4px 20px 4px 0; }}
.tot td:first-child {{ color: #59636e; }}
details.tech {{ margin: 32px 0 0; border-top: 1px solid #d8dee4;
  padding-top: 12px; }}
details.tech > summary {{ cursor: pointer; color: #59636e; }}
details.tech table {{ border-collapse: collapse; margin: 10px 0; }}
details.tech td {{ padding: 3px 16px 3px 0; vertical-align: top; }}
details.tech td:first-child {{ color: #59636e; white-space: nowrap; }}
code {{ font: 12px Consolas, Menlo, monospace; background: #f3f4f6;
  padding: 1px 5px; border-radius: 4px; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def _warning() -> str:

    return """<div class="alarm"><b>Модель не обучена.</b> Веса всех энкодеров и
головы — начальный розыгрыш по seed, ни одного шага обучения не сделано.
Предсказания ниже это шум, и качеством модели они не являются. Страница
показывает, что механика собрана верно: цель выбрана, вход подменён на
<code>[MASK]</code>, три вектора сошлись в голове и дали логиты по словарю.</div>"""


def _head(group: str, shot: Shot, counts: dict) -> str:

    moment = "—" if shot.time is None else shot.time.isoformat(sep=" ")[:16]

    return f"""<h1>MLM-голова</h1>
<p class="who">группа {html.escape(group)} · клиент
 {html.escape(shot.client_id)} · событие {shot.event} ·
 {html.escape(moment)}</p>"""


def _example(shot: Shot) -> str:

    pieces = len(shot.pieces)

    seen = " ".join(["[MASK]"] * pieces)

    return f"""<h2>Что закрыто</h2>
<div class="card"><table>
<tr><td>ключ</td><td class="mono">{html.escape(shot.key)}</td></tr>
<tr><td>видит модель</td><td class="mask">{html.escape(seen)}</td></tr>
<tr><td>кусков в значении</td><td class="mono">{pieces}</td></tr>
<tr><td>исходное значение</td>
<td class="mono">{html.escape(shot.target)}
 <span class="n">— только в отчёте, в модель не подаётся</span></td></tr>
</table></div>
<p>Ключ модель видит: предсказывается значение. Само значение закрыто
<code>[MASK]</code> целиком, всеми своими кусками сразу.</p>"""


def _pieces(shot: Shot, config: MlmConfig) -> str:

    blocks = []

    for number, piece in enumerate(shot.pieces):

        rows = []

        for place, (name, probability) in enumerate(piece.top):

            hit = name == piece.label

            rows.append(
                f'<div class="r{" hit" if hit else ""}">'
                f'<span class="n">{place + 1}</span>'
                f"<span>{html.escape(name)}</span>"
                f'<span><span class="bar" style="width:{max(probability * 80, 1):.0f}px">'
                f"</span></span>"
                f'<span class="n">{probability * 100:.2f}%</span></div>'
            )

        blocks.append(
            f'<div class="piece"><h3>кусок {number} — позиция внутри значения '
            f"{piece.position}, цель <code>{html.escape(piece.label)}</code>, "
            f"потери {piece.loss:.3f}</h3>{''.join(rows)}</div>"
        )

    return f"""<h2>Предсказания по кускам</h2>
<p><b>Каждый блок ниже — отдельный кусок одного и того же значения.</b> Модель
предсказывает куски по одному, а не значение целиком: у каждого своя позиция
внутри значения, своя цель и свои потери. Показаны {config.top_k} лучших
вариантов из словаря.</p>
{"".join(blocks)}"""


def _totals(group: str, counts: dict) -> str:

    unknown = counts["unknown"]

    share = 100.0 * unknown / counts["targets"] if counts["targets"] else 0.0

    return f"""<h2>Итоги группы</h2>
<table class="tot">
<tr><td>целей</td><td class="mono">{counts['targets']}</td></tr>
<tr><td>кросс-энтропия</td><td class="mono">{counts['loss']:.4f}</td></tr>
<tr><td>угадано</td>
<td class="mono">{counts['correct']} ({100.0 * counts['correct'] / max(counts['targets'], 1):.2f}%)</td></tr>
<tr><td>цели с исходным [UNK]</td>
<td class="mono">{unknown} ({share:.1f}%)</td></tr>
</table>
<p>Цели с исходным <code>[UNK]</code> — это значения, которых не было в словаре,
собранном по train. Они остаются законными целями и из потерь не исключаются:
так же поступает эталон. Но считать их отдельно нужно, иначе доля угаданного
завышается — предсказать «неизвестно» легко.</p>
<p>Замены на <code>[UNK]</code>, которые сделал сам маскер, здесь ни при чём:
их он исключил из ошибки ещё на этапе 08, оставив метку −100.</p>"""


def _technical(group, config, implementation, counts, paths) -> str:

    return f"""<details class="tech">
<summary>Технические сведения</summary>

<table>
<tr><td>группа</td><td>{html.escape(group)}</td></tr>
<tr><td>d</td><td>{counts['dim']} — из весов этапа 09</td></tr>
<tr><td>голова</td>
<td>Linear(3d → d) без нормировки и активации, логиты связанными весами общей
 таблицы; seed {config.seed}</td></tr>
<tr><td>сглаживание меток</td><td>{config.label_smoothing}</td></tr>
<tr><td>версия этапа</td><td>{html.escape(implementation)}</td></tr>
<tr><td>клиентов</td><td>{counts['clients']}, событий {counts['events']}</td></tr>
<tr><td>считано на</td><td>{html.escape(counts['device'])}</td></tr>
<tr><td>цели по механизму</td>
<td class="mono">{html.escape(counts['by_reason'])}</td></tr>
<tr><td>результаты по целям</td><td><code>{html.escape(str(paths['targets']))}</code></td></tr>
<tr><td>веса головы</td><td><code>{html.escape(str(paths['weights']))}</code></td></tr>
<tr><td>вход: структура</td><td><code>{html.escape(str(paths['batches']))}</code></td></tr>
<tr><td>вход: видимые значения и метки</td>
<td><code>{html.escape(str(paths['masked']))}</code></td></tr>
</table>

<p><b>Сквозной проход.</b> Векторы считаются моделью здесь и сейчас:
<code>InputEmbedding → Event Encoder → Profile Encoder → History Encoder →
MLM</code>. Векторы этапов 10–12 входом не служат — модель собрана из
начальных весов (data/09_embeddings/train и data/09_backbone), тех же, с которых
начинается обучение. Градиент от потерь доходит до общей таблицы эмбеддингов и
всех трёх энкодеров.</p>

<p><b>Что не подаётся в модель.</b> <code>labels</code> — только в потери и в
этот отчёт; <code>value_ids_source</code> не читается вовсе; <code>reason</code>
— только разбивка в отчёте. Заполнителя во входе не существует: массивы
обрезаны по длине клиента.</p>

<p>Отчёт считается под <code>no_grad</code>. Это единственное его место: внутри
самого прохода нет ни <code>detach</code>, ни NumPy, ни <code>no_grad</code>.</p>

</details>"""


__all__ = ["Piece", "Shot", "render"]
