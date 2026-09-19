from __future__ import annotations

from src.preprocessing.artifacts import _md_table


# ============================================================
# ИДЕЯ
# ============================================================
#
# Отчёт читает человек, а не программа. Поэтому здесь нет ни
# одного числа, которого нет в JSON-артефакте того же этапа:
# markdown только показывает, а источником правды остаётся
# манифест.
# ============================================================


def render_contract_md(report: dict) -> str:

    out: list[str] = []

    corpus = report["corpus"]
    keys = report["keys"]
    readiness = report["readiness"]

    out.append(f"# Контракт входа токенизатора: группа {report['group']}\n")

    out.append(
        f"Формат {report['format_version']}, реализация {report['implementation_version']}. "
        f"Fit разрешён до {report['fit_end'][:10]} исключительно.\n"
    )

    if readiness["status"] == "ready":
        out.append("\nНабор пригоден целиком.\n")
    else:
        out.append(
            "\n**Вердикт: диагностика.** Набор технически исправен, но договорённости выполнены не все, "
            "и всё, что выпустит токенизатор, помечено соответственно:\n"
        )
        out.extend(f"- {item}" for item in readiness["reasons"])
        out.append("")

    # --- корпус ---

    out.append("\n## Что пойдёт в fit\n")

    out.append(
        _md_table(
            [
                ["клиентов", corpus["clients"]],
                ["событий", corpus["events"]],
                ["значений под ключами", corpus["values"]],
                ["действующих анкет (по одной на клиента)", corpus["profiles_as_of"]],
                ["клиентов без профиля", corpus["clients_without_profile"]],
                ["событий с дневной точностью времени", corpus["day_precision_events"]],
            ],
            ["что", "сколько"],
        )
    )

    declared = report["declared_by_split"]

    out.append(
        f"\nРазделение объявило {declared['events_rows']} видимых строк: прочитано столько же. "
        f"Версий профиля к fit_end известно {declared['source_profile_version_rows']}, но признаком клиента "
        f"становится одна действующая на каждого, а прежние значения приходят событиями изменения "
        f"профиля. Контрольная сумма содержимого разделения {declared['content_sha256'][:16]}, "
        f"отпечаток смыслового содержимого fit-корпуса {report['fit_content_sha256'][:16]}.\n"
    )

    out.append(
        "\nОстальные группы в fit не входят вовсе: они существуют только для transform "
        "замороженными артефактами.\n"
    )

    # --- ключи ---

    out.append("\n## Ключи\n")

    out.append(
        _md_table(
            [
                ["numeric", keys["by_kind"]["numeric"]],
                ["categorical", keys["by_kind"]["categorical"]],
                ["text", keys["by_kind"]["text"]],
                ["ссылки (в словарь не входят)", keys["link"]],
                ["всего объявлено", keys["declared"]],
                ["встретилось на train", keys["observed_in_train"]],
            ],
            ["вид значения", "ключей"],
        )
    )

    rows = [row for row in keys["rows"] if row["role"] == "model_feature" and row["observed_in_train"]]
    rows.sort(key=lambda row: (-row["observations"], row["key"]))

    out.append("\nДесять самых частых ключей корпуса:\n")

    out.append(
        _md_table(
            [
                [row["key"], row["value_kind"], row["unit"], row["weight_rule"], row["observations"], row["clients"]]
                for row in rows[:10]
            ],
            ["ключ", "вид", "единица", "правило веса", "наблюдений", "клиентов"],
        )
    )

    if keys["unobserved_in_train"]:
        out.append(
            f"\nНе встретились на train {len(keys['unobserved_in_train'])} ключей. "
            "Они остаются в словаре с пометкой: заранее объявленный смысл без наблюдений это не ошибка, "
            "и подглядывать за ними в validation или test запрещено.\n"
        )

    # --- пропуски ---

    missing = report["missing"]

    out.append("\n## Объявлено, но не пришло\n")

    out.append(f"{missing['rule']}.\n")

    if missing["top"]:
        out.append(
            "\n"
            + _md_table(
                [[item["event_type"], item["key"], item["missing_events"]] for item in missing["top"]],
                ["тип события", "ключ", "событий без значения"],
            )
        )

    if report["absent_reasons"]:
        out.append("\n## Почему расчёта нет\n")
        out.append(
            "Причина стоит рядом со значением, а не вместо него: ноль и «не с чем сравнивать» "
            "это разные вещи.\n\n"
        )
        out.append(
            _md_table(
                [
                    [key, reason, count]
                    for key, reasons in report["absent_reasons"].items()
                    for reason, count in reasons.items()
                ],
                ["ключ", "причина", "случаев"],
            )
        )

    # --- принятые ограничения ---

    if report.get("accepted_limits"):

        out.append("\n## Принятые ограничения V1\n")

        out.append(
            "Это не открытые вопросы, а решения, которые уже приняты и записаны. "
            "Их место здесь, а не среди противоречий: решённое не должно годами выглядеть "
            "нерешённым.\n\n"
        )

        for item in report["accepted_limits"]:
            out.append(f"**{item['key']}** — {item['kind']}")
            out.append(f"- {item['detail']}")
            out.append(f"- решение: {item['decision']}")
            out.append("")

    # --- противоречия ---

    out.append("\n## Противоречия\n")

    if not report["contradictions"]:
        out.append("Не найдено.\n")
    else:
        out.append(
            "Смысл ведёт препроцессинг. Токенизатор следует реестру и называет несоответствие, "
            "а не исправляет его у себя внутри.\n\n"
        )
        for item in report["contradictions"]:
            out.append(f"**{item['key']}** — {item['kind']}")
            out.append(f"- {item['detail']}")
            if item["examples"]:
                out.append("- примеры: " + ", ".join(f"`{value}`" for value in item["examples"]))
            out.append(f"- предложение: {item['proposal']}")
            out.append("")

    # --- ограничения ---

    out.append("\n## Ограничения входа\n")
    out.extend(f"- {item}" for item in report["limitations"])
    out.append("")

    return "\n".join(out)


def render_values_md(report: dict) -> str:
    """
    Смыслы и их значения глазами человека.
    """

    out: list[str] = []

    counts = report["counts"]

    out.append(f"# Смыслы и значения: группа {report['group']}\n")

    out.append(
        f"Ключей с кодом {counts['keys_model_feature']}: {counts['numeric']} числовых, "
        f"{counts['categorical']} категориальных, {counts['text']} текстовых. "
        f"Ссылок {counts['link']} — они остаются связями и кода не получают.\n"
    )

    out.append(
        f"\nДоменов значений {counts['domains']}, из них объединённых {counts['domains_shared']}. "
        f"Различных категориальных значений {counts['values']}"
        + (f", помечено редкими {counts['rare']}" if counts["rare"] else "")
        + ".\n"
    )

    out.append("\n## Правила\n")
    out.extend(f"- **{name}**: {rule}" for name, rule in sorted(report["rules"].items()))
    out.append("")

    # --- объединения ---

    shared = [item for item in report["domains"] if item["shared"]]

    out.append("\n## Что объединено\n")

    if not shared:
        out.append("Ничего: у каждого ключа свой домен.\n")
    else:
        out.append(
            _md_table(
                [[item["name"], ", ".join(item["keys"]), len(item["values"]), item["reason"]] for item in shared],
                ["домен", "ключи", "значений", "почему"],
            )
        )

    out.append("\n## Что не объединено намеренно\n")

    out.append(
        _md_table(
            [[", ".join(item["keys"]), item["reason"]] for item in report["declined_domains"]],
            ["ключи", "почему"],
        )
    )

    out.append(
        "\nОтдельно смысловой реестр объявил несовместимыми "
        f"{len(report['ambiguous'])} групп ключей: объединить их конфигурацией нельзя, "
        "проверка этапа 1 такую настройку отвергает.\n"
    )

    # --- примеры ---

    examples: list[list] = []

    # Ключи с самым богатым набором значений: на них видно, что
    # номер это место в домене, а не порядок появления.
    rich = sorted(
        (row for row in report["keys"] if row["value_kind"] == "categorical" and row["candidates"]),
        key=lambda row: (-len(row["candidates"]), row["key"]),
    )

    for row in rich[:15]:

        domain = next(item for item in report["domains"] if item["name"] == row["domain"])
        order = {(value["value_type"], value["value"]): index for index, value in enumerate(domain["values"])}

        candidate = row["candidates"][len(row["candidates"]) // 2]
        physical = row["physical_fields"][0] if row["physical_fields"] else "—"

        examples.append(
            [
                physical,
                candidate["value"],
                row["key"],
                row["domain"],
                f"{order[(candidate['value_type'], candidate['value'])]} из {len(domain['values'])}",
            ]
        )

    out.append("\n## Примеры\n")

    out.append(
        "Номер это место значения в своём домене. Общий ID появится на этапе 4, "
        "когда известны все виды токенов сразу.\n\n"
    )

    out.append(
        _md_table(examples, ["физическое поле", "значение", "semantic key", "домен", "номер в домене"])
    )

    # --- крупнейшие домены ---

    biggest = sorted(report["domains"], key=lambda item: (-len(item["values"]), item["name"]))[:10]

    out.append("\n## Крупнейшие домены\n")

    out.append(
        _md_table(
            [
                [item["name"], len(item["values"]), ", ".join(value["value"] for value in item["values"][:4])]
                for item in biggest
            ],
            ["домен", "значений", "первые значения"],
        )
    )

    if report["unobserved_in_train"]:
        out.append(
            f"\n## Без наблюдений на train\n\n{len(report['unobserved_in_train'])} ключей: "
            + ", ".join(report["unobserved_in_train"][:20])
            + (" и другие" if len(report["unobserved_in_train"]) > 20 else "")
            + ".\n\nОни остаются в словаре: объявленный заранее смысл без наблюдений это не ошибка. "
            "Заглядывать за ними в validation и test запрещено.\n"
        )

    return "\n".join(out)


def render_buckets_md(report: dict) -> str:
    """
    Числовые диапазоны глазами человека.
    """

    out: list[str] = []

    counts = report["counts"]

    out.append(f"# Числовые диапазоны: группа {report['group']}\n")

    out.append(
        f"Числовых ключей {counts['keys']}, диапазонов всего {counts['buckets']}. "
        f"Границы считаются по train до {report['fit_period']['until'][:10]} исключительно.\n"
    )

    out.append(
        f"\nБез наблюдений на train осталось {counts['buckets_without_observations']} диапазонов. "
        "У квантильных шкал таких нет по построению; у шкал бизнеса они законны: шкала объявлена "
        "заранее и под выборку не подстраивается.\n"
    )

    out.append("\n## Правила\n")
    out.extend(f"- **{name}**: {rule}" for name, rule in sorted(report["rules"].items()))
    out.append("")

    out.append("\n## Откуда границы\n")

    source_names = {
        "train_quantiles": "квантили train",
        "config_fixed": "шкала бизнеса",
        "config_fallback": "объявленная шкала вместо квантилей",
        "none": "шкалы нет",
    }

    out.append(
        _md_table(
            [[source_names.get(name, name), count] for name, count in sorted(counts["by_source"].items())],
            ["источник границ", "ключей"],
        )
    )

    # --- подробно по нескольким ключам ---

    interesting = [
        key
        for key in ("transaction_amount", "balance_after", "days_past_due", "amount_to_declared_income")
        if key in report["encoders"]
    ]

    for key in interesting:

        entry = report["encoders"][key]

        out.append(f"\n## {key}\n")

        out.append(
            f"{entry['note']}. Метод {entry['method']}, источник границ "
            f"{source_names.get(entry['boundary_source'], entry['boundary_source'])}, "
            f"диапазонов {entry['actual_bins']}.\n\n"
        )

        rows = [
            [bucket["label"], count]
            for bucket, count in zip(entry["buckets"], entry["distribution"]["buckets"])
        ]

        out.append(_md_table(rows, ["диапазон", "наблюдений train"]))

    # --- все ключи одной таблицей ---

    out.append("\n## Все ключи\n")

    out.append(
        _md_table(
            [
                [
                    key,
                    entry["unit"],
                    entry["method"],
                    source_names.get(entry["boundary_source"], entry["boundary_source"]),
                    entry["actual_bins"],
                    entry["fit"]["values"],
                    entry["zero_policy"],
                    entry["negative_policy"],
                ]
                for key, entry in report["encoders"].items()
            ],
            ["ключ", "единица", "метод", "границы", "диапазонов", "значений train", "ноль", "минус"],
        )
    )

    if report["warnings"]:
        out.append("\n## Предупреждения\n")
        out.extend(f"- {item}" for item in report["warnings"])
        out.append("")

    return "\n".join(out)


def render_vocab_md(manifest: dict, values: list[dict], bpe) -> str:
    """
    Итог заморозки: размеры, диапазоны и реальное разбиение
    нескольких названий.
    """

    out: list[str] = []

    layout = manifest["layout"]
    sizes = layout["sizes"]

    out.append(f"# Словарь заморожен: {manifest['artifact_id']}\n")

    out.append(
        f"Формат {manifest['format_version']}, реализация {manifest['implementation_version']}, "
        f"группа {manifest['group']}, fit до {manifest['fit_end'][:10]} исключительно. "
        f"Готовность {manifest['readiness']['status']}.\n"
    )

    out.append("\n## Размеры\n")

    out.append(
        _md_table(
            [
                ["специальные", sizes["special"]],
                ["ключи", sizes["keys"]],
                ["категориальные значения", sizes["categorical"]],
                ["числовые диапазоны", sizes["buckets"]],
                ["куски BPE", sizes["bpe"]],
                ["всего ID", sizes["total"]],
            ],
            ["вид токена", "сколько"],
        )
    )

    out.append("\n## Диапазоны\n")

    out.append(
        _md_table(
            [[name, f"[{start}, {end})", end - start] for name, (start, end) in layout["ranges"].items()],
            ["диапазон", "ID", "размер"],
        )
    )

    out.extend(f"- {item}" for item in layout["invariants"])
    out.append("")

    # --- BPE ---

    out.append("\n## Разбиение текста\n")

    info = manifest["bpe"]

    if not info.get("enabled"):
        out.append(f"Выключено: {info.get('reason', 'разрешённых текстовых полей нет')}.\n")
    else:
        out.append(
            f"Байтовый BPE, библиотека {info['library']['name']} {info['library']['version']}. "
            f"Ключи: {', '.join(info['keys'])}. Запрошено {info['vocab_size']['requested']} токенов, "
            f"получилось {info['vocab_size']['actual']}. Корпус: {info['corpus']['texts']} различных "
            f"текстов, {info['corpus']['occurrences']} вхождений, самый длинный "
            f"{info['corpus']['longest_bytes']} байт.\n"
        )

        out.append(f"\nНормализация: {info['normalization']['rule']}.\n")

        pieces = info.get("pieces_per_text")

        if pieces:
            out.append(
                f"\nКусков на текст: в среднем {pieces['mean']}, медиана {pieces['median']}, "
                f"максимум {pieces['max']}. {pieces['note']}.\n"
            )

        examples = [
            text
            for text in (
                "europharma алматы",
                "қазпошта",
                "magnum astana",
                "sulpak шымкент",
            )
        ]

        rows = []

        for text in examples:
            pieces = bpe.pieces(text)
            rows.append([text, len(pieces), " | ".join(bpe.piece(index) for index in pieces)])

        out.append("\n" + _md_table(rows, ["текст", "кусков", "разбиение"]))

        out.append(
            "\nСимвол Ġ в записи куска это пробел: так байтовый алфавит показывает границу слова.\n"
        )

    # --- примеры значений ---

    out.append("\n## Как выглядят значения\n")

    sample = [row for row in values if row["kind"] == "bucket"][:4]
    sample += [row for row in values if row["kind"] == "categorical"][:6]

    out.append(
        _md_table(
            [[row["id"], row["kind"], row["label"], row["train_count"]] for row in sample],
            ["ID", "вид", "метка", "наблюдений train"],
        )
    )

    out.append("\n## Правила\n")
    out.extend(f"- **{name}**: {rule}" for name, rule in sorted(manifest["rules"].items()))
    out.append("")

    return "\n".join(out)


def render_tokenization_md(report: dict) -> str:
    """
    Итог кодирования одной группы.
    """

    out: list[str] = []

    counts = report["counts"]
    specials = report["specials"]

    out.append(f"# Токенизация: группа {report['group']}\n")

    out.append(
        f"Словарь {report['artifact_id']}, формат {report['format_version']}. "
        f"Срезы: {', '.join(item[:10] for item in report['cutoffs']) or '—'}. "
        f"Готовность {report['readiness']['status']}.\n"
    )

    if report["skipped_cutoffs"]:
        out.append("\nПропущенные срезы:\n")
        out.extend(f"- {item['cutoff'][:10]}: {item['reason']}" for item in report["skipped_cutoffs"])
        out.append("")

    out.append("\n## Сколько закодировано\n")

    out.append(
        _md_table(
            [
                ["клиентов", counts["clients"]],
                ["пар «клиент, срез»", counts["client_slices"]],
                ["событий", counts["events"]],
                ["значений всего", counts["values"]],
                ["из них в событиях", counts["event_values"]],
                ["из них в профилях", counts["profile_values"]],
                ["токенов всего", counts["tokens"]],
                ["представлений профиля", counts["profiles"]],
                ["клиентов без профиля", counts["clients_without_profile"]],
                ["клиентов без событий на срезе", counts["silent_clients"]],
                ["событий с дневной точностью", counts["day_precision_events"]],
            ],
            ["что", "сколько"],
        )
    )

    out.append("\n## Специальные токены в слоте значения\n")

    out.append(
        _md_table(
            [
                ["[MISSING] — объявленный ключ без значения", specials["missing"]],
                ["[UNK] — значение вне словаря", specials["unknown"]],
                ["[INVALID] — невозможное число", specials["invalid"]],
                ["[EMPTY] — текст без символов", specials["empty"]],
            ],
            ["токен", "позиций"],
        )
    )

    out.append(
        "\nЭто четыре разные вещи, и ни одна из них не «ноль». "
        "Маскированию они не подлежат: предсказывать отсутствие значения бессмысленно.\n"
    )

    lengths = report["lengths"]

    out.append("\n## Длины\n")

    out.append(
        _md_table(
            [
                ["токенов в событии", lengths["event_tokens"]["mean"], lengths["event_tokens"]["median"],
                 lengths["event_tokens"]["max"]],
                ["кусков в текстовом значении", lengths["text_pieces"]["mean"],
                 lengths["text_pieces"]["median"], lengths["text_pieces"]["max"]],
            ],
            ["что", "в среднем", "медиана", "максимум"],
        )
    )

    out.append(
        "\nНичего не обрезается: предел кусков в значении это явная ошибка, а не тихое укорачивание.\n"
    )

    if report["edge_buckets"]:

        top = sorted(report["edge_buckets"].items(), key=lambda item: (-item[1], item[0]))[:10]

        out.append("\n## Крайние диапазоны\n")
        out.append(
            "Сколько значений попало в первый или последний диапазон своей шкалы. "
            "Большая доля означает, что шкала данным не подходит.\n\n"
        )
        out.append(_md_table([[key, count] for key, count in top], ["ключ", "значений"]))

    if report["unknown_keys"]:
        out.append("\n## Ключи вне словаря\n")
        out.append(
            _md_table(
                [[key, count] for key, count in report["unknown_keys"].items()],
                ["ключ", "значений"],
            )
        )
        out.append(
            "\nИсходное значение при этом не потеряно: оно осталось в canonical, "
            "а запись о нём лежит в колонке unknown_keys.\n"
        )

    out.append("\n## Контракт записи\n")
    out.extend(f"- **{name}**: {rule}" for name, rule in sorted(report["contract"].items())
               if isinstance(rule, str))
    out.append("")

    if report["limitations"]:
        out.append("\n## Ограничения\n")
        out.extend(f"- {item}" for item in report["limitations"])
        out.append("")

    return "\n".join(out)


def render_golden_md(examples: list[dict]) -> str:
    """
    Примеры «исходное → смысл → ID → расшифровка».
    """

    out: list[str] = []

    out.append("# Читаемые примеры\n")

    out.append(
        "Каждая строка это одна содержательная позиция: ключ, его код, коды значения и то, "
        "как они читаются обратно. Точное число берётся по трассировке из canonical: "
        "диапазон его не хранит и хранить не должен.\n"
    )

    for example in examples:

        out.append(
            f"\n## {example['event_type']} · {example['event_time'][:19]} · клиент {example['client_id']}\n"
        )

        out.append(
            f"Срез {example['cutoff'][:10]}, запись {example['event_id']} версии "
            f"{example['event_version']}, токенов {example['n_tokens']}, "
            f"час наблюдался: {'да' if example['hour_known'] else 'нет'}.\n\n"
        )

        out.append(
            _md_table(
                [
                    [
                        item["key"],
                        item["key_id"],
                        _md_cell_short(item["source_value"]),
                        ", ".join(str(value) for value in item["value_ids"]),
                        (
                            f"«{item['decoded_text']}» ({len(item['value_ids'])} кусков)"
                            if "decoded_text" in item
                            else " | ".join(item["decoded"])
                        ),
                    ]
                    for item in example["pairs"]
                ],
                ["ключ", "код ключа", "исходное значение", "коды значения", "расшифровка"],
            )
        )

        if example["refs"]:
            out.append(
                "\nСвязи: "
                + ", ".join(f"{key} = {value}" for key, value in example["refs"].items())
                + ". Кодом они не становятся.\n"
            )

        if example["absent_reasons"]:
            out.append(
                "\nПочему расчёта нет: "
                + ", ".join(f"{key} — {reason}" for key, reason in example["absent_reasons"].items())
                + ".\n"
            )

        if example["provenance"]:
            out.append(
                "\nПроисхождение: "
                + "; ".join(
                    f"{key} ← " + ", ".join(_source(item) for item in sources)
                    for key, sources in example["provenance"].items()
                )
                + ".\n"
            )

    return "\n".join(out)


def _source(item: dict) -> str:

    kind = item.get("kind")

    if kind == "event":
        return f"событие {item.get('event_id', '?')} v{item.get('event_version', '?')}.{item.get('key')}"

    if kind == "profile":
        return f"профиль v{item.get('profile_version')}.{item.get('key')}"

    if kind == "entity":
        return f"сущность {item.get('ref')}.{item.get('key')}"

    if kind == "client_past":
        return f"прошлое клиента ({item.get('events')} событий)"

    if kind == "catalog":
        return f"справочник {item.get('table')} по {item.get('via')}"

    return str(item)


def _md_cell_short(value) -> str:

    if value is None:
        return "—"

    text = str(value)

    return text if len(text) <= 40 else text[:37] + "..."


def render_compatibility_md(report: dict, tokenization: dict | None = None) -> str:
    """
    Чем новый формат отличается от прежнего и что обязан сделать
    будущий потребитель.

    Первая редакция пишется уже здесь: различия следуют из
    контракта, а не из готовых токенов, и знать о них надо до
    того, как словарь построен.
    """

    out: list[str] = []

    out.append("# Совместимость: новый токенизатор против прежнего\n")

    if tokenization is None:
        out.append(
            f"Формат {report['format_version']}, реализация {report['implementation_version']}. "
            "Редакция этапа 1: перечисляет различия контракта. Окончательная редакция выходит на "
            "этапе 5, когда формат подтверждён реальными записями.\n"
        )
    else:
        counts = tokenization["counts"]
        out.append(
            f"Формат {report['format_version']}, реализация {report['implementation_version']}, "
            f"словарь {tokenization['artifact_id']}. Окончательная редакция: формат подтверждён "
            f"{counts['events']} записями группы {tokenization['group']} "
            f"({counts['tokens']} токенов).\n"
        )

    out.append("\n## Состояние прежнего кода\n")

    out.append(
        "Пакет `src/tokenizer` **удалён**. К моменту удаления он не импортировался уже ничем: его "
        "модули обращались к `src.preprocessing.config`, `build`, `buckets`, `cutoffs`, `stats`, `fit`, "
        "исчезнувшим вместе с контрактом RAW v3, и к путям `ARTIFACTS_DIR`, `PROCESSED_DIR`, "
        "`TOKENIZED_DIR`, которых в `src/generator/config.py` больше нет. Ни одного собранного им "
        "словаря или датасета на диске не было. Прежний контракт описан ниже по памяти кода и по "
        "истории git, а не по работающей системе.\n"
    )

    out.append(
        "\n`src/model` остаётся в дереве и по-прежнему не импортируется: он зависел от того же "
        "прежнего токенизатора. Его переделка это отдельная работа, и список требований к ней "
        "перечислен здесь.\n"
    )

    out.append("\n## Различия контракта\n")

    rows = [
        [
            "positions",
            "номер поля внутри события: [EVT]=0, event_type=1, payload с 2",
            "номер куска внутри одного значения: 0 у числа и категории, 0..n−1 у текста",
        ],
        [
            "ширина события",
            "фиксированная, поля в порядке реестра, пропуск занимает своё место",
            "переменная: пары только у объявленных ключей, порядок по key_id и смысла не несёт",
        ],
        [
            "field_ids",
            "обязательный массив, им индексируются маскирование, кандидаты и статистика",
            "массива нет; его роль берут key_id и словарь ключей",
        ],
        [
            "event_type",
            "фиксированная позиция 1",
            "обычная пара ключ/значение",
        ],
        [
            "текст",
            "BPE не применялся вовсе",
            "байтовый BPE по подтверждённым текстовым ключам, куски делят один key_id",
        ],
        [
            "числа",
            "корзины препроцессинга, локальный индекс равен номеру корзины",
            "границы считает сам токенизатор на train, метка бакета несёт ключ, единицу и границы",
        ],
        [
            "ссылки на сущности",
            "сырых идентификаторов в модели не было",
            "восемь ключей-ссылок остаются метаданными связи и в словарь не входят",
        ],
        [
            "происхождение",
            "не хранилось",
            "у каждого расчётного значения есть derived_from: событие, версия профиля, справочник",
        ],
    ]

    out.append(_md_table(rows, ["что", "прежний контракт", "новый контракт"]))

    out.append("\n## Требования к будущему потребителю\n")

    out.extend(
        f"- {item}"
        for item in (
            "маркеры ставит токенизатор: ровно один [EVT] на событие и один [USR] на профиль, "
            "`marker_owner = tokenizer`; добавлять их второй раз нельзя",
            "события переменной ширины: потребитель с фиксированной шириной несовместим автоматически",
            "таблица позиций модели ограничивает номер куска, а не длину текста: выход за её размер "
            "это ошибка потребителя, а не право токенизатора обрезать значение",
            "[MISSING], [UNK], [INVALID] и [EMPTY] в слоте значения маскированию не подлежат",
            "значение и его производное нельзя маскировать по отдельности: список слагаемых лежит "
            "в derived_from каждого расчётного значения",
            "календарь приходит отдельным числовым каналом; у события с дневной точностью час не "
            "наблюдался, и признак hour_known обязан доехать до Masker",
            "ссылки и причины отсутствия это метаданные: в embedding они не входят",
            "сохранённая токенизация это ещё не обучающий датасет: расписание срезов, доступную "
            "историю и периоды целей назначает отдельный построитель",
        )
    )

    out.append("")

    out.append("\n## Что уже решено этим этапом\n")

    out.append(
        f"Ключей объявлено {report['keys']['declared']}, из них {report['keys']['model_feature']} получают код "
        f"и {report['keys']['link']} остаются связями. Правило веса записано по каждому ключу: значения "
        "события считаются по событиям, значения профиля по клиентам.\n"
    )

    if report["contradictions"]:
        out.append(
            "\nОткрытых противоречий: "
            + str(len(report["contradictions"]))
            + ". Они перечислены в contract_report.md и ждут решения человека; "
            "тихо переименовывать смыслы токенизатор не будет.\n"
        )

    if tokenization is not None:

        specials = tokenization["specials"]
        lengths = tokenization["lengths"]

        out.append("\n## Что подтвердила реальная запись\n")

        out.append(
            _md_table(
                [
                    ["событий закодировано", tokenization["counts"]["events"]],
                    ["токенов", tokenization["counts"]["tokens"]],
                    ["токенов в событии, медиана", lengths["event_tokens"]["median"]],
                    ["токенов в событии, максимум", lengths["event_tokens"]["max"]],
                    ["кусков в текстовом значении, максимум", lengths["text_pieces"]["max"]],
                    ["[MISSING] в слоте значения", specials["missing"]],
                    ["[UNK] в слоте значения", specials["unknown"]],
                    ["[INVALID] в слоте значения", specials["invalid"]],
                    ["[EMPTY] в слоте значения", specials["empty"]],
                ],
                ["что", "сколько"],
            )
        )

        out.append(
            f"\nСобытие занимает до {lengths['event_tokens']['max']} токенов, и это не предел формата: "
            "прежний лимит «до 12 токенов на событие» снят вместе с фиксированной шириной. "
            "Потребитель, рассчитывающий на 12, несовместим.\n"
        )

    return "\n".join(out)


__all__ = [
    "render_buckets_md",
    "render_compatibility_md",
    "render_contract_md",
    "render_values_md",
    "render_vocab_md",
]
