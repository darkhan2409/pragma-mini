from __future__ import annotations

from src.preprocessing.artifacts import _md_table


# ============================================================
# ИДЕЯ
# ============================================================
#
# Отчёт читает человек, и он обязан отвечать на четыре вопроса:
# сколько собрано, насколько длинны примеры, что потеряно при
# отборе и чего в наборе нет вовсе.
#
# Потери контекста печатаются даже когда их ноль: строка «целей
# потеряно 0» это утверждение, а отсутствие строки — умолчание.
# ============================================================


def render_contract_md(report: dict) -> str:
    """
    Контракт входа и формат примера.
    """

    out: list[str] = []

    inputs = report["inputs"]

    out.append("# Контракт датасета\n")

    out.append(
        f"Набор получит имя {report['dataset_id']}. Формат {report['format_version']}, "
        f"готовность {report['readiness']['status']}.\n"
    )

    out.append("\n## Входы\n")

    out.append(
        _md_table(
            [
                ["словарь", inputs["vocabulary"]["artifact_id"]],
                ["отпечаток словаря", inputs["vocabulary"]["vocab_sha256"][:16]],
                ["fit-группа словаря", inputs["vocabulary"]["fit_group"]],
                ["fit_end", inputs["vocabulary"]["fit_end"][:10]],
            ],
            ["что", "значение"],
        )
    )

    out.append("\n## Группы и срезы\n")

    out.append(
        _md_table(
            [
                [
                    name,
                    group["clients"],
                    ", ".join(item[:10] for item in group["cutoffs"]),
                    round(group["weight"], 4),
                    group["window"]["target_start"][:10] + " … " + group["window"]["target_end"][:10],
                    group["declared_eligible_at_final_cutoff"],
                ]
                for name, group in sorted(inputs["groups"].items())
            ],
            ["группа", "клиентов", "срезы", "вес примера", "период целей", "целей по разделению"],
        )
    )

    out.append("\n## Что лежит в примере\n")

    for name, columns in sorted(report["channels"].items()):
        out.append(f"\n**{name}**: {', '.join(columns)}\n")

    out.append("\n## Правила\n")
    out.extend(f"- **{name}**: {rule}" for name, rule in sorted(report["contract"].items())
               if isinstance(rule, str))
    out.append("")

    if report["readiness"]["reasons"]:
        out.append("\n## Почему это диагностика\n")
        out.extend(f"- {item}" for item in report["readiness"]["reasons"])
        out.append("")

    return "\n".join(out)


def render_report_md(report: dict) -> str:
    """
    Итог сборки.
    """

    out: list[str] = []

    counts = report["counts"]

    out.append(f"# Датасет {report['dataset_id']}\n")

    out.append(
        f"Формат {report['format_version']}, политика контекста "
        f"{report['config']['context']['policy']}, готовность {report['readiness']['status']}.\n"
    )

    out.append("\n## Объём\n")

    out.append(
        _md_table(
            [
                [
                    group,
                    item["samples"],
                    item["clients"],
                    item["events"],
                    item["tokens"],
                    item["eligible_events"],
                    item["samples_without_targets"],
                    item["empty_history"],
                    item["truncated"],
                ]
                for group, item in sorted(counts["by_group"].items())
            ],
            ["группа", "примеров", "клиентов", "событий", "токенов", "целей",
             "без целей", "пустых историй", "усечённых"],
        )
    )

    out.append("\n## Длины примеров\n")

    for group, item in sorted(counts["by_group"].items()):

        lengths = item["lengths"]

        out.append(f"\n**{group}**\n\n")
        out.append(
            _md_table(
                [
                    [name, value["p50"], value["p90"], value["p95"], value["p99"], value["max"]]
                    for name, value in sorted(lengths.items())
                ],
                ["что", "p50", "p90", "p95", "p99", "максимум"],
            )
        )

    out.append("\n## Потери контекста\n")

    out.append(
        _md_table(
            [
                [
                    group,
                    item["excluded_events"],
                    item["excluded_tokens"],
                    item["excluded_eligible"],
                    item["excluded_milestones"],
                ]
                for group, item in sorted(counts["by_group"].items())
            ],
            ["группа", "событий исключено", "токенов", "потеряно целей", "потеряно вех"],
        )
    )

    out.append(
        "\nПотерянная цель это событие периода целей, не попавшее в пример. "
        "В оценочных группах такое запрещено вовсе.\n"
    )

    lost = {
        group: item["excluded_by_type"]
        for group, item in counts["by_group"].items()
        if item["excluded_by_type"]
    }

    if lost:
        out.append("\n### Что именно исключено\n")
        for group, types in sorted(lost.items()):
            top = sorted(types.items(), key=lambda item: (-item[1], item[0]))[:10]
            out.append(f"\n**{group}**: " + ", ".join(f"{name} {count}" for name, count in top) + "\n")

    out.append("\n## Сверка периода целей с разделением\n")

    if report["eligible_agreement"]:
        out.append(
            _md_table(
                [
                    [group, item["dataset"], item["corpus_manifest"], item["agree"], item["comparable"]]
                    for group, item in sorted(report["eligible_agreement"].items())
                ],
                ["группа", "в наборе", "в разделении", "совпало", "сравнимо"],
            )
        )
        out.append(
            "\nСравнимо только на конечном срезе и при политике all: отбор контекста и "
            "дополнительные срезы меняют число законно.\n"
        )
    else:
        out.append("\nРазделение числа целей не объявило.\n")

    out.append("\n## Состояния и специальные токены\n")

    for group, item in sorted(counts["by_group"].items()):
        out.append(
            f"\n**{group}**: "
            + ", ".join(f"{name} {count}" for name, count in sorted(item["specials"].items()))
            + "; анкета: "
            + ", ".join(f"{name} {count}" for name, count in sorted(item["profile_states"].items()))
            + "\n"
        )

    out.append("\n## Происхождение значений\n")

    for group, item in sorted(counts["by_group"].items()):
        if item["dependencies"]:
            out.append(
                f"\n**{group}**: "
                + ", ".join(f"{name} {count}" for name, count in sorted(item["dependencies"].items()))
                + "\n"
            )

    out.append("\n## Доступность источников\n")

    out.append(
        f"\nПорядок колонки: {', '.join(report['sources'])}. "
        f"Коды состояний: {', '.join(f'{number} = {name}' for number, name in enumerate(report['coverage_states']))}. "
        "Это состояние на срез, одно на пример.\n"
    )

    if report["limitations"]:
        out.append("\n## Ограничения\n")
        out.extend(f"- {item}" for item in sorted(report["limitations"])[:20])
        out.append("")

    if report["readiness"]["reasons"]:
        out.append("\n## Почему это диагностика\n")
        out.extend(f"- {item}" for item in report["readiness"]["reasons"])
        out.append("")

    return "\n".join(out)


def render_golden_md(examples: list[dict]) -> str:
    """
    Читаемые примеры: от семантики до координат batch.
    """

    out: list[str] = []

    out.append("# Читаемые примеры\n")

    out.append(
        "Примеры выбраны по устойчивым признакам, а не по именам клиентов: пустая история, "
        "клиент без анкеты, пример без целей, запись с исправлением, усечённый контекст и "
        "самая длинная история.\n"
    )

    names = {
        "empty_history": "пустая история",
        "no_profile": "клиент без анкеты",
        "no_targets": "пример без целей",
        "corrected_event": "есть исправленная запись",
        "truncated": "контекст усечён",
        "longest": "самая длинная история",
    }

    for example in examples:

        out.append(f"\n## {names.get(example['trait'], example['trait'])}\n")

        out.append(
            f"Клиент {example['client_id']}, группа {example['group']}, срез "
            f"{example['cutoff'][:10]}, вес {example['weight']:.4f}.\n\n"
        )

        out.append(
            _md_table(
                [
                    ["событий в примере", example["n_events"]],
                    ["токенов", example["n_tokens"]],
                    ["значений", example["n_values"]],
                    ["событий периода целей", example["n_eligible_events"]],
                    ["анкета", example["profile_state"] + f" ({example['profile_tokens']} токенов)"],
                    ["возраст истории, дней", example["history_age_days"]
                     if example["history_age_days"] is not None else example["history_age_reason"]],
                ],
                ["что", "сколько"],
            )
        )

        if example["events"]:

            out.append("\nСобытия примера:\n\n")

            out.append(
                _md_table(
                    [
                        [
                            row["event_index"],
                            row["event_type"],
                            row["event_time"][:16],
                            row["n_tokens"],
                            row["eligible"],
                            row["selection_reason"],
                        ]
                        for row in example["events"]
                    ],
                    ["номер", "тип", "время", "токенов", "цель", "час известен", "почему взято"],
                )
            )

        if example["excluded"]:

            out.append("\nЧто осталось за границей контекста:\n\n")

            out.append(
                _md_table(
                    [
                        [row["event_type"], row["event_time"][:16], row["eligible"],
                         row["exclusion_reason"]]
                        for row in example["excluded"]
                    ],
                    ["тип", "время", "была бы целью", "причина"],
                )
            )

        batch = example["batch"]

        out.append("\nТот же пример в координатах batch из одного примера:\n\n")

        out.append(
            _md_table(
                [
                    ["строк событий", batch["rows"]],
                    ["ширина", batch["width"]],
                    ["после выравнивания", batch["padded_tokens"]],
                    ["настоящих токенов", batch["real_tokens"]],
                    ["допустимых целей", batch["target_candidates"]],
                    ["токенов анкеты", batch["profile_tokens"]],
                ],
                ["что", "сколько"],
            )
        )

    return "\n".join(out)


def render_measure_md(report: dict) -> str:
    """
    Измерение длин до выбора бюджетов.
    """

    out: list[str] = []

    out.append(f"# Длины историй: группа {report['group']}\n")

    out.append(
        f"Срез {report['cutoff'][:10]}, клиентов {report['clients']}, "
        f"событий {report['events']}, токенов {report['tokens']}.\n"
    )

    out.append("\n## Распределения\n")

    out.append(
        _md_table(
            [
                [name, value["p50"], value["p90"], value["p95"], value["p99"], value["max"]]
                for name, value in sorted(report["lengths"].items())
            ],
            ["что", "p50", "p90", "p95", "p99", "максимум"],
        )
    )

    out.append(
        "\nЛимит событий и лимит токенов это разные настройки: сто экранов приложения и "
        "сто кредитных событий стоят модели по-разному.\n"
    )

    out.append("\n## Сколько клиентов затронет предел\n")

    out.append(
        _md_table(
            [
                [item["limit"], item["events_above"], item["tokens_above"]]
                for item in report["candidates"]
            ],
            ["предел", "клиентов выше по событиям", "клиентов выше по токенам"],
        )
    )

    out.append("\n## Важные старые события\n")

    out.append(
        _md_table(
            [[name, count] for name, count in sorted(report["milestones"].items())],
            ["тип", "событий"],
        )
    )

    out.append(
        "\nЭто кандидаты в резерв вех: при усечении они переживают отбор раньше обычных "
        "событий той же давности.\n"
    )

    return "\n".join(out)


def render_check_md(report: dict) -> str:
    """
    Итоговая проверка собранного набора.
    """

    out: list[str] = []

    out.append(f"# Проверка набора {report['dataset_id']}\n")

    out.append(
        f"Проверок {len(report['checks'])}, нарушений {report['violations']}. "
        f"Вердикт: {'набор исправен' if report['ok'] else 'НАБОР НЕИСПРАВЕН'}.\n"
    )

    out.append("\n## Что проверено\n")

    out.append(
        _md_table(
            [[item["name"], item["ok"], item["detail"]] for item in report["checks"]],
            ["проверка", "в порядке", "подробности"],
        )
    )

    if report.get("recomputed"):

        out.append("\n## Пересчёт из семантики\n")

        out.append(
            _md_table(
                [
                    [item["sample_id"], item["events"], item["tokens"], item["identical"]]
                    for item in report["recomputed"]
                ],
                ["пример", "событий", "токенов", "совпал"],
            )
        )

        out.append(
            "\nПример собран заново из смыслового слоя на ту же дату и сверен с записанным "
            "массив за массивом.\n"
        )

    if report.get("batch"):

        batch = report["batch"]

        out.append("\n## Смешанный batch\n")

        out.append(
            _md_table(
                [
                    ["примеров", batch["samples"]],
                    ["строк событий", batch["rows"]],
                    ["ширина", batch["width"]],
                    ["пустых историй", batch["empty"]],
                    ["настоящих токенов", batch["real_tokens"]],
                    ["допустимых целей", batch["target_candidates"]],
                ],
                ["что", "сколько"],
            )
        )

    return "\n".join(out)


__all__ = [
    "render_check_md",
    "render_contract_md",
    "render_golden_md",
    "render_measure_md",
    "render_report_md",
]
