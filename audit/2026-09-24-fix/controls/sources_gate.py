"""
Контроль механизма привязки прогона к состоянию кода.

    python audit/2026-09-24-gen/controls/sources_gate.py

Проверяется, что сверка отпечатка действительно срабатывает, а не
проходит всегда. `src/` при этом НЕ трогается: расхождение
подаётся подменой ОЖИДАЕМОГО отпечатка, а не подменой файла.

Три случая:

  1. положительный  — текущее дерево против своего же отпечатка;
  2. изменённый файл — ожидаемый хеш одного файла испорчен;
  3. пропавший файл — файл убран из ожидаемого отпечатка.

В случаях 2 и 3 сверка обязана отказать и назвать этот файл.
"""

from __future__ import annotations

import sys
from pathlib import Path

AUDIT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(AUDIT / "harness"))

from sources import SourcesChanged, snapshot, state_id, stored, verify  # noqa: E402


VICTIM = "src/generator/finance/ledger.py"


def case(name: str, expected: dict, must_fail: bool, mention: str | None) -> bool:

    try:
        identifier = verify(name, expected)

    except SourcesChanged as error:

        if not must_fail:
            print(f"[ОШИБКА КОНТРОЛЯ] {name}: отказ там, где его быть не должно\n{error}")
            return False

        text = str(error)

        if mention is not None and mention not in text:
            print(f"[ОШИБКА КОНТРОЛЯ] {name}: отказ есть, но файл {mention} не назван")
            return False

        print(f"[ok] {name}: отказано по нужной причине, назван {mention}")
        return True

    if must_fail:
        print(f"[ОШИБКА КОНТРОЛЯ] {name}: сверка прошла там, где обязана была отказать")
        return False

    print(f"[ok] {name}: сверка прошла, состояние {identifier}")
    return True


def main() -> int:

    base = stored()
    current = snapshot()

    if VICTIM not in current:
        print(f"[ОШИБКА КОНТРОЛЯ] в дереве нет {VICTIM}: контроль не на что ставить")
        return 1

    results = []

    # 1. Положительный контроль.
    results.append(case("контроль: дерево как есть", base, False, None))

    # 2. Изменённое содержимое файла.
    changed = dict(base)
    changed[VICTIM] = "0" * 64
    results.append(case("контроль: изменённый файл", changed, True, VICTIM))

    # 3. Пропавший файл.
    missing = dict(base)
    missing.pop(VICTIM)
    results.append(case("контроль: пропавший файл", missing, True, VICTIM))

    # 4. Идентификатор состояния обязан меняться вместе с деревом.
    if state_id(base) == state_id(changed):
        print("[ОШИБКА КОНТРОЛЯ] state_id не изменился при изменении хеша файла")
        results.append(False)
    else:
        print("[ok] state_id меняется вместе с содержимым дерева")
        results.append(True)

    ok = all(results)

    print()
    print("контроль пройден" if ok else "КОНТРОЛЬ НЕ ПРОЙДЕН")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
