# Проверка исправлений генератора — 2026-09-24

Каталог отдельный от `audit/2026-09-24-gen/`: тот аудит описывает
состояние кода ДО правок и не переписывается. Здесь лежат правки,
их проверки и регрессионная генерация.

## Состояние кода

Отпечаток `src/` и `reference/` записан в `state/sources.json`,
идентификатор — в `state/code_state.txt`. Каждый прогон сверяет
его ДО и ПОСЛЕ работы и пишет в свой `run_record.json`. Отпечаток
менялся по ходу работы (правки вносились не за один раз), поэтому
в отчёте у каждого результата назван свой `code_state`.

    python audit/2026-09-24-fix/state/fingerprint.py --verify

## Что и чем проверяется

| проверка | команда |
|---|---|
| F-1 общий префикс при продлении окна | `checks/prefix.py --short runs/A --long runs/B --boundary ГГГГ-ММ-ДД` |
| F-1 планы не зависят от границы | `checks/plan_probe.py --clients N --seed S --end ДАТА --out …`, затем `checks/plan_diff.py --left … --right … --boundary ДАТА` |
| F-2 несколько выгрузок в одном процессе | `checks/inprocess.py --clients 48 --seed 100` |
| R2d, F-3, F-5 возобновление и целостность | `checks/resume.py` |
| F-12 ответ клиента не выгружается | `checks/f12.py --runs regression/train …` |
| F-7 границы розыгрыша | `checks/rng_bounds.py` |
| мошенничество: порядок эпизода | `checks/fraud.py` |
| путь от решения до наблюдаемого исхода | `checks/fraud_path.py --run s1-observed` |
| детерминизм | `checks/determinism.py --left A --right B` |
| деньги, зеркала переводов, общий баланс | `checks/money.py --name … --clients … --seed … --start … --end … --compare …` |
| кредиты | `checks/loans.py --run regression/train` |
| лента и время | `checks/tape.py --run regression/train --end 2026-01-01` |
| разделение групп и SPLIT-5 | `checks/split.py --groups regression/train regression/val regression/test` |
| граница RAW → препроцессинг | `checks/pipeline.py --run regression/train --group train` |
| словарь учится только на train | `checks/vocab_control.py --source regression` |
| этапы 05-08 и будущее во входах | `checks/stages.py --source regression` |

Вспомогательные пробы разбора: `key_probe.py` (поток ключей
розыгрыша), `lookahead_probe.py` (что видит решение о трате).
Их выводы в `evidence/` не хранятся: они снимались при
промежуточных состояниях кода и итоговыми результатами не
являются. Снимки планов (`plan_probe.py`) тоже не хранятся —
это полные дампы на полтора мегабайта; в `evidence/f1-plans.json`
лежит результат их сравнения, а сами снимки пересобираются
двумя командами из таблицы.

`checks/determinism.py`, `fraud.py`, `loans.py`, `pipeline.py`,
`rng_bounds.py`, `split.py`, `tape.py` и `money.py` — копии
проверок прежнего аудита. В копиях изменено: путь в строке
запуска; в `money.py` сверка зеркальных проводок (прежняя
требовала совпадения секунды и потому объявляла односторонними
все до одной) плюс общий баланс; в `split.py` SPLIT-5 считается
от `target_start`; в `rng_bounds.py` добавлен розыгрыш
максимального uint64 через настоящий путь, а горизонт сдвигается
внутри `PLANNING_END`. Оригиналы в `audit/2026-09-24-gen/` не
тронуты.

Тяжёлые каталоги прогонов (`runs/`) в репозиторий не идут —
`.gitignore` их исключает. `runs/stages-root` и
`runs/vocab-control-root` пересобираются своими командами из
таблицы выше.

## Регрессионная генерация

    python audit/2026-09-24-fix/harness/run_group.py --group train --workers 4

Пишет в `runs/regression/<группа>` по параметрам
`config.DATASETS`. `data/` не используется.

## Границы

Проверки говорят о содержимом проверенных прогонов и о названных
контрактах. Реализм по сравнению с настоящими клиентами здесь не
проверяется и подтверждённым не считается: внешних данных нет.
