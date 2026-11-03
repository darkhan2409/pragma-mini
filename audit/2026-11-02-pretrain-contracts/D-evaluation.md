# Аудит D: контракт оценки PRAGMA [USR] ↔ CatBoost (churn_active90)

Дата: 2026-11-02. Только чтение: ни один файл репозитория, `data/` и `churn_baseline/reports` не изменён;
пайплайн, обучение, embed и probe не запускались. Эталон: `data/13_downstream/ref-2026-10-09`,
`churn_baseline/reports/*`, `churn_baseline/reports/plus_usr/ref-2026-10-09/*`.

## Итог

На эталоне контракт соблюдён. Момент T один, клиенты и метка одни и те же, строки соединяются
по `client_id`, всё обучение и все пороги берутся только с train. Расхождения, которое сдвигало бы
0.570 / 0.452 / 0.586, не найдено: все три числа заново посчитаны из `predictions.parquet` и
совпали с отчётом.

Найдено:

- **1 подозрение.** Доверительный интервал и временный шумовой пол не учитывают случайность
  обучения голов. Это влияет на то, как читать 0.586 против 0.570 и порог 0.005 для plus_usr.
- **5 подтверждённых, но латентных дыр в lineage.** Сейчас ни одна не срабатывает: это проверено
  на файлах. Но каждая позволила бы пробе молча сравнить несогласованную пару.

## Что проверено на реальных файлах

Скрипты лежат в `scratchpad/pretrain-audit/check_*.py`.

| Проверка | Результат |
|---|---|
| T обеих функций `cutoff` | Совпадают побитно как моменты для всех групп. train 2025-12-31T19:00Z = 2026-01-01 00:00+05; val 2026-02-28T19:00Z = 2026-03-01 00:00+05; test 2026-06-30T19:00Z = 2026-07-01 00:00+05 |
| T в строках churn | train_rows: только 2025-12-31T19:00Z; eval_rows: только 2026-02-28T19:00Z |
| T в векторах ref | meta и каждая строка: train 2025-12-31T19:00Z, val 2026-02-28T19:00Z |
| Строки catboost и plus_usr | val: 1366 = 1366, train: 7689 = 7689. Те же `client_id` в том же порядке (отсортированы, без дублей), метки равны строка в строку |
| `predictions.parquet` | 3 набора по 1366 строк. `client_id` у всех трёх равны eval_rows, `y` равен `churn` |
| Метрики заново | PR-AUC: catboost 0.5705, usr 0.4520, catboost_plus_usr 0.5862. ROC-AUC: 0.9674 / 0.9419 / 0.9671. Совпадает с report.json и metrics.json |
| Прогнозы | score catboost в predictions равен score eval_rows; score plus в predictions равен score plus eval_rows |
| Покрытие векторами | train: 8498 векторов ⊇ 7689 строк; val: 1493 ⊇ 1366. Ни одной строки без вектора, повторов клиентов нет |
| Join usr по id (здравый смысл) | Spearman(score usr, gap_seconds того же клиента) = 0.30 (у catboost 0.34). После перестановки −0.04 |
| inner split | Одинаков у catboost и plus_usr: inner_train 6151/315, inner_holdout 1538/79 |
| sha строк | `eval_rows`/`train_rows` равны `*_sha256` в metrics.json, и у бейзлайна, и у plus_usr |
| Источники plus_usr | `sources` plus_usr равны `sources` бейзлайна |
| 02 против 01 | Число строк: train 21 839 587 = 21 839 587, val 4 813 343 = 4 813 343. Период в метаданных 02 равен окну. mtime: 01/val 11:03 → 02/val 11:12 → векторы 13:49–13:51 |
| События ровно на границах | Ровно в val T: 13 событий (12 `installment_due`, 1 `transfer_in`). В продолжении ровно в train T: 95, ровно в T+60д: 76. В val T+60д: 0. Действий клиента среди них **нет** |
| Популяция = act_count_90>0 | Выполняется на train (8387) и val (1465) |
| Отбор истории (`recent`) | Обрезано 585 из 7689 строк train (2 ушедших). Минимальный оставшийся охват 249 дней, так что 90-дневные окна целы |
| Сдвиг [USR] между train и val | Средний \|стандартизованный сдвиг средних\| 0.048 (максимум 0.18), отношение sd 0.993, нормы равны. Log-loss usr: OOF на train 0.1255, на val 0.1246 |

Тесты: `tests/test_downstream.py` на CPU — 34 прошли, exit 0. `churn_baseline/tests/test_build.py`,
`test_train.py`, `test_plus_usr.py`, `test_activity.py` — 138 прошли.

---

## Находки (по влиянию на опубликованные числа)

### 1. Интервал Δ и шумовой пол не учитывают случайность обучения голов — ПОДОЗРЕНИЕ

Механизм подтверждён кодом, величина не измерена.

**Стороны**
- `src/downstream/probe.py:258-289` (`paired`). Bootstrap переразыгрывает только клиентов val при
  фиксированных прогнозах. Прогнозы — результат одного обучения каждой головы.
- `churn_baseline/churn/train.py:52-66`. `random_seed: SEED` = 42; CatBoost обучается один раз
  (`train.py:164-177`).
- `churn_baseline/churn/plus_usr.py:132`. Тот же `fit_task` с тем же seed, но на 116 + 128 колонках.
  Это другое дерево розыгрышей, поэтому разница catboost_plus_usr − catboost включает собственный
  шум CatBoost.
- `src/downstream/probe.py:175`. Фолды LR зависят от `--seed`. У выпуклой LR вклад мал.
- `README.md:1559-1568`. Временный пол: usr 0.025, catboost_plus_usr **0.005**. Это одна разность
  \|Δ\| одной пары (B0 против N) на старых данных до Generator V1.

**Почему подозрение.** Прежняя scratch-диагностика 2026-10-09 шла ещё при ранней остановке по PRAUC,
здесь она не повторялась:
- best_iteration CatBoost между seed: 230 / 404 / 412;
- при 231–1200 деревьях: plus_usr на val 0.55–0.60, catboost 0.56–0.58.

Разброс по seed при нынешней остановке по Logloss не измерен.

**Покрытие тестами.** Нет. `test_paired_bootstrap_is_zero_for_the_same_scores_and_positive_for_a_better_one`
проверяет только одинаковые и заведомо лучшие прогнозы.

**Что сдвинет.** Точечные значения не меняются. Расширится интервал
Δ catboost_plus_usr − catboost = +0.013 [−0.064, +0.086]. Разница 0.586 против 0.570 (+0.016)
может целиком уложиться в разброс CatBoost по seed. В правиле принятия Δ plus_usr > 0.005 может
пройти по шуму; частично страхует условие «log-loss не выше». Для usr (−0.120 [−0.209, −0.022])
шумит в основном эталонная сторона, CatBoost. Знак вывода вряд ли изменится.

**Самая дешёвая проверка.** Только по команде владельца, это обучение. `fit_task` для catboost и
plus_usr с `random_seed` 43–47 на готовых `features.parquet` и векторах ref: минуты CPU, без
пересборки. Сравнить разброс PR-AUC на val и разброс Δ. Если разброс ≥ 0.01, есть два выхода:
- усреднять прогнозы по seed в обоих сценариях;
- брать пол plus_usr из этого разброса, а не из одной пары.

### 2. Векторы не привязаны к сборке 02, по которой построен вход — ПОДТВЕРЖДЕНО (латентно)

**Стороны**
- `src/downstream/embed.py:148-160, 144`. `raw_record` в момент съёма читает manifest
  **01_raw** и записывает его sha в meta векторов.
- `src/downstream/at_cutoff.py:117` → `src/preprocessing/read.py:159`. Вход на T на деле читается
  из **02_preprocessed**/<group>/events.parquet.
- `src/preprocessing/canonical/build.py:97-100`. В метаданных 02 лежат только `period_start` и
  `period_end`, отпечатка 01 нет.
- Для train `embed.py:72-81` сверяет с чекпойнтом digest `05_dataset/train`. Но `at_cutoff` не
  читает 05: он собирает вход заново из 02/train. 02, пересобранная после 05, эту сверку проходит.
- `src/downstream/probe.py:114-121` сверяет записанный sha 01 с текущим 01.

**Покрытие тестами.** `test_vectors_of_another_export_are_refused` ловит только смену 01 после
съёма. Устаревшую или чужую 02 при текущем 01 не ловит ничто.

**Что сдвинет.** Сейчас ничего: 02 согласована с 01 (строки, период, порядок mtime, см. таблицу).
Если 01 перегенерировать без пересборки 02, а `client_id` совпадут (тот же seed, новый код
генератора), векторы опишут другие истории, а проба примет их. Молча сдвинутся 0.452 и 0.586.

**Исправление.** В препроцессинге записывать sha событий и анкеты из manifest в метаданные 02.
`raw_record` должен брать sha оттуда, а не из 01. Можно дополнительно записывать в meta векторов
digest файла 02 по группе.

### 3. Handcrafted X у catboost и catboost_plus_usr не отпечатан — ПОДТВЕРЖДЕНО (латентно)

**Стороны**
- `churn_baseline/churn/sources.py:84-101` и `train.py:282-294`: `sources` содержат только sha RAW,
  T и окно метки. Отпечатка `data/<group>/features.parquet` и версии кода признаков нет.
- `plus_usr.py:157-172`: то же самое.
- `src/downstream/tasks.py:131-140`: сверяются только тег, путь чекпойнта, записи групп векторов и
  `sources`.
- `src/downstream/probe.py:355-361`: сверяются только `client_id` и метки.

**Покрытие тестами.** `test_catboost_plus_usr_of_other_vectors_or_sources_is_refused` и
`test_catboost_plus_usr_on_other_clients_is_refused` проверяют источники, векторы, клиентов и
метки, но не X.

**Что сдвинет.** Сейчас ничего: `features.parquet` 13:41/13:42 старше обоих обучений
(catboost 14:14:56, plus_usr 14:15:17), признаков в обоих 116. Но если перезапустить `churn.build`
с изменённым кодом признаков на той же выгрузке между `churn.train` и `churn.plus_usr` (или
переобучить только бейзлайн), то Δ catboost_plus_usr − catboost (сейчас 0.586 против 0.570)
смешает эффект [USR] со сменой X. Изменение «действия клиента» отловит сверка меток, изменение
только признаков не отловит ничто.

**Исправление.** Писать sha `features.parquet` по группам в оба `metrics.json` и сравнивать его в
`tasks.plus_usr`.

### 4. Строки и порог CatBoost не сверяются между собой при чтении — ПОДТВЕРЖДЕНО (латентно)

**Стороны**
- `src/downstream/tasks.py:171-186`: строки и score читаются из `eval_rows`/`train_rows.parquet`.
- `tasks.py:98-105`: порог читается из `metrics.json`.
- `tasks.py:145-155`: так же для plus_usr.
- `churn_baseline/churn/train.py:278-279, 288-289` и `plus_usr.py:154-169` пишут
  `eval_rows_sha256` и `train_rows_sha256`, но downstream их не проверяет. Parquet пишется раньше
  metrics.json: если запуск оборвётся между этими записями, останутся новые строки при старом пороге.

**Покрытие тестами.** Нет.

**Что сдвинет.** Сейчас ничего: sha совпадают (проверено). При рассинхроне сдвинулись бы F1,
precision и recall, а при подменённых строках и PR-AUC catboost и plus_usr.

**Исправление.** Сравнивать sha файлов строк с `metrics.json` в `churn_rows` и `plus_usr`.

### 5. Чекпойнт векторов записан изменяемым путём без digest — ПОДТВЕРЖДЕНО (латентно)

**Стороны**
- `src/downstream/embed.py:85`: записывается `{"checkpoint": str(checkpoint), "epoch": ...}`.
  В ref это `data/12_train/best_checkpoint.pt`.
- `src/downstream/tasks.py:131`: идентичность plus_usr сверяется сравнением строки пути и записей
  групп; повторный съём ловится по полю `seconds`.
- `src/downstream/probe.py:88-147`: чекпойнт не проверяется вовсе. По замыслу векторы и есть
  предмет оценки.

**Покрытие тестами.** Нет.

**Что сдвинет.** Сейчас ничего: `data/12_train/best_checkpoint.pt` от 2026-10-09 13:15, векторы
от 13:49–13:51; у файла 2 жёсткие ссылки, вторая в `data/runs/ref-2026-10-09/`. Риск в финальной
оценке. `embed --final-test` по пути из meta после любого `src.mlm.train` без `--out` (он очищает
`data/12_train`) снимет train, val и test **другой** модели. Проба примет их: всё сверено между
собой, просто модель не та, что выбрана на val.

**Исправление.** Писать sha256 файла чекпойнта в meta векторов. Финальный съём делать по пути
`data/runs/<тег>/best_checkpoint.pt`.

### 6. Предел кусков на значение при съёме берётся из умолчаний кода — ПОДТВЕРЖДЕНО (латентно, низкий)

**Стороны**
- `src/downstream/at_cutoff.py:93`: `TokenizerConfig.load(None).max_pieces_per_value`.
- Отбор истории и точка отсчёта сверяются с `05_dataset/train/meta.json` (`at_cutoff.py:98-115`),
  а предел кусков нет: его нет ни в `data/03_vocab/*`, ни в `data/04_tokenized/*/meta.json` (grep).

**Покрытие тестами.** `test_input_at_the_group_cutoff_is_what_training_reads` собирает обе стороны
с умолчаниями, поэтому расхождение не поймает.

**Что сдвинет.** Сейчас ничего: эталонный конвейер шёл с умолчаниями (`data/runs/full-2026-10-09/chain.sh`,
`python -m src.pipeline` без конфига). Если 04 собрать с не-умолчательным `TokenizerConfig`, вход
val и test на T закодировался бы иначе, чем при обучении, и это прошло бы молча (0.452, 0.586).

**Исправление.** Записывать `max_pieces_per_value` в meta 04/05 и сверять его в `ClientsAtCutoff`
так же, как `context`.

---

## Ответы на вопросы

### 1. Тот же T

**Да.**
- PRAGMA: `src/downstream/settings.py:67-91`. Для train это `windows.train.final_cutoff`; для val и
  test — местная полночь 1-го числа месяца, в котором лежит `final_cutoff − 60д`. Пояс
  фиксированный UTC+5 (`src/preprocessing/settings.py:214-219`), окна в `settings.py:178-193`.
- Бейзлайн: `churn_baseline/churn/config.py:74-95`. Для train это `period_end` из manifest; для
  остальных `month_start(period_end − HORIZON)` в смещении строки +05:00.
- Источники разные (окна в коде против manifest RAW), на текущих данных моменты совпадают для всех
  трёх групп.
- Страховка во время выполнения: `tasks.py:176-184` (T строк), `plus_usr.py:60-68` (T векторов и
  строк), `probe.py:127-143` (T векторов).

**Границы.**

| Где | Правило | Файл |
|---|---|---|
| Признаки | `t < T` | `features.py:65` |
| Окна признаков | `age ≤ w`, то есть `T − w ≤ t` | `features.py:80` |
| Популяция | `T − 90д ≤ t < T` | `target.py:55-56` |
| Метка val | `(T, T+60д]` | `target.py:60` |
| Метка train | `(T, T+60д]` | `sources.py:125` |
| Вход PRAGMA | `< T` | `read.py:317-318`, страховка `at_cutoff.py:190-199` |
| Откат анкеты | `≥ T` с обеих сторон | `profile.py:100`; `profile_state.py:354` |
| Вехи | `< T` | `profile.py:140`; `profile_state.py:351` |
| Записи о работе | `< T` | `profile.py:130`; `profile_state.py:420` |

Событие ровно в T не попадает никуда, на всех трёх сторонах одинаково. На данных в T и в T+60д нет
ни одного действия клиента.

### 2. Те же клиенты и строки

**Да, join по `client_id`.**
- Популяция определяется только в бейзлайне (`build.py:75-76`, `target.py:80-86`). PRAGMA берёт
  строки готовыми (`tasks.py:158-186`) и требует вектор у каждой (`probe.py:337-342`).
- usr: `probe.py:344-345`, `usr_matrix` через `.loc[rows.index]` (`probe.py:155-160`).
- plus_usr при обучении: `merge(on="client_id", validate="one_to_one")` и сверка порядка
  (`plus_usr.py:78-81`).
- plus_usr в пробе: `set_index("client_id")` (`tasks.py:151-154`), затем `index.equals` и сверка
  меток (`probe.py:355-361`).
- vs_baseline: `reindex` по `client_id` и сверка меток (`probe.py:428-431`).
- Позиционного соединения нет нигде. На данных совпадение полное (см. таблицу).

### 3. Горизонт информации

**Совпадает, утечки нет.**
- CatBoost: `Past` (`features.py:64-74`), анкета на T (`profile.py:87-118`), снимочные поля
  исключены (`profile.py:25-27`).
- PRAGMA: `Group._build` (`read.py:298-360`), `profile_at` (`profile_state.py:322-430`), исключения
  `SHORTCUT_FIELDS` и `FROM_LIFELONG`, страховка `at_cutoff.py:190-199`.
- 02 повторяет RAW 1:1 (число строк равно): только типизация и порядок.
- Снимок остатка в 23:55 лежит до T и есть с обеих сторон (`features.py:271-277`; событие во входе).
  Это законный факт прошлого, а не окно метки.
- Продолжение читается только для метки (`sources.py:48-127`). Признаки строятся из выгрузки
  (`build.py:52`). Этапы PRAGMA продолжения не видят (тест
  `test_the_label_only_continuation_is_outside_every_pragma_stage`). Начало продолжения равно T,
  оно покрывает окно метки (`sources.py:72-76`), его прошлое сверено с выгрузкой (`sources.py:69-70`).
- Вход val **не** из 05: `embed.py:103, 214` строит его через `at_cutoff` на `settings.cutoff`;
  meta и строки ref стоят на 2026-02-28T19:00Z. `05_dataset/val` (до 2026-04-30T19:00Z) в съёме
  не участвует.
- Асимметрия без утечки: из-за отбора `recent` у 7.6% строк train CatBoost видит всю историю
  (`act_days_since_first`, `act_count_all`, `insurance_premium_count`, `consent_withdrawn`), а
  PRAGMA только хвост длиной не меньше 249 дней.

### 4. Определение «действия клиента»

**Одна функция:** `churn_baseline/churn/activity.py:73-93`. Её используют:
- признаки `act_*`: вызов в `build.py:53`, затем `features.py:148-180` через `past.action`;
- популяция и `active90`: `target.py:55-56`;
- метка val: `target.py:60`;
- метка train: `sources.py:125`.

Своего определения в downstream нет (`tasks.py`, комментарий в шапке). На данных
`act_count_90 > 0 ≡ active90`.

### 5. Гигиена обучения и оценки

- **LR и скейлер** обучаются только на train: pipeline внутри `GridSearchCV` (`probe.py:163-178, 347`).
- **CatBoost** обучается только на train: `train.py:150-193, 235`, `plus_usr.py:132`. val только
  предсказывается (`train.py:238-246`).
- **Holdout ранней остановки** берётся из train: `train.py:141-147, 161-165`.
- **Пороги стоят там, где сказано в коде:**
  - CatBoost: max F1 на inner_holdout остановленной модели (`train.py:168`), затем перенос на
    модель, переобученную на всём train (`train.py:175-177`);
  - LR: max F1 по OOF на тех же фолдах, что выбирали C (`probe.py:215-226, 353`).

  В обоих случаях одна и та же выборка служит и для остановки или выбора C, и для порога, а сам
  порог переносится на переобученную модель. Это сдвигает только F1, precision и recall
  (0.599 / 0.470 / 0.633), но не PR-AUC, ROC-AUC и log-loss. Поведение задокументировано и
  одинаково на обеих сторонах.
- **Bootstrap парный.** `default_rng(seed)` заново в каждом вызове (`probe.py:265`), индексы по
  строкам val, упорядоченным по клиенту (`probe.py:271-279`). Выборки для usr, plus_usr и
  vs_baseline одни и те же.
- **На val ничего не подгоняется.** In-sample прогнозы CatBoost для train из `train_rows` проба не
  использует (`probe.py:349-351`). val — набор выбора по замыслу (`settings.py:36-38`, README §13).

### 6. Lineage

**Проба отказывает**, если:
- T строк не равен T группы (`tasks.py:176-184`);
- RAW признаков, анкеты или метки, либо продолжение не текущие (`tasks.py:35-95`);
- векторы сняты с другого 01 (`probe.py:114-121`);
- векторы сняты не на T: в meta или в любой строке (`probe.py:127-143`);
- в векторах повторяется клиент (`probe.py:139-140`);
- у строки нет вектора (`probe.py:337-342`);
- plus_usr обучен на других векторах или источниках (`tasks.py:131-143`);
- у plus_usr другие клиенты или метки (`probe.py:355-361`);
- у базовой модели другие строки или метки (`probe.py:428-431`).

Пересъём одной группы под тем же тегом перезаписывает meta без остальных групп, и проба отказывает.

**Молча прошли бы:**
- устаревшая 02 (находка 2);
- другой X CatBoost (находка 3);
- рассинхрон строк и порога (находка 4);
- веса чекпойнта, подменённые по тому же пути (находка 5);
- не-умолчательный предел кусков токенизатора (находка 6).

---

## Чистые проверки

- T одинаков для train, val и test, моменты совпадают — `src/downstream/settings.py:67-91` ↔ `churn_baseline/churn/config.py:74-95`.
- Пояс фиксированный +05:00 с обеих сторон — `src/preprocessing/settings.py:214-219` ↔ `churn_baseline/churn/config.py:47-48`, `raw.py:108-109`.
- Расхождение T строк и группы отвергается — `src/downstream/tasks.py:176-184` (тест `test_churn_tasks_are_the_baseline_rows_of_the_current_sources`).
- Событие ровно в T не идёт ни в признаки, ни в метку, ни во вход — `features.py:65`, `target.py:60`, `sources.py:125`, `read.py:317`.
- На границах T и T+60д нет действий клиента — проверено на `01_raw/val` и `churn_baseline/data/future/train`.
- Окно 90 дней одинаково в популяции и в `act_*` — `target.py:56` ↔ `features.py:80`; на данных act_count_90>0 ≡ active90.
- Откат анкеты по `profile_change` ≥ T одинаков — `churn_baseline/churn/profile.py:100` ↔ `src/preprocessing/profile_state.py:354`.
- Вехи и записи о работе берутся строго до T с обеих сторон — `profile.py:130,140` ↔ `profile_state.py:351,420`.
- Снимочные производные поля (contracts_count, holds_*, credit_*, relationship_months) исключены с обеих сторон — `profile.py:25-27` ↔ `profile_state.py` (`EXCLUDED_FIELDS`).
- Вход на T строго из прошлого и с охранной проверкой — `src/downstream/at_cutoff.py:190-199`.
- Вход val строится через at_cutoff, а не из 05/val — `src/downstream/embed.py:103,214`; ref meta: val cutoff 2026-02-28T19:00Z.
- Продолжение читается только для метки — `churn_baseline/churn/sources.py:48-127`, `build.py:52`.
- Одна функция «действия клиента» — `churn_baseline/churn/activity.py:73-93`.
- Популяция берётся только из бейзлайна, вектор обязателен — `src/downstream/probe.py:337-342`.
- usr соединяется с метками по `client_id` — `src/downstream/probe.py:155-160, 344-345`.
- plus_usr соединяется по `client_id` (one_to_one и порядок) — `churn_baseline/churn/plus_usr.py:78-81`.
- В пробе у plus_usr те же клиенты и метки — `src/downstream/probe.py:355-361`; на данных 1366/7689 совпали.
- vs_baseline соединяется по `client_id` со сверкой меток — `src/downstream/probe.py:428-431`.
- LR и скейлер обучаются только на train, скейлер внутри фолда — `src/downstream/probe.py:163-178` (тесты `test_the_scaler_is_fitted_inside_each_fold`, `test_the_chosen_C_depends_on_train_only`).
- CatBoost и plus_usr обучаются только на train — `churn_baseline/churn/train.py:150-193`, `plus_usr.py:132`.
- Holdout ранней остановки из train, split одинаков у обоих CatBoost — `train.py:141-147`; на данных 6151/315 и 1538/79.
- Ранняя остановка plus_usr по Logloss реально в артефакте — `plus_usr/ref-2026-10-09/metrics.json`: `eval_metric` Logloss, best_iteration 437.
- Пороги только от train — `train.py:168`, `probe.py:215-226` (тест `test_probe_thresholds_come_from_train_only`).
- Bootstrap парный по клиенту, выборки одни и те же — `src/downstream/probe.py:265-279`.
- In-sample score CatBoost на train в пробу не попадает — `src/downstream/probe.py:349-351`.
- Порядок в embed: строка векторов i — клиент i — `src/mlm/inputs.py:350-375`, `embed.py:128-135` (тест `test_readouts_are_the_client_embedding_and_its_events`).
- Сдвига [USR] между train и val нет (энкодер видел клиентов train): mean \|d\| 0.048, log-loss OOF 0.1255 против val 0.1246 — согласуется со спайком 2026-11-02.
- Словарь обучен только на train — `src/tokenization/settings.py:386`.
- Обрезка истории не задевает 90-дневные окна: у 585 обрезанных строк train охват не меньше 249 дней — `05_dataset/train/meta.json` (`policy: recent`).

## Статус уже известных пунктов

| Пункт | Статус |
|---|---|
| Окно метки train до 2026-03-02 против val T 2026-03-01 (пересечение 1 день) | Без изменений: `metrics.json` `sources.train.target_window` |
| Окно маскирования MLM val [2026-01-01, 2026-05-01) накрывает окно метки val | Без изменений: `05_dataset/val/meta.json` |
| 77 положительных в val | Без изменений |
| Ранняя остановка plus_usr по Logloss | Внедрена и отражена в эталонных артефактах (best_iteration 437, у бейзлайна 228) |
| Ушедшие val «молчат» дольше ушедших train | Здесь заново не проверялось |
