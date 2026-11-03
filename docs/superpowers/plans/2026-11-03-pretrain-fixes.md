# Исправления предобучения PRAGMA — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** четыре переключателя (по умолчанию — поведение эталона) для кандидатов 1–4 и скрипт шага 0, готовые к прогонам по команде владельца.

**Architecture:** каждое изменение — поле конфига (`MlmConfig`, `MaskingConfig`, конфиг `fit`), выключенное по умолчанию; со значениями по умолчанию маска, поток и шаг побитно прежние. Конфиги кандидатов и процедура прогона — в `audit/2026-11-03-pretrain-fixes/`. Прогоны, пересборка данных и решения — вне плана, по команде.

**Tech Stack:** Python 3.14, PyTorch 2.13 (flash-attn, bf16), numpy, pyarrow, CatBoost (venv churn_baseline), pytest.

**Spec:** `docs/superpowers/specs/2026-11-03-pretrain-fixes-design.md`

## Global Constraints

- Со значениями по умолчанию новых полей маска, порядок, шаг и прямой проход побитно равны нынешним (CPU); чекпойнт эталона и конфиги без новых ключей грузятся с прежним поведением.
- Обучение, пересборка 03–07, embed, plus_usr, probe и шаг 0 — только по команде владельца. Коммиты и ветки — только по его команде; в задачах вместо коммита — прогон тестов и отчёт.
- Тесты не читают и не пишут `data/`; CUDA-тесты — `@pytest.mark.cuda` + `world.flash_ready()`, пропуск не считается пройденной проверкой.
- Ошибка, найденная тестом, чинится в коде; допуски не ослабляются.
- Статистики — только по train; ошибки данных не исправляются вымышленными значениями.
- Тексты, комментарии, ошибки — по-русски; комментарии объясняют причины и инварианты. README обновляется в той же задаче, что и поле.
- Группы кандидата 3: `(direction, transaction_amount)`, `(app_domain, firebase_screen, operation)`, `(campaign_code, offer, template, purpose)`, `(is_online, operation_channel)`, `(mcc, merchant_category, merchant_name)`.
- Не трогать работу LoRA (`src/downstream/lora.py`, `finetune.py`, их тесты) и `.superpowers/`.

## Review Focus

1. Resume посреди эпохи с `aux_per_client` и `grad_accum_steps > 1` — окна и шаги те же, что без остановки. Тест — задача 2.
2. `shuffle_order` с `Prefetch(workers=2)` — тот же поток клиентов, что с `workers=0`. Тест — задача 3.
3. `event_type_event_only` убирает только цели `event_type` value/key: цели и `[UNK]` остальных значений побитно те же. Тест — задача 4.
4. Поле группы отсутствует в событии или текст из нескольких кусков BPE — замыкание закрывает только присутствующие поля, текст целиком. Тест — задача 4.
5. `05/meta.json` без ключа `vocabulary` (набор эталона) читается; с другим отпечатком — отказ. Тест — задача 7.

---

### Task 1: Шаг 0 — разброс CatBoost по seed

**Files:**
- Create: `audit/2026-11-03-pretrain-fixes/catboost_seeds.py`

**Interfaces:**
- Consumes: `churn_baseline/churn/train.py` (`load`, `fit_task`, `evaluate`, `PARAMS["random_seed"]`, `holdout` с фиксированным 42), `churn_baseline/churn/plus_usr.py` (`with_usr`, путь сборки признаков `train_plus_usr`).
- Produces: CLI `python audit/2026-11-03-pretrain-fixes/catboost_seeds.py --tag ref-2026-10-09 [--seeds 43 44 45 46 47]` (запуск из `churn_baseline/` его venv, `PYTHONPATH` на корень). Печатает по seed PR-AUC val `catboost` и `catboost_plus_usr`, Δ, затем среднее, размах и стандартное отклонение Δ. Ничего не пишет.

- [ ] **Step 1:** реализовать: для каждого seed — `PARAMS["random_seed"] = seed`, `fit_task` на тех же строках train и признаках, что `churn.train` и `churn.plus_usr` (holdout не меняется); оценка val через `evaluate`. Seed 42 включить первой строкой как контроль: он обязан дать 0,570 и 0,586.
- [ ] **Step 2:** `churn_baseline/.venv/bin/python -m py_compile audit/2026-11-03-pretrain-fixes/catboost_seeds.py` и `--help` — без ошибок. Сам прогон — по команде владельца.

### Task 2: Кандидат 1 — E1′ среднее по клиентам окна

**Files:**
- Modify: `src/mlm/settings.py` (`MlmConfig`), `src/mlm/model.py` (`RecentTypes.forward`, `ModelOutput.aux`), `src/mlm/train.py` (цикл шага, `close_window`), `README.md` (раздел 12, нормировка)
- Test: `tests/test_aux_window.py`

**Interfaces:**
- Produces:
  - `MlmConfig.aux_per_client: bool = False` (в `as_dict`/`from_dict`).
  - `RecentTypes.forward(data, usr) -> Tensor` — квадраты ошибок `[B, len(RECENT_DAYS) + 1, types]`, fp32 (`F.mse_loss(..., reduction="none")`); `ModelOutput.aux` — этот тензор или `None`. Единственный потребитель — `train.py` (`:1235`, `:1375`).
  - Флаг выключен: `aux.mean()` по всем элементам — прежний путь побитно (`objective = loss + w·aux.mean()`, `(objective·count).backward()`, деление на `window_targets` в `close_window`).
  - Флаг включён: окно — список micro-batch'ей (`grad_accum_steps` штук или хвост эпохи) до прохода; `T = Σ client.n_targets`, `C = Σ len(clients)`; на micro-batch `backward(loss·count / T + w · aux.mean(dim=(1, 2)).sum() / C)`; `close_window` не делит; micro-batch без целей MLM даёт только E1′; телеметрия `aux` — среднее по клиентам. Границы окон считаются от начала эпохи, поэтому resume (`micro_batches_done`) попадает в те же окна.

- [ ] **Step 1: Тесты** (мир с типами событий — по образцу `head()` из `tests/test_recent_types.py` и `audit/2026-11-02-pretrain-contracts/scripts/test_aux_partition.py`; градиенты перехватываются у `clip_grad_norm_`):
  - `test_window_split_does_not_change_the_step` — `usr_aux_weight=0.5`, `aux_per_client=True`: одно окно (`grad_accum_steps=1`, большой бюджет) и то же окно из нескольких micro-batch'ей (малый бюджет, `grad_accum_steps = их число`) — градиенты в допуске `test_one_big_batch_and_several_micro_batches_train_the_same`.
  - `test_client_aux_weight_is_one_over_window_clients` — член E1′ шага равен `w · mean_по_клиентам(среднее ячеек клиента)` при любом составе micro-batch'ей.
  - `test_micro_batch_without_targets_still_trains_aux` — клиент без целей MLM даёт ненулевой градиент `recent.proj`.
  - `test_aux_off_is_the_previous_step` — флаг выключен: градиент шага `torch.equal` градиенту по прежней формуле, посчитанному в тесте.
  - `test_resume_mid_epoch_with_aux_per_client` (Review Focus 1) — `grad_accum_steps=3`, остановка посреди эпохи и `--resume`: веса и телеметрия шагов совпадают с прогоном без остановки (как в `tests/test_checkpoint_resume.py`).
- [ ] **Step 2:** `pytest tests/test_aux_window.py -v` — FAIL.
- [ ] **Step 3:** реализовать по Interfaces; README §12 — описание нормировки обоих режимов.
- [ ] **Step 4:** `pytest tests/test_aux_window.py tests/test_training_math.py tests/test_recent_types.py tests/test_checkpoint_resume.py -v` — PASS; `pytest -o addopts="" -q` — без регрессий.

### Task 3: Кандидат 2 — перемешивание порядка

**Files:**
- Modify: `src/mlm/settings.py`, `src/mlm/train.py` (`train_source`), `src/mlm/inputs.py` (`Source`), `README.md` (раздел 12)
- Test: `tests/test_shuffle_order.py`

**Interfaces:**
- Produces:
  - `MlmConfig.shuffle_order: bool = False`.
  - `Source.shuffle: int | None = None` — seed перестановки клиентов внутри группы строк; входит в `__getstate__`. При `None` поведение прежнее.
  - `Source.batch(index)` при `shuffle` переставляет клиентов группы `np.random.default_rng(stable_hash("order", shuffle, index) % 2**32).permutation(n)`; `Source.sizes()` даёт длины в том же порядке (на этом стоят число micro-batch'ей и `horizon`).
  - `train_source(config, masking, epoch)` при флаге: `seed = stable_hash("order", config.seed, epoch)`, `source.order = list(np.random.default_rng(seed % 2**32).permutation(source.count))`, `source.shuffle = seed`. Val не трогается.

- [ ] **Step 1: Тесты:**
  - `test_epochs_get_different_orders_and_an_epoch_repeats` — порядок клиентов эпох 1 и 2 различается; эпоха 1 дважды — тот же.
  - `test_client_mask_does_not_depend_on_order` — `value_ids`/`labels` каждого клиента одинаковы с перемешиванием и без.
  - `test_sizes_follow_the_shuffled_order` — `sizes()` и длины `clients()` совпадают поэлементно; `horizon` равен фактическому числу шагов.
  - `test_resume_mid_epoch_with_shuffle` — resume посреди эпохи даёт тот же поток.
  - `test_prefetch_workers_keep_the_shuffled_stream` (Review Focus 2) — `Prefetch(source, 2)` и `Prefetch(source, 0)` отдают клиентов в одном порядке.
  - `test_shuffle_off_is_the_previous_order` — флаг выключен: порядок файла.
- [ ] **Step 2:** `pytest tests/test_shuffle_order.py -v` — FAIL.
- [ ] **Step 3:** реализовать; README §12 — порядок прохода.
- [ ] **Step 4:** `pytest tests/test_shuffle_order.py tests/test_source_inputs.py tests/test_scheduler.py tests/test_checkpoint_resume.py -v` — PASS; `pytest -o addopts="" -q`.

### Task 4: Кандидат 3 — групповая маска и `event_type` только при event

**Files:**
- Modify: `src/masking/settings.py` (`MaskingConfig`), `src/masking/choose.py` (`choose`, новое `FieldGroups`), `src/mlm/inputs.py` (`Source` строит `FieldGroups` и передаёт в `choose`), `src/masking/report.py` (передаёт `FieldGroups` в `choose`), `README.md` (раздел о маске)
- Test: `tests/test_masking_groups.py`

**Interfaces:**
- Produces:
  - `MaskingConfig.field_groups: tuple[tuple[str, ...], ...] = ()`, `MaskingConfig.event_type_event_only: bool = False`; `validate()`: группа из одного ключа или ключ в двух группах — `ConfigError`; `as_dict`/`from_dict` (списки ↔ кортежи; отсутствие ключей — значения по умолчанию).
  - `@dataclass(frozen=True) class FieldGroups: of_key: np.ndarray  # key_id → номер группы, −1 вне групп; event_type: int  # key_id ключа event_type`.
  - `field_groups(config: MaskingConfig, key_vocab: dict[str, int]) -> FieldGroups | None` — `None`, если групп нет и флаг выключен; ключ не из словаря — `ConfigError` с именем.
  - `choose(group, row, config, weights=None, fields: FieldGroups | None = None) -> Selection` — все розыгрыши как сейчас; затем, только при `fields`:
    1. `event_type_event_only`: выбранные значения ключа `event_type` с причиной KEY или VALUE выбрасываются из `picked` вместе с их `unknown`; порча контекста ключа `event_type` (`spoiled`) не делается. Розыгрыши не перетягиваются — остальные значения не меняются.
    2. Замыкание: для каждого выбранного значения с причиной KEY/VALUE и `unknown = False`, чей ключ в группе, все значения ключей той же группы в том же событии, ещё не выбранные, добавляются в `picked` с той же причиной (при нескольких источниках — сильнейшей: меньший ненулевой код `REASONS`) и `unknown = False`. `picked` остаётся отсортированным.
  - `Source` и `masking_report` строят `FieldGroups` один раз из `key_vocab.json` текущего словаря.

- [ ] **Step 1: Тесты** (группы — из ключей синтетического мира; много клиентов и seed'ов):
  - `test_no_target_has_a_visible_group_partner` — у каждой цели value/key все поля её группы в том же событии — цели (`labels != −100`, во входе `[MASK]`).
  - `test_group_member_absent_or_multi_piece` (Review Focus 4) — событие без части полей группы закрывает только присутствующие; текстовое значение закрыто всеми кусками, метки на всех.
  - `test_event_type_is_a_target_only_by_event` — при флаге у `event_type` нет целей с причиной key/value, цели с причиной event есть.
  - `test_event_type_event_only_leaves_other_values_alone` (Review Focus 3) — без групп, флаг вкл/выкл: цели, причины и `[UNK]` всех значений, кроме `event_type`, побитно равны.
  - `test_groups_off_is_the_previous_mask` — `field_groups=()`, флаг выключен: `choose` с `fields=None` и без аргумента дают побитно одно и то же; существующие тесты маски проходят без изменений.
  - `test_field_groups_config_errors` — неизвестный ключ, ключ в двух группах, группа из одного ключа — `ConfigError`; `as_dict`/`from_dict` туда и обратно; словарь без новых ключей — значения по умолчанию.
- [ ] **Step 2:** `pytest tests/test_masking_groups.py -v` — FAIL.
- [ ] **Step 3:** реализовать; README — правило групп и `event_type`.
- [ ] **Step 4:** `pytest tests/test_masking_groups.py tests/test_masking.py tests/test_masking_windows.py tests/test_key_hiding.py tests/test_source_inputs.py -v` — PASS; `pytest -o addopts="" -q`.

### Task 5: Отчёт маски — цели с видимым партнёром

**Files:**
- Modify: `src/masking/report.py`
- Test: `tests/test_masking_groups.py` (дописать)

**Interfaces:**
- Consumes: Task 4 (`FieldGroups`, `choose(..., fields)`).
- Produces:
  - `dependent_pairs(group: str, row_groups: int = 8, threshold: float = 0.05) -> list[tuple[str, str, float]]` — пары `(B, A, H(B|A)/H(B))` ключей одного события с отношением ниже порога, по первым `row_groups` группам строк набора группы; значение — кортеж его токенов. Логика — `audit/2026-11-02-pretrain-contracts/scripts/check8.py`.
  - В отчёт `masking_report` — `"partners": {"pairs": [...], "covered": [bool, ...], "targets_with_visible_partner": int, "share": float}`: цель считается, если в том же событии видно (не `[MASK]`, не `[UNK]`) значение ключа-партнёра по любой найденной паре; `covered` — пара внутри одной группы конфига.
  - CLI: `--pairs-row-groups N` (8); `show` печатает долю и непокрытые пары.

- [ ] **Step 1: Тесты:**
  - `test_report_finds_a_copied_field` — в мире с ключом-копией (Y = f(X) в каждом событии) пара `(Y, X)` найдена, отношение < 0,05.
  - `test_report_partner_share_drops_to_zero_with_the_group` — без групп `share > 0`; с группой `(X, Y)` `share == 0` и пара `covered`.
- [ ] **Step 2:** FAIL. **Step 3:** реализовать. **Step 4:** PASS; `python -m src.masking.report --help`; `pytest -o addopts="" -q`.

### Task 6: Кандидат 3 — ключи закрытого события (E6)

**Files:**
- Modify: `src/mlm/settings.py`, `src/mlm/model.py` (`PackedBatch`, `pack`, `Model`), `src/mlm/train.py` (`load_trained`, старт `train`), `README.md`
- Test: `tests/test_hide_event_keys.py`

**Interfaces:**
- Produces (восстановление волны 4, коммит 62170f4: `src/mlm/model.py:560-600, 630`, `src/mlm/train.py:647, 659-668, 972`):
  - `MlmConfig.hide_event_keys: bool = False`.
  - `PackedBatch.event_masked: Tensor  # [T] bool` — `pack` строит из `client.reason == "event"`.
  - `Model.hidden_key: int | None`, `Model.hide_event_keys(hidden: int)`, `Model._event_keys(data)` — `data.key_ids.masked_fill(data.event_masked, hidden_key)`; используется в эмбеддинге событий (`model.py:748`) для путей sdpa и flash. `RecentTypes` и головы читают исходные `data.key_ids`.
  - `train.hide_keys(model)` — `model.hide_event_keys(load_special_tokens()[MASK])`; вызывается в `train` и `load_trained` при флаге.

- [ ] **Step 1: Тесты:**
  - `test_event_masked_keys_are_masked_in_the_input` — у токенов события, закрытого механизмом event, ключи во входе эмбеддинга — `[MASK]`; у прочих — исходные.
  - `test_hide_keys_off_is_the_previous_forward` — флаг выключен: выход `torch.equal` прежнему.
  - `test_load_trained_restores_hide_keys` — чекпойнт с флагом грузится со скрытием.
  - `test_zero_mask_input_is_unaffected` — при `ZERO_MASK` выход с флагом и без побитно равен.
  - CUDA: `test_flash_and_sdpa_agree_with_hidden_keys` — в допуске прохода целиком `tests/test_cuda_flash.py` (5e-2).
- [ ] **Step 2:** FAIL. **Step 3:** реализовать; README. **Step 4:** `pytest tests/test_hide_event_keys.py tests/test_model_equivalence.py tests/test_pack.py -v` и `pytest -m cuda tests/test_hide_event_keys.py -v` — PASS; `pytest -o addopts="" -q`.

### Task 7: Кандидат 4 — шкала суммы по типу и отпечаток словаря в 05

**Files:**
- Create: `audit/2026-11-03-pretrain-fixes/configs/tokenizer-amount-by-type.json`
- Modify: `src/dataset/build.py` (meta), `src/temporal/samples.py` (проверка), при необходимости `src/tokenization/numeric.py` (совпавшие границы), `README.md` (разделы 3 и 5)
- Test: `tests/test_numeric_scales.py`, `tests/test_dataset_build.py`

**Interfaces:**
- Produces:
  - `05/<группа>/meta.json` получает `"vocabulary"` — значение из meta `04` (`TokenizedGroup.meta["vocabulary"]`, уже сверенное с `vocabulary_digest()` в `src/dataset/tokenized.py:219`).
  - `TemporalGroup` (`src/temporal/samples.py:112-140`): ключ `vocabulary` есть и ≠ `vocabulary_digest()` — `SamplesError` с обоими отпечатками; ключа нет (набор, собранный до изменения) — принимается.
  - Конфиг кандидата: `{"numeric_encoders": {"transaction_amount": <as_dict() кодировщика по умолчанию с "split_by": "event_type">}}`; грузится `TokenizerConfig` без ошибок.
  - Поведение квантильной шкалы условия с числом различных значений меньше числа корзин закреплено тестом: границы строго возрастают, кодирование не падает. Если сейчас границы совпадают — убрать совпавшие только в этом случае: шкалы, где совпадений нет, побитно прежние.

- [ ] **Step 1: Тесты:**
  - `test_amount_scale_by_event_type` — `split_by="event_type"` на синтетическом мире: свои диапазоны у каждого типа (`transaction_amount_<тип>_bucket_N`), тип, которого train не видел, — `[UNK]`.
  - `test_rare_condition_scale_has_increasing_boundaries` — условие с 3 различными значениями при 16 корзинах.
  - `test_default_scales_are_unchanged` — словарь мира с конфигом по умолчанию побитно прежний.
  - `test_candidate_tokenizer_config_loads` — файл кандидата грузится, `split_by == "event_type"`, остальные поля равны умолчанию.
  - `test_dataset_meta_records_the_vocabulary` — `05/meta.json["vocabulary"] == vocabulary_digest()`.
  - `test_samples_refuse_another_vocabulary_and_accept_legacy_meta` (Review Focus 5).
- [ ] **Step 2:** FAIL. **Step 3:** реализовать; README — шкала по типу как настройка, отпечаток словаря в 05. **Step 4:** `pytest tests/test_numeric_scales.py tests/test_dataset_build.py tests/test_time_anchor.py tests/test_source_inputs.py -v` — PASS; `pytest -o addopts="" -q`.

### Task 8: Конфиги кандидатов и процедура прогона

**Files:**
- Create: `audit/2026-11-03-pretrain-fixes/README.md`, `configs/train-aux-per-client.json` (`{"aux_per_client": true}`), `configs/train-shuffle.json` (`{"shuffle_order": true}`), `configs/train-hide-keys.json` (`{"hide_event_keys": true}`), `configs/masking-groups.json` (`field_groups` из Global Constraints, `event_type_event_only: true`), `chain.sh`
- Test: `tests/test_candidate_configs.py`

**Interfaces:**
- Consumes: поля задач 2–7.
- Produces:
  - `chain.sh <имя> <train-config> [masking-config] [baseline]` — `src.mlm.train --config … [--masking-config …] --out data/runs/<имя>` → `src.downstream.embed --checkpoint data/runs/<имя>/best_checkpoint.pt --tag <имя>` → из `churn_baseline/` `python -m churn.plus_usr --tag <имя>` → `src.downstream.probe --tag <имя> --baseline <baseline, по умолчанию ref-2026-10-09>`; логи и коды выхода — в `data/runs/<имя>/`; останов на первой ошибке. Запуск владельцем: `setsid nohup bash … &`.
  - README аудита: основание (ссылки на `audit/2026-11-02-pretrain-contracts/`), порядок 1→4, команды шага 0 и каждого кандидата, копия `data/03–07` перед кандидатом 4 и команда пересборки `python -m src.pipeline --from fit --args fit="--config audit/2026-11-03-pretrain-fixes/configs/tokenizer-amount-by-type.json"`, правило решения (README §13) и порядок действий после решения (флаг → умолчание, удаление старой ветки; при отказе — удаление переключателя; при отказе от кандидата 3 — `event_type_event_only` остаётся для кандидата 4), пустая таблица итогов, заметка про LoRA до кандидата 4.

- [ ] **Step 1: Тест** `test_candidate_configs_load` — каждый `configs/*.json` грузится своим `from_dict`/`load` (`MlmConfig`, `MaskingConfig`, `TokenizerConfig`) и включает ровно свой флаг; группы равны Global Constraints.
- [ ] **Step 2:** FAIL. **Step 3:** создать файлы. **Step 4:** PASS; `bash -n audit/2026-11-03-pretrain-fixes/chain.sh`.

## Verification (до прогонов)

1. `pytest -o addopts="" -q` и `pytest -o addopts="" -q -m cuda -rs` — всё прошло, CUDA не пропущены.
2. `python -m src.mlm.train --help`, `python -m src.masking.report --help` — без ошибок.
3. `git status` — изменены только файлы задач; `data/` не тронут.
4. Дальше — по командам владельца: шаг 0, затем кандидат 1 (`setsid nohup bash audit/2026-11-03-pretrain-fixes/chain.sh aux-per-client audit/2026-11-03-pretrain-fixes/configs/train-aux-per-client.json &`) и решение по нему.
