<!-- Обзор агента аудита (research-efficiency), текст без правок. -->

# Аудит пропускной способности Stage 14 (обучение MLM) и CatBoost-бейзлайнов, RTX 3050 Laptop 4 GB под WSL2

Режим только чтения: код я не менял, GPU, pytest, CatBoost и этапы конвейера не запускал. Все замеры — это небольшие Python-пробы на CPU. Машина при этом была занята другими агентами, поэтому точность порядка ±20%. Метки: **[ТОЧНО]** — проверено в коде или данных; **[ВЕРОЯТНО]** / **[ГИПОТЕЗА]** — нужен профиль на GPU.

## 0. Итог

1. **[ТОЧНО] Причина стопов «каждые ~32 шага».**
   - Одна группа строк `07_batches` содержит 32 клиента (`src/batching/settings.py:46`). Её целиком читают, переводят в Python-списки и маскируют в главном потоке до выдачи первого клиента (`src/mlm/inputs.py:208-215`, `:227-236`).
   - 85,6% micro-batch'ей состоят из одного клиента, поэтому одна группа строк ≈ 32 шага.
   - На медианной группе CPU-путь занимает ≈1,39 с. В пересчёте на эпоху это ≈640 с ≈ 10,7 мин, или ~38% от 28 мин. Для val ≈24 с.
2. **[ТОЧНО] Что внутри этих 1,39 с:**
   - `choose` (маскирование, blake2b на каждый розыгрыш и циклы Python по токенам): 0,86 с;
   - `to_pylist`: 0,37 с, из них ~85% — перевод колонки `event_time` в Python `datetime`, хотя в модель она не подаётся;
   - остальное мелочь.
3. **[ВЕРОЯТНО] Главный резерв на GPU — внимание энкодера события.**
   - В flash-attn 2 ядро для головы размерности 32 считает плитки 128×128 на каждый сегмент, а сегмент здесь — одно событие средней длины 8,2 токена.
   - Лишняя работа ≈176× против ΣL². Оценка: ≈0,49 TFLOP на шаг при полезных ≈3 GFLOP. Нужен профиль.
4. **[ТОЧНО] Память.**
   - Полные логиты `[M, 5717]` bf16 живут весь backward: при M=33 138 это +379 MB, при M=54 165 — +619 MB.
   - Под WSL есть рабочая защита от выноса памяти в системную RAM без `expandable_segments`: лимит `per_process_memory_fraction` вместе с `garbage_collection_threshold`, это задаётся переменной окружения.
5. **[ГИПОТЕЗА] Суммарный эффект.** Фоновый процесс загрузки плюс E1 и E4 дают ~28 → 17-19 мин на эпоху. С правками GPU-части (E5-E7) реалистично ~2× и больше, но цифру стоит фиксировать только после профиля.

## 1. Замеры и факты

### 1.1 Разбиение на micro-batch
Симуляция тем же жадным алгоритмом, что `inputs.micro_batches`, по реальным длинам из `07_batches/train`:

| Параметр | Значение |
|---|---|
| `token_budget` | 16384 (`src/mlm/settings.py:107`) |
| `grad_accum_steps` | 1 (`src/mlm/settings.py:111`) |
| micro-batch'ей на эпоху | 6876 (val: 1099) |
| из одного клиента | 85,6% |
| дороже бюджета | 63,2% |
| максимальная стоимость | 69 959 позиций |
| стоимость p10 / p50 | 11 187 / 23 509 |
| среднее на micro-batch | ≈26,6 тыс. токенов, ≈3342 события |
| доля целей | ≈0,26 на токен, то есть M ≈ 6,9 тыс. |
| при `token_budget=65536` | 3837 micro-batch'ей (−44%), максимум стоимости тот же (69 959) |

Итого ≈0,24 с на micro-batch в среднем (28 мин / 6876 с учётом val). Без стопа GPU-часть шага занимает ~0,13-0,15 с.

### 1.2 CPU-путь одной группы строк
Группа 32 — медиана по объёму: 32 клиента, 395 786 токенов, 102 848 целей.

| Этап | Время | Где |
|---|---|---|
| `read_row_group` | 0,039 с | `inputs.py:209` |
| `to_pylist` (19 колонок) | 0,370 с; из них `event_time` 0,237 из 0,278 (замер по колонкам) | `inputs.py:211` |
| `choose` | 0,864 с (`values_of` 0,245) | `masking/choose.py:94-128`, `:136-201`; `generator/rng.py:238-248` |
| `apply` | 0,041 с | `masking/apply.py` |
| `_client` | 0,078 с | `inputs.py:270-315` |
| `pack` (numpy, CPU) | 0,6 мс на micro-batch | `model.py:159-265` |

- Длинная группа (90-й перцентиль, 1,66 млн токенов) по линейной оценке даёт ≈5,8 с стопа.
- **Векторные прототипы на той же группе:**
  - Arrow → numpy через offsets/`flatten()` по всем колонкам: 0,140 с;
  - `values_of` на numpy: 0,002 с (340 692 значения, 45 491 допустимое событие).
- **Розыгрыши blake2b** (386 тыс. штук): 0,356 с через `KeyedRandom.chance` и 0,315 с пачкой. Решения побитно совпали. Значит, при сохранении blake2b нижняя граница — ~0,4 с на группу.
- `event_time` и `reason` читаются только отчётом и диагностикой: `src/mlm/build.py:290,327`, `src/mlm/diagnostics.py:83,89`. Обучение и `validate` их не используют.
- `src/mlm/inputs.py:14` импортирует `CALENDAR_PER_EVENT` из `src/embedding/inputs.py`, а там на строке 9 `import torch`. Путь данных тянет torch: RSS после `import src.mlm.inputs` — 544 MB.

### 1.3 Синхронизации CPU↔GPU и копирования на устройство [ТОЧНО]
- Внутри forward: `count = int((targets != IGNORE).sum())` в `model.py:333`. Цели уже отобраны по `labels != IGNORE` (`model.py:227,435`), поэтому это просто `targets.numel()`.
- На каждом шаге: `hits` делает два `int()` (`model.py:379`), плюс `out.loss.item()` в `train.py:215` и ещё раз в `train.py:731`. Печать на каждом шаге — `train.py:655`.
- Из-за этих синхронизаций CPU не может готовить следующий шаг, пока GPU считает текущий.
- `pack` на micro-batch делает 47-57 вызовов `torch.as_tensor(np, device)` (замер счётчиком, группа 198). По туториалу PyTorch, копия без `non_blocking` — это `cudaMemcpyAsync` плюс `cudaStreamSynchronize` на каждую.
- `VarlenLayout.build` всегда строит корзины для SDPA и отправляет их на GPU (`varlen.py:155-180`), даже на flash-пути, где они не нужны. Это ~24 копии из 47-57.

### 1.4 Память [ТОЧНО, кроме доли фрагментации]
- `logits = self.head(...)` на полном M с графом градиента (`model.py:433`) живёт в `out` до `del out` уже после backward (`train.py:723-738`). Пик выше на M×5717×2 байт: ≈79 MB в среднем, 379 MB при M=33 138, 619 MB при M=54 165 (эти M взяты из сообщения коммита 25915e0).
- Резерв 4,41 GiB при пике выделенного 2,35 GiB означает около 2 GiB фрагментации. Рост времени эпохи с 28,0 до 29,4 мин совпадает с учащающимся выносом памяти драйвером в системную RAM [ВЕРОЯТНО].

### 1.5 Внимание энкодера события
- Длины событий по трём группам (256 619 событий): p50 6, p90 15, p99 18, максимум 31, среднее 8,23. Отношение E·128²/ΣL² = 176.
- Ядро flash-attn для головы 32 [ТОЧНО по исходникам]:
  - forward: `Flash_fwd_kernel_traits<32,128,128,4>`;
  - backward: `<32,128,128,8,...>`;
  - сетка — (ceil(max_seqlen/128), число сегментов, головы).
- Оценка [ГИПОТЕЗА]:
  - на слой ≈13,4 тыс. блоков (3342 события × 4 головы), каждый ≈2,1 MFLOP в forward и ≈5,2 в backward;
  - итого ≈98 GFLOP на слой, ≈0,49 TFLOP на шаг за 5 слоёв;
  - пик bf16 у RTX 3050 Laptop ≈9-14 TFLOPS. Это моя оценка: 16 SM × 512 FLOP/такт × 1,06-1,74 ГГц; частоты по спецификациям, 512 FLOP/такт выведены из цифр Ampere для RTX 3090, отдельно не проверял;
  - значит, даже при 100% загрузке это ≥35-55 мс из ~130-150 мс шага.
- Попутно: `attend` не передаёт `deterministic` (`varlen.py:334-340`), по умолчанию там `False` (`flash_attn_interface.py:1384`). Backward энкодера истории на CUDA, вероятно, не побитно воспроизводим. Для A/B-сравнений нужен допуск.

### 1.6 Прочее
- AdamW без `fused` (`train.py:506`).
- Цикл `grad.div_` по ~103 тензорам — ~103 запуска ядра на шаг (`train.py:641-643`).
- **Внимание:** свежий запуск `train()` удаляет `data/14_train/checkpoint.pt` и `best_checkpoint.pt` (`train.py:602-609`). Любой замер через `python -m src.mlm.train` без `--resume` сотрёт чекпойнты законченного 10-эпохного прогона.

## 2. Источники

**Прочитано:**
- [PyTorch CUDA notes 2.14](https://docs.pytorch.org/docs/2.14/notes/cuda.html): `max_split_size_mb`, `roundup_power2_divisions`, `garbage_collection_threshold` работают только с native-бэкендом; CUDA graphs запрещают динамические формы. → Графы здесь неприменимы (форма меняется каждый шаг, `max_seqlen` передаётся как Python int).
- [Исходник CUDACachingAllocator.cpp](https://raw.githubusercontent.com/pytorch/pytorch/main/c10/cuda/CUDACachingAllocator.cpp): сборка мусора запускается только если задан `allowed_memory_maximum`; при нехватке — `release_cached_blocks` и повтор. → `garbage_collection_threshold` без лимита доли памяти ничего не делает. В установленном `libc10_cuda.so` есть ключ `per_process_memory_fraction` [ТОЧНО по strings], так что лимит можно задать одной переменной окружения. Имена `PYTORCH_CUDA_ALLOC_CONF` и `PYTORCH_ALLOC_CONF` поддерживаются оба.
- [unsloth-zoo PR #1235](https://github.com/unslothai/unsloth-zoo/pull/1235): Windows и WSL исключены из `expandable_segments` (нужен cuMemMap / VMM); вместо него — `roundup_power2_divisions`. → Подтверждает ваш опыт; запасной вариант — округление размеров.
- [microsoft/WSL #11050](https://github.com/microsoft/WSL/issues/11050): CUDA в WSL2 не соблюдает «Sysmem Fallback Policy» и молча выносит память в системную RAM. → Настройка драйвера не поможет, лимит нужно ставить внутри процесса.
- [NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html): pinned-память ограничена, Unified Memory не поддерживается, NVML не отдаёт загрузку GPU. → Закреплять только маленькие буферы; загрузку мерить `torch.profiler`, а не `nvidia-smi`.
- [NVIDIA forum: cudaHostRegister на WSL2](https://forums.developer.nvidia.com/t/cudahostregister-not-supported-on-wsl2/279429): `cudaHostAlloc` работает, `cudaHostRegister` — нет. → Не включать `pinned_use_cuda_host_register`.
- [PyTorch: pin_memory и non_blocking](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html): блокирующий `to()` делает `cudaStreamSynchronize` после каждой копии; `pin_memory()` на лету медленнее прямой копии. → Один заранее закреплённый буфер на micro-batch и одна копия с `non_blocking`.
- [PyTorch DataLoader 2.14](https://docs.pytorch.org/docs/2.14/data.html): у `IterableDataset` каждый worker получает свою копию и шардирует через `get_worker_info`; есть `prefetch_factor`, `in_order`; своим типам для `pin_memory` нужен метод. → Один worker на группы строк, порядок сохраняется.
- [Python 3.14 What's New](https://docs.python.org/3/whatsnew/3.14.html): старт по умолчанию — `forkserver`; free-threaded сборка официальна, штраф 5-10%; у `InterpreterPoolExecutor` ограничена совместимость расширений. → Worker данных стартует «чисто» (нужен picklable dataset, `ParquetFile` открывать внутри). Наша venv — сборка с GIL (`Py_GIL_DISABLED`=False, проверено).
- [hashlib](https://docs.python.org/3/library/hashlib.html): GIL отпускается только на входах больше 2047 байт. → У нас входы по 32 байта, поэтому поток для маскирования будет драться за GIL — нужен процесс.
- [pyarrow ListArray](https://arrow.apache.org/docs/python/generated/pyarrow.ListArray.html): `flatten()` учитывает смещение среза, `.values` — нет. → Векторный путь делать через `flatten()` и offsets.
- flash-attn: [fwd launch](https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/csrc/flash_attn/src/flash_fwd_launch_template.h), [bwd launch](https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/csrc/flash_attn/src/flash_bwd_launch_template.h): для головы 32 плитки 128×128. → См. 1.5.
- Локальный `flash_attn_interface.py:76-393`: varlen forward/backward зарегистрированы как `torch.library.custom_op` с fake-реализациями. → `torch.compile` может пройти через них без разрыва графа.
- [PyTorch 2.10 blog](https://pytorch.org/blog/pytorch-2-10-release-blog/): `torch.compile` на Python 3.14; 3.14t экспериментально; `varlen_attn()` на FA2, «A100 or newer»; combo-kernels.
- Локальный `torch/nn/attention/varlen.py:101,118`: в `varlen_attn` dropout жёстко равен 0. → Заменой flash-attn он станет только если убрать attention-dropout, а это изменение модели.
- CatBoost: [ускорение обучения](https://catboost.ai/docs/en/concepts/speed-up-training), [общие параметры](https://catboost.ai/docs/en/references/training-parameters/common), [CTR](https://catboost.ai/docs/en/references/training-parameters/ctr), [output](https://catboost.ai/docs/en/references/training-parameters/output), [#3094](https://github.com/catboost/catboost/issues/3094). Значения по умолчанию на CPU: `boosting_type` Plain, `bootstrap` MVS с `subsample` 0.8, `one_hot_max_size` 2, `max_ctr_complexity` 4, `border_count` 254. Документация рекомендует `max_ctr_complexity` 1-2, больший `one_hot_max_size`, Bernoulli, `pandas.Categorical` и повторное использование квантизованного Pool. PRAUC на GPU не поддерживается.

**Только по выдаче поиска, страницы не открывал:**
- NVIDIA a_id/5490: fallback появился в драйвере 536.40.
- notebookcheck / gpu-monkey: 2048 CUDA-ядер, шина 128 бит, 192 GB/s, 35-80 Вт.
- Доки AdamW: на CUDA по умолчанию foreach, есть `fused`.
- Issue #121857 (fused бывает медленнее) и дока «Reducing compile time» (региональная компиляция плюс `mark_dynamic`).

## 3. Оптимизации

### A. Данные — главный выигрыш
- **E1.** Не читать `event_time` (и `reason`) в train и val. Сделать поля отчётными или читать их лениво. Даёт ≈−110 с train и ≈−18 с val на эпоху (~7%), а ещё меньше RAM и pickling. Риск минимальный.
- **E2.** Подготовку групп строк вынести в отдельный процесс:
  - `DataLoader(IterableDataset по группам строк, batch_size=None, num_workers=1, prefetch_factor=2)`;
  - маскирование эпохи передавать в конструктор;
  - `micro_batches` и `pack` оставить в главном процессе, поэтому порядок и логика `skip` при resume не меняются;
  - GPU-время на группу (~32 шага × 0,13 с ≈ 4-5 с) больше CPU-времени (1,4-5,8 с), так что одного worker хватит;
  - ожидание: снимается почти весь стоп, 28 → ~17-19,5 мин;
  - RAM worker'а ≈0,5-0,6 GB. Если убрать импорт torch из `src/mlm/inputs.py:14` и пойти через `multiprocessing` без DataLoader — ~0,1-0,15 GB [ВЕРОЯТНО].
- **E3.** Векторизовать маскирование: `values_of` и `apply` на numpy.
  - Вариант (а) с тем же blake2b даёт побитно те же маски: ≈1,39 → ~0,5-0,6 с на группу.
  - Вариант (б) — counter-based numpy `Philox(key=(seed, stream, client))`: ≈0,2 с, но это смена контракта маскера: новая версия, пересборка `08`.
  - Эффект — запас для E2 и меньше RAM.

### B. Синхронизации и копирования
- **E5.**
  - `count = targets.numel()` в `mlm_loss`;
  - `Scores` копить на устройстве, `.item()` — раз в N шагов или в конце эпохи. Это меняет формат лога шага, нужно решение пользователя;
  - `pack` через один закреплённый буфер и одну копию с `non_blocking=True`, дальше views;
  - на flash-пути не строить корзины SDPA.
  - Ожидание: 3-10% [ГИПОТЕЗА].
  - Мерить: число и время `cudaStreamSynchronize` в профиле, простои GPU между шагами.

### C. Память
- **E4.** Два шага:
  - переменная окружения `PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.75,garbage_collection_threshold:0.8`. Долю подобрать по `torch.cuda.mem_get_info()` на старте. `roundup_power2_divisions:4` и `max_split_size_mb:128` — отдельными вариантами A/B: они могут как помочь, так и увеличить число `cudaMalloc`;
  - в train не держать полные логиты: top-1/top-5 считать кусками под `no_grad`. Это меняет `Predicted.logits`, от которого зависят тесты и отчёт.
  - Дополнительно: `torch.cuda.empty_cache()` после val.
  - Ожидание: резерв ≤ ~3,0 GiB, нет выноса памяти, время эпох перестаёт расти (−5% и больше).
  - Мерить: `memory_stats()` → `reserved_bytes.all.peak`, `num_alloc_retries`, `num_ooms`; время по эпохам.

### D. GPU-вычисления
- **E6.** Внимание энкодера события: flash-attn заменить на плотное внимание по log2-корзинам длины 4/8/16/32 (bmm или SDPA на math-бэкенде).
  - Лишняя работа ≈1,6× против ≈176×; flash оставить для анкеты и истории.
  - Проверка — существующие тесты flash ↔ корзины с допуском 5e-2 (`tests/test_cuda_flash.py:254-300`).
  - Ожидание: −20-40% времени шага, если профиль покажет ≥30% на flash-ядрах события [ГИПОТЕЗА].
- **E7.** Региональный `torch.compile(dynamic=True)` только для плотных частей: norm1 → qkv, затем out_proj → dropout → residual → norm2 → FFN.
  - Flash и цикл по группам оставить в eager: `max_seqlen` и число групп — Python-значения, из-за них будут перекомпиляции.
  - Ожидание: −10-20% времени элементных операций [ГИПОТЕЗА].
  - Проверять `TORCH_LOGS=recompiles` и детерминизм dropout (`torch._inductor.config.fallback_random`).
- **E8.** `AdamW(fused=True)` и `torch._foreach_div_` вместо цикла по градиентам: ~1%.
- **E9.** `token_budget=65536`: −44% шагов при том же пике памяти.
  - Меняет оптимизацию: меньше шагов, больше батч.
  - Сравнивать val loss при равном времени; ожидание по скорости 5-10%.
- **E10** (решение о модели). dropout=0 даёт меньше ядер и открывает `torch.nn.attention.varlen`. Нужна проверка качества.
- **Не рекомендую:** CUDA graphs (динамические формы, лишняя память на 4 GB), `expandable_segments`, Python 3.14t, polars (узкое место — циклы Python, а не библиотека; и новая зависимость).

### F. CatBoost fraud
Параметры: `fraud_baseline/fraud/train.py:32-44`, `metric_period=5` на строке 105.

- **Данные:** 527 776 строк train, 64 признака, 8 категориальных.
  - Кардинальности: `merchant_name` 4996, `mcc` 63, `merchant_category` 46, `region` 20, `merchant_country` 17, `income_type` 7, `channel` 4, `gender` 2.
  - При `one_hot_max_size=2` на CTR идут 7 из 8, с комбинациями до 4 признаков.
  - Около 2822 итераций (лучшая 2521 плюс 300 ожидания), ≈0,6 с на итерацию.
- **Порядок проверок:**
  - **C1:** `one_hot_max_size=32` и `max_ctr_complexity` 1 или 2 — главный кандидат [ВЕРОЯТНО в разы];
  - **C2:** ранняя остановка по Logloss, а PR-AUC считать отдельно или с `metric_period` 25-50. PR-AUC на 279 позитивах val шумит;
  - **C3:** `learning_rate` 0.1 при 1500 итерациях (лучшая итерация 2521 почти упёрлась в лимит 3000);
  - **C4:** `border_count` 128, Bernoulli с `subsample` 0.5-0.66;
  - **C5:** `thread_count` 12, если машина свободна;
  - **C6:** `pandas.Categorical` и повторное использование квантизованного Pool;
  - **C7:** GPU — только без одновременного обучения PRAGMA, с `gpu_ram_part` ~0.5 и ранней остановкой по AUC или Logloss (PRAUC на GPU нет).
- **Мерить:** сек/итерацию на 300 итерациях, затем полный прогон; PR-AUC val/test с bootstrap-CI. Разница < ~0,02, вероятно, шум.
- Churn — ~9,5 тыс. строк, узким местом не является.

### G. Генератор и этапы 01–09
- `default_workers = cpu_count - 1` (`src/generator/emit.py:545-546`), `Pool` без `maxtasksperchild` (`:438-443`). При `forkserver` каждый worker импортирует всё заново.
  - Число worker'ов по умолчанию выводить из `MemAvailable` / пиковый RSS worker'а.
  - Добавить `maxtasksperchild`.
- В этапах много `to_pylist` и разбора по строкам: `preprocessing/canonical/events.py:449`, `rawdata.py:933`, `temporal/build.py:93`, `masking/build.py:103`. Сначала завести таймеры по этапам, потом векторизовать. Время отдельных этапов я не измерял.

## 4. Список экспериментов по приоритету

| # | Что | Ожидание | Риск или контракт | Как мерить |
|---|---|---|---|---|
| P0 | Отдельный стенд профилирования: не `train()`, копия `14_train`, группы строк 198/32/249, `torch.profiler` + CUDA events + `memory_stats` | — | Нет | Доли ядер по энкодерам через `record_function`, простои GPU, ожидание данных |
| P1 | E1 (`event_time`/`reason`) | ~−7% | Поля отчёта | Время эпохи, те же потери |
| P2 | E2 (процесс загрузки) | 28 → ~17-19,5 мин | RAM worker'а | Время `next()` ≈ 0, время эпохи |
| P3 | E4 (лимит памяти + без полных логитов) | Нет выноса памяти, эпохи без дрейфа | `Predicted.logits` | Пик резерва, `num_alloc_retries` |
| P4 | E6 (плотное внимание события) | До −20-40% шага | Тесты допусков | Профиль, top-1 и val за 1 эпоху |
| P5 | E5 (синхронизации и копирования) | 3-10% | Формат лога | Профиль |
| P6 | E3 (векторное маскирование) | 1,39 → 0,5 с (побитно) или 0,2 с (Philox) | (б) — версия маскера | Тест равенства масок на синтетике |
| P7 | E7 (compile) | −10-20% элементных операций | Перекомпиляции, RNG | Шаг p50/p95, логи recompiles |
| P8 | E8 | ~1% | Нет | Шаг |
| P9 | E9, E10 | 5-10% и больше | Оптимизация и модель | val loss при равном времени |
| P10 | C1–C7 CatBoost | Вероятно в разы | Качество | Сек/итерацию, PR-AUC с CI |

Все запуски — только по прямой команде пользователя (CLAUDE.md).

## 5. Ограничения аудита
- GPU-гипотезы (E5–E7, доля flash) без профиля — это оценки.
- Экстраполяция CPU-пути линейна по токенам и сделана на занятой машине.
- CatBoost не запускался.
- Время этапов 01–09 не измерялось.
