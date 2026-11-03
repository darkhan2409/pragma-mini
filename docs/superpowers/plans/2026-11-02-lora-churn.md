# LoRA-дообучение PRAGMA под отток — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** сценарий `lora` — дообученная заплатками LoRA модель PRAGMA с головой на `[USR]` даёт вероятности ухода на val, сравнимые в `probe` с CatBoost и с `[USR]` → LR.

**Architecture:** заплатки ставятся `torch.nn.utils.parametrize` на веса загруженной эталонной модели, код модели и путей flash/sdpa не меняется. Свой цикл дообучения в `src/downstream/finetune.py` (вход train из `05_dataset/train` без маски, val — `at_cutoff`), прогнозы и meta — в `data/13_downstream/<name>/`, проба читает их флагом `--lora`.

**Tech Stack:** Python 3.14, PyTorch 2.13 (flash-attn, bf16 autocast), pyarrow/pandas, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-11-02-lora-churn-design.md` (+ отклонения ниже).

## Context

Аудит 2026-11-02 показал: `[USR]` → LR (0,452) проигрывает CatBoost (0,570) в основном из-за линейной головы над замороженным вектором, а нелинейная голова на сыром `[USR]` не помогает. В статье PRAGMA выигрыш у бейзлайнов даёт LoRA-дообучение. Владелец согласовал дизайн по частям и спек; этот план — шаги реализации. Обучение на GPU — отдельной командой владельца после реализации.

### Отклонения от спека (решения владельца 2026-11-02 и проверка плана)

1. `finetune --final-test` не обучает: грузит готовый `data/runs/<name>/lora.pt` и считает прогнозы val и test той же моделью.
2. `merge` и его тест убраны (в рабочем пути не вызывается).
3. Доля и seed отложенной части — константы `HOLDOUT = 0.2`, `SPLIT_SEED = 42`, а не поле конфига: иначе разбиение перестаёт совпадать с `churn.train.fit_task`.
4. Эталон в meta сверяется по sha256 файла чекпойнта (`train.file_digest`), а не по пути: `data/runs/ref-2026-10-09/best_checkpoint.pt` — жёсткая ссылка на `data/12_train/best_checkpoint.pt`.
5. Отдельного `FinetuneError` нет: `ConfigError` (`src/mlm/settings.py:71`), `TrainingError` (`src/mlm/train.py:172`), `ValueError`.

## Global Constraints

- Учимся только на train задачи `churn_active90`; val только оценивается; test — только с `--final-test`; `truth/*` не читается.
- Эталон заморожен; обучаемы только `A`, `B` заплаток и голова. Код `src/mlm/model.py`, `src/mlm/varlen.py`, энкодеров и предобучения не меняется.
- Параметры по умолчанию: rank 8, alpha 8.0, AdamW lr 1e-3, weight_decay 0.01, max_grad_norm 1.0, clients_per_step 32 (не меньше стольких клиентов на шаг), token_budget 48 000, max_epochs 10, patience 3 (по log-loss отложенной части).
- Голова считается в fp32 вне autocast; добавка `B·A` — в fp32 (autocast выключен внутри `LoraDelta.forward`).
- Тесты не трогают `data/`: новый каталог `RUNS_DIR` добавить в `PLACES` (`tests/conftest.py:39`), читать через функцию в момент вызова.
- CUDA-тесты — `@pytest.mark.cuda` + `world.flash_ready()`; пропуск не считается пройденной проверкой.
- Ошибка, найденная тестом, чинится в коде; допуски не ослабляются.
- Коммиты и ветки — только по команде владельца (CLAUDE.md): в задачах вместо коммита — прогон тестов и отчёт.
- Тексты, комментарии, сообщения об ошибках — по-русски, как в проекте; комментарии объясняют причины и инварианты.

## Review Focus

1. `--name` совпадает с тегом векторов или каталогом предобучения (`data/runs/ref-2026-10-09` уже хранит `telemetry.jsonl`) — отказ, ничего не перезаписано. Тест — задача 3.
2. `--final-test` при `lora.pt`, обученном на другом эталоне (sha256 не совпал) или без `lora.pt` — отказ. Тест — задача 3.
3. Клиент задачи отсутствует в `05_dataset/train` или без событий — отказ с именем клиента, не пропуск. Тест — задача 3.
4. Прогнозы `lora` с лишними или недостающими клиентами, другой T или от другого эталона — проба отказывает. Тест — задача 4.
5. Голова под bf16-autocast считает в fp32 (логит около −2,9 в bf16 теряет точность). Тест — задача 2.

---

### Task 1: Заплатки и голова (`src/downstream/lora.py`)

**Files:**
- Create: `src/downstream/lora.py`
- Test: `tests/test_lora.py`

**Interfaces:**
- Produces:
  - `TARGETS_EVENT = ("self_attn.in_proj_weight", "self_attn.out_proj.weight", "linear1.weight", "linear2.weight")`, `TARGETS_BLOCK = ("qkv.weight", "out.weight", "ffn.0.weight", "ffn.3.weight")`.
  - `class LoraDelta(nn.Module)`: `__init__(self, out_features: int, in_features: int, rank: int, alpha: float, generator: torch.Generator)`; `A` [rank, in] ~ N(0, дисперсия 1/in) разыгрывается на CPU генератором; `B` [out, rank] — нули; `forward(self, weight: Tensor) -> Tensor` = `weight + (B @ A) * (alpha / rank)`, добавка в fp32 при выключенном autocast, результат в dtype `weight`.
  - `inject(model: Model, rank: int, alpha: float, seed: int) -> list[str]` — замораживает все параметры модели, ставит `LoraDelta` в фиксированном порядке: `event.layers[i]` (TARGETS_EVENT), `profile.layers[i]`, `history.layers[i]` (TARGETS_BLOCK); возвращает полные имена целей (`"event.layers.0.self_attn.in_proj_weight"`, …); повторный вызов — `ValueError`; после — проверяет, что обучаемы ровно `A` и `B`.
  - `class ChurnHead(nn.Module)`: `__init__(self, dim: int, base_rate: float)` — `Linear(dim, 1)`, веса нули, смещение `log(p/(1-p))`; `forward(usr: Tensor) -> Tensor` — логиты `[B]`.
  - `trainable(model: Model, head: ChurnHead) -> list[nn.Parameter]`.
  - `adapter_state(model: Model, head: ChurnHead) -> dict[str, Tensor]` — CPU-клоны: ключи `"<цель>.A"`, `"<цель>.B"`, `"head.weight"`, `"head.bias"`.
  - `load_adapters(model: Model, head: ChurnHead, state: dict[str, Tensor]) -> None` — точное совпадение ключей и форм, иначе `ValueError`.
  - `delta_ratio(model: Model) -> float` — ‖ΔW‖/‖W‖ по всем целям (для телеметрии: ловит «проглоченные» bf16 добавки).

- [ ] **Step 1: Тесты** (`world.model()` из `tests/world.py:365`; сравнения базы и LoRA — в одном режиме и одном grad-состоянии, чтобы не попасть на fused fast path `TransformerEncoderLayer` в eval/no_grad):
  - `test_zero_adapters_are_the_reference_bitwise` — `client_embeddings` до и после `inject` `torch.equal`.
  - `test_only_adapters_and_head_learn` — задать ненулевые `B` и голову, один шаг AdamW по `trainable`: у всех прочих параметров `grad is None` и значения `torch.equal` исходным; у `A`, `B`, головы `grad is not None`.
  - `test_targets_and_count` — имена целей в порядке Interfaces; число обучаемых = Σ rank·(in+out) по целям (+ dim+1 у головы).
  - `test_adapters_survive_save_and_load` — `adapter_state` → новая модель + `inject` + `load_adapters` → `client_embeddings` и логиты `torch.equal`; чужие ключи/формы — `ValueError`.
  - `test_head_starts_at_the_base_rate` — `sigmoid(head(x)) == base_rate` для любого `x`.
  - `test_inject_twice_is_refused`.
- [ ] **Step 2:** `pytest tests/test_lora.py -v` — FAIL (модуля нет).
- [ ] **Step 3:** реализовать `src/downstream/lora.py` по Interfaces.
- [ ] **Step 4:** `pytest tests/test_lora.py -v` — PASS; `pytest -q` — без регрессий.

### Task 2: Цикл дообучения и прогноз (`src/downstream/finetune.py`, ядро)

**Files:**
- Create: `src/downstream/finetune.py`
- Test: `tests/test_finetune.py`

**Interfaces:**
- Consumes: Task 1 (`inject`, `ChurnHead`, `trainable`, `adapter_state`, `load_adapters`, `delta_ratio`); `src.mlm.model.pack`, `src.mlm.inputs.micro_batches`, `src.mlm.varlen.autocast`, `src.mlm.train.append_telemetry`, `TrainingError`, `ConfigError`.
- Produces:
  - `@dataclass(frozen=True) class FinetuneConfig` — поля и значения из Global Constraints; `from_dict(data: dict) -> FinetuneConfig` (незнакомый ключ — `ConfigError`), `load(path: Path | None)`, `validate()`, `as_dict() -> dict`.
  - `HOLDOUT = 0.2`, `SPLIT_SEED = 42`.
  - `@dataclass class FitResult: state: dict[str, Tensor]; best_epoch: int; held_scores: dict[str, float]; history: list[dict]`.
  - `predict(model, head, clients: Iterable[Client], device, token_budget: int) -> dict[str, float]` — eval, `no_grad`, `micro_batches`; модель под `autocast(device)`, голова на `usr.float()` вне autocast; вероятность — `sigmoid` в fp32.
  - `fit(model, head, fit_clients: Callable[[int], Iterable[Client]], held_clients: Callable[[], Iterable[Client]], labels: dict[str, int], config: FinetuneConfig, seed: int, device, telemetry: Path | None) -> FitResult`:
    - `torch.manual_seed(seed)` в начале (после `inject` и головы); AdamW по `trainable`.
    - `model.train()` и `head.train()` в начале каждой эпохи (`predict` переводит в eval): dropout замороженных слоёв работает, как в спеке.
    - Micro-batch: `usr = model.client_embeddings(pack(clients, device))` под autocast → `head(usr.float())` → `binary_cross_entropy_with_logits(..., reduction="sum")` → `backward`; когда накоплено ≥ `clients_per_step` клиентов (и в конце эпохи): градиенты делятся на число клиентов, `clip_grad_norm_`, нечисловая потеря или норма — `TrainingError` до `step`.
    - Конец эпохи: `predict` по `held_clients()`, log-loss и PR-AUC; запись `{"kind": "epoch", epoch, train_loss, held_log_loss, held_pr_auc, delta_ratio, seconds, cuda_peak_allocated_gib}` в телеметрию; лучшее — по строгому уменьшению log-loss; останов — `patience` эпох без улучшения или `max_epochs`.
    - Возвращает `adapter_state` лучшей эпохи и её `held_scores`.

- [ ] **Step 1: Тесты** (мир: `world.install` + `many()` из `tests/test_scheduler.py:44`, модель `world.model()` или `load_model(..., CPU, "sdpa")`, клиенты — `Source("train", ZERO_MASK)`; метки — синтетические по `client_id`):
  - `test_config_is_strict` — незнакомый ключ и `rank=0` — `ConfigError`; значения по умолчанию равны Global Constraints.
  - `test_fit_learns_only_adapters_and_keeps_the_reference` — после `fit` на 2 эпохи веса эталона `torch.equal` исходным; `history` из 2 записей; телеметрия — 2 строки JSON.
  - `test_fit_is_reproducible_on_cpu` — два `fit` с одним seed дают `torch.equal` состояния и одинаковые `held_scores`.
  - `test_step_waits_for_enough_clients` — шпион на `optimizer.step`: число шагов = ⌈клиентов эпохи / clients_per_step⌉ с учётом неделимых micro-batch'ей.
  - `test_nonfinite_loss_stops_before_the_step` — голова с NaN в весах: `TrainingError`, `A`/`B` не изменились.
  - `test_patience_stops_training` — подменённый `predict` с ростом log-loss: эпох = 1 + patience, лучшая эпоха 1.
  - `test_head_runs_in_fp32_under_autocast` (Review Focus 5) — `autocast`, через который `finetune` входит в смешанную точность, подменён monkeypatch'ем в модуле `finetune` на CPU-autocast bf16: `predict` равен `sigmoid(head(usr.float()))`, посчитанному отдельно, бит в бит.
- [ ] **Step 2:** `pytest tests/test_finetune.py -v` — FAIL.
- [ ] **Step 3:** реализовать `FinetuneConfig`, `FitResult`, `predict`, `fit`.
- [ ] **Step 4:** `pytest tests/test_finetune.py -v` — PASS; `pytest -q`.

### Task 3: Данные, команда и выходы

**Files:**
- Modify: `src/downstream/settings.py` (`RUNS_DIR = DATA_DIR / "runs"`, `run_dir(name) -> Path`)
- Modify: `src/downstream/tasks.py` (`churn_order`)
- Modify: `src/downstream/finetune.py` (данные, CLI)
- Modify: `tests/conftest.py` (`PLACES` += `("src.downstream.settings", "RUNS_DIR", "runs")`)
- Modify: `tests/test_downstream.py:105` (параметризовать группой `train` и `val`)
- Test: `tests/test_finetune.py`

**Interfaces:**
- Consumes: Task 2; `embed.trained_model` (`src/downstream/embed.py:59`), `embed.raw_record` (`:148`), `at_cutoff.ClientsAtCutoff`, `at_cutoff.clients_at` (`src/downstream/at_cutoff.py:86,279`), `tasks.churn_rows`, `tasks.churn_sources`, `settings.cutoff`, `settings.downstream_dir`, `probe.threshold_max_f1` (`src/downstream/probe.py:201`), `train.training_device` (`:184`), `train.limit_cuda_memory` (`:207`), `train.file_digest` (`:557`), `train.save_checkpoint` (`:853`), `src.mlm.inputs.Source`, `Prefetch`.
- Produces:
  - `tasks.churn_order(task: str, group: str) -> pd.Index` — `client_id` в порядке файла отчёта (`train_rows.parquet`/`eval_rows.parquet`), без сортировки; дубликаты — `ValueError`.
  - `split(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — `StratifiedShuffleSplit(n_splits=1, test_size=HOLDOUT, random_state=SPLIT_SEED)`, отсортированные номера (как `churn_baseline/churn/train.py:141-147`).
  - `ZERO_MASK = MaskingConfig(value_probability=0.0, event_probability=0.0, key_probability=0.0, unknown_probability=0.0, informativeness_weighted_masking=False)`.
  - `preflight(ids: pd.Index) -> None` — `05_dataset/train/meta.json`: `events_cutoff == cutoff("train")` и `time_anchor == "cutoff"`; `client_id` из `samples.parquet` (pyarrow, одна колонка) покрывают `ids` без дублей; иначе `ValueError` с именами.
  - `train_clients(ids: set[str], order: tuple[int, int] | None, workers: int) -> Iterator[Client]` — `Source("train", ZERO_MASK)`, порядок групп строк — перестановка `np.random.default_rng(list(order))` (`order = (seed, epoch)`) или файловый при `None` (отложенная часть); `Prefetch`; отдаёт клиентов из `ids`; клиент без событий или неполный проход — `ValueError`.
  - `val_clients(group: str, ids: list[str], workers: int) -> Iterator[Client]` — `clients_at(ClientsAtCutoff(group, cutoff(group)), workers)`, фильтр по `ids`; отсутствующий клиент или без событий — `ValueError`. Модульная функция: тесты подменяют её monkeypatch'ем (в мире `world.install` нет этапа 02).
  - CLI `build_parser()`, `run(args) -> int`, `main(argv)`: `--checkpoint` (обяз.), `--name` (обяз.), `--seed` (0), `--config`, `--final-test`, `--device` (`auto`), `--workers` (2). Ошибки `ConfigError`, `ValueError`, `CheckpointError`, `BackboneError`, `DeviceError`, `TrainingError`, `FileNotFoundError` → сообщение и `EXIT_BLOCKED`.
  - Выходы:
    - `run_dir(name)/lora.pt` (атомарно): `{"kind": "lora", "adapters", "config", "seed", "reference": {"checkpoint", "sha256", "epoch"}, "held": [client_id отложенной части], "best_epoch", "threshold"}`.
    - `run_dir(name)/telemetry.jsonl` — пересоздаётся при новом обучении.
    - `downstream_dir(name)/predictions.parquet` — `task="churn_active90", set="lora", group, client_id, y, score` (float64), по `client_id`; только val, с `--final-test` — val и test.
    - `downstream_dir(name)/meta.json` — `{"kind": "lora", "name", "reference", "groups": {g: {"cutoff", **raw_record(g)}}, "sources": {g: churn_sources(g)}, "threshold", "best_epoch", "trainable", "seed", "config", "final_test"}`; пишется последним.
  - Поток обучения: защита имени (каталоги `run_dir(name)` и `downstream_dir(name)` либо отсутствуют, либо их meta/`lora.pt` с `"kind": "lora"`) → `training_device` (+ `limit_cuda_memory` на CUDA) → `trained_model` → строки `churn_rows(...,"train").loc[churn_order(...,"train")]` → `preflight` → `split` → `inject`, `ChurnHead(d, y.mean())` → `fit` (`fit_clients = lambda e: train_clients(fit_ids, (seed, e), workers)`, `held_clients = lambda: train_clients(held_ids, None, workers)`) → лучшее состояние → порог `threshold_max_f1` по отложенной части → `lora.pt` → прогнозы val → `predictions.parquet`, `meta.json`.
  - Поток `--final-test`: `lora.pt` обязателен; `reference.sha256 == file_digest(--checkpoint)`; `inject` + `load_adapters` без обучения; прогнозы val и test.

- [ ] **Step 1: Тесты:**
  - `test_split_follows_the_file_order` — у фикстуры порядок файла ≠ сортировке; `split` равен `StratifiedShuffleSplit(1, 0.2, 42)` по меткам в порядке файла; части не пересекаются, доля ушедших сохранена.
  - `test_preflight_refuses_other_cutoff_anchor_or_missing_client` (Review Focus 3) — три случая, `ValueError` с именем клиента или поля.
  - `test_zero_mask_input_is_the_cutoff_input` — расширить `tests/test_downstream.py:105` параметром `group` ∈ {train, val}: для train `T = cutoff окна train`, `same(built, stored)`; и нет ни одной метки ≠ −100.
  - `test_finetune_end_to_end_on_cpu` — `settle(stage, train_people=many())`, `train(tiny(token_budget=6), 1, None, every_value())` даёт чекпойнт; churn-отчёты через `write_churn_reports`; `cutoff` подменён в `tasks` и `finetune` на `world.CUTOFF`; `finetune.val_clients` подменён чтением `Source("val", ZERO_MASK)`; `main([... "--device", "cpu", "--name", "ft"])`: есть `lora.pt`, `telemetry.jsonl`, `predictions.parquet` только с group=val, `meta.json` с `"kind": "lora"`; повтор с тем же seed — те же `score`.
  - `test_name_of_foreign_outputs_is_refused` (Review Focus 1) — `--name` = каталог с `telemetry.jsonl` предобучения или тег векторов: `EXIT_BLOCKED`, файлы не изменились.
  - `test_final_test_reuses_the_trained_adapters` (Review Focus 2) — после обычного прогона `--final-test` не вызывает `fit` (шпион), пишет val и test; `score` val равны прежним; без `lora.pt` или при другом sha256 эталона — `EXIT_BLOCKED`.
- [ ] **Step 2:** `pytest tests/test_finetune.py tests/test_downstream.py -v` — новые FAIL.
- [ ] **Step 3:** реализовать по Interfaces.
- [ ] **Step 4:** те же тесты PASS; `pytest -q`.

### Task 4: Сценарий `lora` в `probe`

**Files:**
- Modify: `src/downstream/tasks.py` (`lora_rows`)
- Modify: `src/downstream/probe.py` (`SCENARIOS`, `run_probe`, `show`, CLI)
- Test: `tests/test_downstream.py`

**Interfaces:**
- Consumes: выходы Task 3; `probe.paired` (`:258`), проверка `index.equals` + метки как у `catboost_plus_usr` (`probe.py:355-361`), `train.file_digest`.
- Produces:
  - `tasks.lora_rows(name: str, used: tuple[str, ...], embedded: dict) -> dict` — читает `downstream_dir(name)/meta.json` и `predictions.parquet`; отказ (`ValueError`, нет файлов — `FileNotFoundError`), если: `kind != "lora"`; `reference.sha256 != file_digest(Path(embedded["checkpoint"]))` или `reference.epoch != embedded["epoch"]`; `sources[g] != churn_sources(g)` или `groups[g]` (cutoff, sha выгрузок) ≠ текущим для оцениваемых групп; множество групп прогнозов ≠ `used[1:]`. Возвращает `{"threshold": float, "rows": {(task, group): DataFrame[churn, score] по client_id}, "meta": dict}`.
  - `SCENARIOS = ("catboost", "usr", "catboost_plus_usr", "lora")`; `COMPARED` не меняется.
  - `run_probe(tag, draws, seed, baseline=None, final_test=False, lora: str | None = None) -> dict` — при `lora`: строки равны строкам `catboost` (иначе `ValueError("lora/...: другие клиенты или метки, чем у catboost")`); `results["lora"] = {g: метрики, "threshold", "best_epoch", "trainable", "seed", "vs_reference": {g: paired(..., catboost)}, "vs_usr": {g: paired(..., usr)}}`; `report["lora"] = name`.
  - `show(report)` — строка `lora` в общей таблице и строки «Δ lora − catboost (g)», «Δ lora − usr (g)»; без `--lora` — подсказка «lora нет: python -m src.downstream.finetune …».
  - CLI: `--lora <name>`.

- [ ] **Step 1: Тесты** (`probe_world`, `write_vectors` из `tests/test_downstream.py:482,522`; вспомогательный `write_lora(name, baseline, damage=None)` пишет `predictions.parquet` и `meta.json`; `checkpoint` в meta векторов — маленький файл в `stage`, его sha256 — в meta LoRA):
  - `test_lora_scenario_is_compared_with_catboost_and_usr` — `run_probe(..., lora="ft")`: `results["lora"]["vs_reference"]["val"]` и `["vs_usr"]["val"]` равны `paired(...)`, посчитанному вручную; порог из meta; в `show` есть строки lora.
  - `test_lora_of_other_clients_reference_or_sources_is_refused` (Review Focus 4) — параметризация: лишний клиент, недостающий клиент, другая метка, другой sha256 эталона, другая эпоха, другие источники, test без `--final-test`, `kind` не lora — `ValueError` с понятной причиной.
  - `test_probe_without_lora_says_how_to_get_it`.
- [ ] **Step 2:** `pytest tests/test_downstream.py -v` — новые FAIL.
- [ ] **Step 3:** реализовать по Interfaces.
- [ ] **Step 4:** PASS; `pytest -q`.

### Task 5: CUDA-проверки и документация

**Files:**
- Test: `tests/test_lora.py` (CUDA-часть), `tests/test_finetune.py` (CUDA-часть)
- Modify: `README.md` (раздел 13), `audit/2026-11-02-usr-gap/README.md` (раздел 11), `docs/superpowers/specs/2026-11-02-lora-churn-design.md` (отклонения 1–5)

**Interfaces:**
- Consumes: Tasks 1–4; `world.flash_ready()`, `load_model(3, 512, 0.0, CUDA, "flash")` по образцу `tests/test_cuda_backbone.py:82`.

- [ ] **Step 1: CUDA-тесты** (`pytestmark = pytest.mark.cuda`, `skipif(not world.flash_ready())`):
  - `test_flash_and_sdpa_agree_with_adapters` — ненулевые `A`, `B`: `client_embeddings` под flash (bf16) и sdpa в допуске прохода целиком `tests/test_cuda_flash.py:238` (atol/rtol 5e-2).
  - `test_bf16_finetune_step_on_cuda` — один `fit` на CUDA: параметры fp32, градиенты только у заплаток и головы, потеря конечна, эталон побитно не изменился.
- [ ] **Step 2:** `pytest -m cuda tests/test_lora.py tests/test_finetune.py -v` — PASS (пропуск при отсутствии CUDA называть отдельно).
- [ ] **Step 3: Документация** — README §13: сценарий `lora` (четвёртый, решение владельца 2026-11-02), команды `finetune` и `probe --lora`, как читать (Δ к `usr` выше нуля — лучше замороженного вектора; интервал Δ к CatBoost не целиком ниже нуля — догнал; выше нуля — обогнал; у границы — ещё 2 seed), `--final-test` грузит готовый `lora.pt`; аудит §11 — ссылка на спек; спек — отклонения 1–5.
- [ ] **Step 4: Итоговая проверка** — `pytest -q` (число тестов, CUDA прошли/пропущены — отдельно); `python -m src.downstream.finetune --help` и `python -m src.downstream.probe --help` без ошибок; `git status` — изменены только перечисленные файлы, в `data/` ничего.

## Verification (после реализации, до GPU-прогона)

1. `pytest -q` — все тесты, включая CUDA (30+ CUDA-тестов не пропущены на RTX 3050).
2. Сквозной CPU-тест задачи 3 доказывает цепочку `finetune → predictions/meta → probe --lora` на синтетическом мире.
3. GPU-прогон — только по команде владельца: `python -m src.downstream.finetune --checkpoint data/12_train/best_checkpoint.pt --name lora-ref`, затем `python -m src.downstream.probe --tag ref-2026-10-09 --lora lora-ref`; в телеметрии смотреть `held_log_loss`, `delta_ratio` (рост > 0 с первых эпох) и `cuda_peak_allocated_gib` (ожидание ≤ 2 ГиБ при лимите 2,97).

## Риски GPU-прогона

- bf16 может «проглатывать» малые добавки в первых шагах (`W + ΔW` приводится к bf16 внутри линейного слоя): следить за `delta_ratio`.
- Отложенная часть читает весь `05_dataset/train` (8 500 клиентов) ради 1 538: `--workers 2`, смотреть время эпохи.
- Модель с параметризацией не сериализуется pickle: сохраняется только `adapter_state`; в `Prefetch` уходит только `Source`.
