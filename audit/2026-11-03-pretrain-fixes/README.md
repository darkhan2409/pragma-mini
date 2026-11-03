# Аудит 2026-11-03: исправления предобучения PRAGMA

Четыре кандидата, каждый — отдельным прогоном и решением владельца. Дизайн —
`docs/superpowers/specs/2026-11-03-pretrain-fixes-design.md`, основания — отчёты аудиторов в
`audit/2026-11-02-pretrain-contracts/` (A — цикл обучения, B — контракты данных, C — внутренности
модели, D — оценка, E — паритет информации, F — lineage) и `audit/2026-11-02-usr-gap/`.

Всё ниже запускает только владелец. Обучение, пересборка `data/03–07`, embed, plus_usr, probe и
шаг 0 сами не стартуют.

## Порядок

| Шаг | Что | Конфиги |
|---|---|---|
| 0 | шумовой пол CatBoost по seed | — |
| 1 | E1′: среднее по клиентам окна | `configs/train-aux-per-client.json` |
| 2 | перемешивание порядка клиентов | `configs/train-shuffle.json` |
| 3 | маска без дублей: группы полей, `event_type` только при маске события, скрытые ключи | `configs/masking-groups.json` + `configs/train-hide-keys.json` |
| 4 | шкала суммы по типу события | `configs/tokenizer-amount-by-type.json` |

Следующий кандидат — только после решения по предыдущему. Эталон для `--baseline` — текущий:
`ref-2026-10-09`, а после принятия кандидата — его имя.

## Шаг 0: шумовой пол

Только читает и печатает, `test` не читает. CatBoost есть в окружении `churn_baseline`:

```bash
cd churn_baseline
PYTHONPATH=.. .venv/bin/python ../audit/2026-11-03-pretrain-fixes/catboost_seeds.py --tag ref-2026-10-09
```

Seed 42 печатается первым как контроль и обязан дать PR-AUC val 0.570 (`catboost`) и 0.586
(`catboost_plus_usr`). По итогу владелец обновляет временный пол в README §13 (сейчас `usr` 0.025,
`catboost_plus_usr` 0.005).

## Прогон кандидата: `chain.sh`

```bash
setsid nohup bash audit/2026-11-03-pretrain-fixes/chain.sh <имя> <train-config> [masking-config] [baseline] &
```

`""` или `-` вместо `masking-config` — без него; `-` вместо `train-config` — значения по умолчанию.
`baseline` по умолчанию `ref-2026-10-09`. Конфиги — пути относительно корня репозитория.

Цепочка (5 эпох, seed 42, останов на первой ошибке):

1. `python -m src.mlm.train [--config …] [--masking-config …] --out data/runs/<имя>`
2. `python -m src.downstream.embed --checkpoint data/runs/<имя>/best_checkpoint.pt --tag <имя>`
3. из `churn_baseline/` его venv: `python -m churn.plus_usr --embeddings <корень>/data/13_downstream/<имя>`
   (у `plus_usr` флага `--tag` нет, вектора задаёт `--embeddings`)
4. `python -m src.downstream.probe --tag <имя> --baseline <baseline>`

**Замороженная копия кода.** Цепочка работает не с рабочим деревом, а с его снимком. Правка `src/`
посреди прогона иначе попадала бы в процессы-загрузчики DataLoader, которые заново импортируют
модули: 2026-11-03 прогон LoRA упал на третьей эпохе именно так. Что делает `chain.sh`:

1. Снимает рабочее дерево как объект-дерево git без коммита: временный индекс
   (`GIT_INDEX_FILE`), `git read-tree HEAD`, `git add -A` (новые файлы входят, `.gitignore`
   соблюдается), `git write-tree`.
2. `git archive <дерево> | tar -x -C data/runs/<имя>/code`.
3. Кладёт в копию ссылки `data`, `churn_baseline` и `.venv` на настоящие. `churn_baseline`
   заменяется ссылкой целиком: `plus_usr` пишет в его `reports/` и `models/`, а `probe` оттуда же
   читает `catboost_plus_usr`. Код churn поэтому не заморожен; в этом плане он не правится.
4. Каждый шаг идёт после `cd` в копию с `PYTHONPATH=<копия>` и `PYTHONUNBUFFERED=1`; перед
   обучением проверяется, что `import src` берёт модуль из копии, а не из редактируемой установки
   `.venv`.
5. Печатает и записывает действующие конфиги (см. «Принятые флаги» ниже).

До создания `data/runs/<имя>` `chain.sh` проверяет, что файлы конфигов существуют, что есть
`data/13_downstream/<baseline>/` и отчёт plus_usr эталона
`churn_baseline/reports/plus_usr/<baseline>/metrics.json`; иначе отказ с кодом 2. Если в копии на
месте `data`, `churn_baseline` или `.venv` уже что-то лежит, ссылка не кладётся — отказ.

Хеш дерева, коммит `HEAD` и путь копии записываются в `data/runs/<имя>/status`. Правки рабочего
дерева после старта в прогон не попадают; конфиги тоже читаются из копии.

В `data/runs/<имя>/` лежат: `status` (дерево, копия, коды выхода шагов с временем), по логу на шаг
(`train.log`, `embed.log`, `plus_usr.log`, `probe.log`), `code/` и то, что пишет обучение
(`best_checkpoint.pt`, `telemetry.jsonl`), а также `effective-config.json` — действующие
`MlmConfig` и `MaskingConfig` прогона (`{"mlm": …, "masking": …}`, `as_dict()`); путь к нему есть в
`status`.

**Принятые флаги.** Каждый конфиг кандидата несёт только свой флаг. Если принятый флаг не сделать
умолчанием в коде, следующий кандидат молча пойдёт без него. Поэтому владелец записывает принятое в
`audit/2026-11-03-pretrain-fixes/accepted.json` как `{"mlm": {"aux_per_client": true}, "masking": {}}`
(сейчас оба раздела пусты). Перед первым шагом `chain.sh` строит действующие конфиги, пишет их в
`effective-config.json` и сверяет с `accepted.json` из копии: ключ, которого нет или значение которого
другое, — `effective-config FAILED` в `status`, список расхождений (или ошибка конфига) в
`effective-config.log` и выход с кодом 2, до обучения.

**Отказ вместо перезаписи.** `chain.sh` не стартует, если уже есть `data/runs/<имя>`,
`data/13_downstream/<имя>` или `churn_baseline/reports/plus_usr/<имя>`. `probe --tag <имя>` пишет
`data/13_downstream/<имя>/report.json` — отчёт эталона `data/13_downstream/<baseline>/report.json`
он не трогает, потому что тег нового прогона другой. Не давайте новому прогону имя эталона.
Отказ после создания `data/runs/<имя>` (снимок, заморозка, сверка конфигов, любой шаг) оставляет
этот каталог: перед повтором под тем же именем его нужно удалить вручную.

Не запускайте `chain.sh` без команды владельца и не проверяйте его запуском «на пробу»: каждый
запуск — это настоящее обучение.

## Команды кандидатов

Кандидат 1, E1′ — среднее по клиентам окна (`aux_per_client`):

```bash
setsid nohup bash audit/2026-11-03-pretrain-fixes/chain.sh aux-per-client \
    audit/2026-11-03-pretrain-fixes/configs/train-aux-per-client.json - ref-2026-10-09 &
```

Кандидат 2, перемешивание (`shuffle_order`); `<эталон>` — `ref-2026-10-09` или принятый кандидат 1:

```bash
setsid nohup bash audit/2026-11-03-pretrain-fixes/chain.sh shuffle \
    audit/2026-11-03-pretrain-fixes/configs/train-shuffle.json - <эталон> &
```

Кандидат 3, маска без дублей (`field_groups`, `event_type_event_only`, `hide_event_keys`):

```bash
setsid nohup bash audit/2026-11-03-pretrain-fixes/chain.sh groups-hide-keys \
    audit/2026-11-03-pretrain-fixes/configs/train-hide-keys.json \
    audit/2026-11-03-pretrain-fixes/configs/masking-groups.json <эталон> &
```

Кандидат 4, шкала суммы по типу события (`split_by: "event_type"`). Сначала копия и пересборка,
потом цепочка.

1. Копия текущих `03–07` (~1,5 ГБ): `embed` и LoRA сверяют sha `05` эталона, а пересборка заменит
   каталоги. Без этой копии вернуться к эталону можно только полной пересборкой.

   ```bash
   mkdir -p data/runs/data-ref-2026-10-09
   cp -a data/03_vocab data/04_tokenized data/05_dataset data/06_embeddings data/07_backbone \
       data/runs/data-ref-2026-10-09/
   ```

2. Пересборка, ~35 мин, тоже **из замороженной копии**: словарь меняется на ходу этапов, и правка
   кода посреди пересборки дала бы данные от двух версий кода. Снимок как в `chain.sh`:

   ```bash
   CODE=data/runs/rebuild-amount-by-type/code
   TREE=$(GIT_INDEX_FILE=$(mktemp -u) sh -c 'git read-tree HEAD && git add -A && git write-tree')
   mkdir -p $CODE && git archive $TREE | tar -x -C $CODE
   rm -rf $CODE/churn_baseline
   ln -s $PWD/data $CODE/data; ln -s $PWD/churn_baseline $CODE/churn_baseline; ln -s $PWD/.venv $CODE/.venv
   (cd $CODE && PYTHONPATH=$PWD .venv/bin/python -m src.pipeline --from fit \
       --args fit="--config audit/2026-11-03-pretrain-fixes/configs/tokenizer-amount-by-type.json")
   ```

   До пересборки проверить, что делает `fit` с совпавшими границами у редких типов: на 16 различных
   границ значений может не хватить. После — число диапазонов по типам.

3. Цепочка. Обучение на пересобранном `05`, поэтому тот же код, что у принятого эталона.
   Какую пару масок брать:
   - кандидат 3 принят — `event_type_event_only` уже умолчание, маска не нужна (`-`);
   - **кандидат 3 отвергнут — обязательно `configs/masking-event-type-only.json`**: без
     `event_type_event_only` сумма выдавала бы цель `event_type` (спек §4). Поле остаётся в коде
     и после отказа от остального кандидата 3.

   ```bash
   setsid nohup bash audit/2026-11-03-pretrain-fixes/chain.sh amount-by-type - \
       <маска: - или audit/2026-11-03-pretrain-fixes/configs/masking-event-type-only.json> <эталон> &
   ```

   Если эталон `<эталон>` обучен на старых `05`, его векторы `probe --baseline` читает с диска и от
   `05` не зависят; а вот `embed` и LoRA эталона на новом `05` не пройдут сверку sha — для них
   нужна копия из шага 1.

LoRA: GPU-прогон LoRA на нынешнем эталоне — до кандидата 4 или после восстановления копии `03–07`
(`rm -rf data/03_vocab data/04_tokenized data/05_dataset data/06_embeddings data/07_backbone;
cp -a data/runs/data-ref-2026-10-09/* data/`). При смене эталона LoRA
перезапускается на новом по команде владельца. Правки `src/` на время такого прогона без замороженной
копии не вносить.

## Правило решения

README §13, «Правило принятия эксперимента». Кандидат сравнивается с текущим эталоном PRAGMA,
решает val, test не участвует. Δ — PR-AUC кандидата минус эталона, парный bootstrap по клиентам val,
сценарии `usr` и `catboost_plus_usr`. Принят, если хотя бы по одному из двух:

- нижняя граница 95% интервала Δ выше нуля, или
- Δ больше шумового пола, а log-loss сценария не выше, чем у эталона,

и ни по одному интервал Δ не лежит целиком ниже нуля. Шумовой пол — после шага 0 (временный:
`usr` 0.025, `catboost_plus_usr` 0.005). Δ у границы — решение ждёт повтора эталона с seed 43.

Что записать по кандидату: Δ и интервалы по `usr` и `catboost_plus_usr`, log-loss, вердикт по
правилу, телеметрия (`val_loss` там, где сравним: он сравним, если маска и словарь те же).

Для кандидата 1 дополнительно сравнить с эталоном телеметрию `clipped_share` и `grad_norm_mean` /
`grad_norm_max` (`telemetry.jsonl`, поля шага `grad_norm`). K на эпоху фиксирован, поэтому масштаб
градиента E1′ на шаге зависит от числа клиентов в micro-batch'е — от 0,53× до 5,3×. Клип срабатывает
на 49–86% шагов эталона, так что кандидат 1 заодно перевзвешивает MLM между шагами, и разницу нельзя
целиком относить к самой цели E1′.

## После решения

Решение — владельца; до него код не меняется.

- **Принят.** Флаг и его значение записываются в `accepted.json`. Флаг становится поведением по
  умолчанию, старая ветка кода и её тесты удаляются, имя прогона — новый эталон для `--baseline`
  следующих кандидатов. Конфиг кандидата и его строку в
  `tests/test_candidate_configs.py` убрать вместе с полем. README PRAGMA обновить по итогу решения.

  Снимая поле флага, оставить старые чекпойнты читаемыми: `MlmConfig.from_dict` и
  `MaskingConfig.from_dict` отвергают неизвестные ключи, а `as_dict` пишет в чекпойнт каждый новый
  ключ. Поэтому `from_dict` обязан и дальше принимать снятый ключ с его итоговым значением — например,
  множество `RETIRED = {ключ: значение}`, проверяемое в `from_dict`, а расхождение значения —
  `ConfigError`. Иначе чекпойнты этой кампании, включая новый эталон, перестанут грузиться в
  `load_trained`, `embed`, LoRA и resume. (Здесь только требование; `RETIRED` не реализован.)
- **Отвергнут.** Код переключателя удаляется, итог остаётся здесь, в таблице. Ключ при этом
  остаётся читаемым со значением по умолчанию — то же требование к `from_dict` (`RETIRED`), что и
  для принятого флага. Для кандидата 3
  исключение: `event_type_event_only` остаётся в коде для кандидата 4, остальное
  (`field_groups`, `hide_event_keys`, колонка отчёта маски) удаляется.
- **Кандидат 4 отвергнут** — вернуть `03–07` из `data/runs/data-ref-2026-10-09/`:

  ```bash
  rm -rf data/03_vocab data/04_tokenized data/05_dataset data/06_embeddings data/07_backbone
  cp -a data/runs/data-ref-2026-10-09/* data/
  ```

## Итоги

Δ — PR-AUC кандидата минус эталона, [95% интервал]; пусто до прогонов.

| Кандидат | Имя прогона | Эталон | Δ usr | Δ catboost_plus_usr | log-loss usr / plus_usr | val_loss | Вердикт |
|---|---|---|---|---|---|---|---|
| 0. шумовой пол | — | ref-2026-10-09 | | | | | |
| 1. aux_per_client | | | | | | | |
| 2. shuffle_order | | | | | | | |
| 3. группы + hide_event_keys | | | | | | | |
| 4. шкала суммы по типу | | | | | | | |
