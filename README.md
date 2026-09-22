# mini_pragma_v2

Синтетический генератор банковских событий и конвейер подготовки данных для
модели: генерация → препроцессинг → словарь → кодирование → датасет.

Каждый этап это отдельная команда с одним видимым результатом. Автоматической
сквозной цепочки нет: следующий шаг запускает человек, посмотрев на предыдущий.
Исключение одно — `tokenization.run fit`, который прогоняет подряд уже
существующие этапы обучения словаря и ничего своего не считает.

## Каталоги данных

```
data/
├── 01_raw/<group>/          выгрузка генератора: события и профиль
├── 02_preprocessed/<group>/ очищенная лента: events.parquet
├── 03_vocab/                чем кодируются данные: шесть файлов словаря
├── 04_tokenized/<group>/    результат кодирования: events.parquet, profile.parquet
└── 05_dataset/<group>/      готовые примеры: samples.parquet
```

Номер в имени каталога это порядок этапов конвейера: каждый следующий
читает предыдущий.

`<group>` это `train`, `val` или `test`. Клиенты групп не пересекаются, словарь
и все статистики учатся только на `train`.

## 1. Генерация

```bash
python -m src.generator.emit train
python -m src.generator.emit val
python -m src.generator.emit test
```

Посмотреть на выгрузку глазами: `python -m src.generator.report.show --help`.

Результат: `data/01_raw/<group>/` — лента событий и профиль клиентов. Конверт
события: `client_id`, `event_time`, `source`, `payload`; тип события лежит в
`payload.type`.

## 2. Препроцессинг

```bash
python -m src.preprocessing.run preprocess train
python -m src.preprocessing.run preprocess val
python -m src.preprocessing.run preprocess test
```

Проверяет RAW по контракту и раскрывает payload в типизированные колонки.
Результат группы это ровно один файл: `events.parquet`. Анкета не
копируется — чистить в ней нечего, и следующие этапы читают её прямо
из `data/01_raw/<group>/profile.parquet`. Любая строка, которую нельзя разобрать
по контракту, останавливает этап, и частичный результат не сохраняется.

## 3. Словарь

Все словари, границы чисел и BPE учатся только на train: лента из
`data/02_preprocessed/train/events.parquet`, анкета из `data/01_raw/train/profile.parquet`.

```bash
python -m src.tokenization.run special-tokens   # data/03_vocab/special_tokens.json
python -m src.tokenization.run key-vocab        # data/03_vocab/key_vocab.json
python -m src.tokenization.run value-vocab      # data/03_vocab/value_vocab.json
python -m src.tokenization.run buckets          # data/03_vocab/buckets.json
python -m src.tokenization.run bpe              # data/03_vocab/bpe.json
python -m src.tokenization.run final-vocab      # data/03_vocab/final_vocab.json
```

Для прода те же шесть этапов запускаются одной командой:

```bash
python -m src.tokenization.run fit
```

`fit` вызывает те же функции, что и отдельные команды, поэтому результат
совпадает. При ошибке он останавливается и называет проблемный этап;
`final_vocab.json` при этом не остаётся от прежней сборки.

Каждый файл решает ровно одну задачу и не повторяет содержимое соседних.

`special_tokens.json` — служебные токены и их ID. Их пять:

```json
{"[PAD]": 0, "[UNK]": 1, "[MASK]": 2, "[EVT]": 3, "[USR]": 4}
```

`key_vocab.json` — название ключа и его ID:

```json
{"accrual_period": 5, "amount_due": 6, "event_type": 29}
```

`value_vocab.json` — категории train, сгруппированные по ключу. Идентификатор
определяет связка ключ + значение: одинаковая запись у разных ключей это разные
факты.

```json
{"app_domain": {"auth": 118, "home": 119}}
```

`buckets.json` — диапазоны каждого числового ключа. `min` включительно, `max`
не включительно; `null` означает открытую сторону, а `min = max = 0` —
отдельный нулевой диапазон, который проверяется первым.

```json
{"amount_due": {"amount_due_bucket_1": {"id": 283, "min": null, "max": 1000}}}
```

`bpe.json` — стандартный файл библиотеки `tokenizers`, записанный её же
`Tokenizer.save`, и читаемый обратно `Tokenizer.from_file`. Настройки обучения
живут в конфигурации, а не в файле.

`final_vocab.json` — уникальное имя токена и его глобальный ID. Префикс отделяет
виды токенов друг от друга:

```json
{"[PAD]": 0, "key:amount_due": 6, "value:app_domain=auth": 118,
 "bucket:amount_due_bucket_1": 283, "bpe:a": 470}
```

ID идут подряд от нуля: специальные, ключи, категории, диапазоны, куски BPE.
После сборки словарь не меняется.

## 4. Кодирование групп

```bash
python -m src.tokenization.run encode train
python -m src.tokenization.run encode val
python -m src.tokenization.run encode test
```

Вход: `data/02_preprocessed/<group>/events.parquet`, `data/01_raw/<group>/profile.parquet`
и словарь из `data/03_vocab/`. Выход: `data/04_tokenized/<group>/events.parquet` и
`profile.parquet`.

Ничего не дообучается: значение, которого на train не было, кодируется `[UNK]`.
Отсутствующее поле в последовательность не попадает вовсе, пустой после
нормализации текст — тоже.

```
events.parquet   client_id, event_time, key_ids, value_ids, positions,
                 value_starts, value_lengths, calendar
profile.parquet  client_id, key_ids, value_ids, positions,
                 value_starts, value_lengths
```

`value_starts` и `value_lengths` обозначают границы одного значения, которое
может состоять из нескольких кусков BPE; `positions` это номер куска внутри
значения. Числа токенов и названия ключей не хранятся: первое считается по
длине массивов, второе восстанавливается словарём.

## 5. Датасет

```bash
python -m src.dataset.run train
python -m src.dataset.run val
python -m src.dataset.run test
```

Вход: `data/04_tokenized/<group>/` и словарь из `data/03_vocab/`.
Результат: `data/05_dataset/<group>/samples.parquet`. Строка это один клиент на
конечный cutoff своей группы.

```
client_id
key_ids, value_ids, positions          последовательность событий клиента
event_starts, event_lengths            границы каждого события
event_time, calendar                   время событий и календарный канал
value_starts, value_lengths            границы значений, включая составные
target_event_mask                      что разрешено маскировать и предсказывать
profile_key_ids, profile_value_ids,    токены профиля
profile_positions,
profile_value_starts,
profile_value_lengths                  границы значений профиля
```

Границы значений считаются от начала общей последовательности клиента, поэтому
принадлежность значения событию видна по `event_starts`, а ключ значения берётся
из `key_ids` по его первой позиции. Числа событий, токенов и значений это длины
массивов, и отдельными колонками они не хранятся.

`target_event_mask` это период целей СВОЕЙ группы: train разрешает свой период,
val только свой, test только свой. Старая история остаётся видимым контекстом,
но целью чужой группы не становится. Клиенты групп не пересекаются.

Длинная история усекается объявленной политикой контекста; для оценочных групп
потеря событий периода целей запрещена и останавливает сборку.

Маскирование и обучение в конвейер не входят.

## Установка

```bash
pip install -e .
```

Python 3.11+. Зависимости: numpy, pyarrow, pandas, scikit-learn, scipy,
tokenizers.
