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
├── raw/<group>/            выгрузка генератора: события и профиль
├── preprocessed/<group>/   очищенная группа: events.parquet, profile.parquet
├── tokenizer/              словарь по этапам и итоговый tokenizer.json
├── tokenized/<group>/      закодированная группа: events.parquet, profile.parquet
└── dataset/<group>/        готовые примеры: samples.parquet
```

`<group>` это `train`, `val` или `test`. Клиенты групп не пересекаются, словарь
и все статистики учатся только на `train`.

## 1. Генерация

```bash
python -m src.generator.emit train
python -m src.generator.emit val
python -m src.generator.emit test
```

Посмотреть на выгрузку глазами: `python -m src.generator.report.show --help`.

Результат: `data/raw/<group>/` — лента событий и профиль клиентов. Конверт
события: `client_id`, `event_time`, `source`, `payload`; тип события лежит в
`payload.type`.

## 2. Препроцессинг

```bash
python -m src.preprocessing.run preprocess train
python -m src.preprocessing.run preprocess val
python -m src.preprocessing.run preprocess test
```

Проверяет RAW по контракту и раскрывает payload в типизированные колонки.
Результат группы это ровно два файла: `events.parquet` и `profile.parquet`.
Любая строка, которую нельзя разобрать по контракту, останавливает этап, и
частичный результат не сохраняется.

## 3. Словарь

Все словари, границы чисел и BPE учатся только на `data/preprocessed/train`.

```bash
python -m src.tokenization.run special-tokens   # data/tokenizer/special_tokens.json
python -m src.tokenization.run key-vocab        # data/tokenizer/key_vocab.json
python -m src.tokenization.run value-vocab      # data/tokenizer/value_vocab.json
python -m src.tokenization.run buckets          # data/tokenizer/buckets.json
python -m src.tokenization.run bpe              # data/tokenizer/bpe.json
python -m src.tokenization.run final-vocab      # data/tokenizer/tokenizer.json
```

Для прода те же шесть этапов запускаются одной командой:

```bash
python -m src.tokenization.run fit
```

`fit` вызывает те же функции, что и отдельные команды, поэтому результат
совпадает. При ошибке он останавливается и называет проблемный этап;
`tokenizer.json` при этом не остаётся от прежней сборки.

Что в каком файле:

| файл | что содержит |
|---|---|
| `special_tokens.json` | служебные токены, их ID и назначение |
| `key_vocab.json` | все поля, поступающие в модель, и их ID |
| `value_vocab.json` | категориальные значения train, их ID, частоты, связь с ключом |
| `buckets.json` | метод, границы диапазонов и токены каждого числового ключа |
| `bpe.json` | разбиение текста train со всем нужным для кодирования и декодирования |
| `tokenizer.json` | единое пространство ID и таблица всех токенов |

Пространство ID непересекающимися диапазонами:
`специальные → ключи → категории → диапазоны чисел → BPE`. После сборки
`tokenizer.json` словарь не меняется.

## 4. Кодирование групп

```bash
python -m src.tokenization.run encode train
python -m src.tokenization.run encode val
python -m src.tokenization.run encode test
```

Вход: `data/preprocessed/<group>/` и `data/tokenizer/tokenizer.json`. Выход:
`data/tokenized/<group>/events.parquet` и `profile.parquet`. Ничего не
дообучается: значение, которого на train не было, кодируется специальным
токеном. `client_id` и `event_time` сохраняются, границы записей едут
массивами начал и длин.

## 5. Датасет

```bash
python -m src.dataset.run train
python -m src.dataset.run val
python -m src.dataset.run test
```

Результат: `data/dataset/<group>/samples.parquet`. Строка это один клиент на
конечный cutoff своей группы: токены событий по времени, токены профиля,
границы событий и значений, каналы времени, маска допустимых целей, вес и
служебные признаки. Длинная история усекается объявленной политикой контекста,
и усечение названо в самой строке.

Маскирование и обучение в конвейер не входят.

## Установка

```bash
pip install -e .
```

Python 3.11+. Зависимости: numpy, pyarrow, pandas, scikit-learn, scipy,
tokenizers.
