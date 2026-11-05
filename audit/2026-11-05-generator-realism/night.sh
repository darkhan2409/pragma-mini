#!/usr/bin/env bash
# Ночная цепочка нового мира (генератор 2.0, аудит реализма 2026-11-05): от генерации до оценки.
#
#   setsid nohup bash audit/2026-11-05-generator-realism/night.sh <имя> > /dev/null 2>&1 &
#
#   <имя>   имя прогона: data/runs/<имя> (код, логи, обучение), векторы data/13_downstream/<имя>
#
# Шаги:
#   backup    прежние data/01–07 и хвост churn_baseline/data/future/train переносятся в
#             data/runs/<имя>-before/ (mv в той же файловой системе: мгновенно, ничего не теряется)
#   gen_*     генерация data/01_raw/train, val, test (размеры — config.DATASETS)
#   future    продолжение train на 60 суток — метки churn train (churn_baseline/README.md)
#   pipeline  этапы 02–07 (python -m src.pipeline)
#   train     предобучение, умолчания кода (accepted.json аудита 2026-11-03 пуст), EPOCHS эпох
#   embed     векторы [USR] на T для train и val
#   churn_*   признаки и CatBoost churn-бейзлайна на новом мире
#   plus_usr  CatBoost на признаках + [USR]
#   probe     сравнение catboost, usr, catboost_plus_usr на val
#
# Код — из ЗАМОРОЖЕННОЙ копии рабочего дерева (память проекта: правка src/ посреди прогона
# попадает в процессы-загрузчики). data, churn_baseline и .venv в копии — ссылки на настоящие.
# Каждый шаг идёт под /usr/bin/time -v: в status — код выхода и пиковая память. Первая ошибка
# останавливает цепочку; продолжить можно с упавшего шага вручную (генерация — --resume,
# конвейер — --from <этап>, обучение — --resume).
#
# Переменные для пробного прогона на крошечном мире: EPOCHS (по умолчанию 5), MAX_STEPS.

set -uo pipefail

if [ "$#" -ne 1 ]; then
    echo "использование: night.sh <имя>" >&2
    exit 2
fi

NAME=$1
EPOCHS=${EPOCHS:-5}
MAX_STEPS=${MAX_STEPS:-}

if ! [[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "[night] недопустимое имя прогона: $NAME" >&2
    exit 2
fi

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
RUN=$REPO/data/runs/$NAME
BACKUP=$REPO/data/runs/$NAME-before
COPY=$RUN/code
VECTORS=$REPO/data/13_downstream/$NAME
PY=$REPO/.venv/bin/python
CHURN_PY=$REPO/churn_baseline/.venv/bin/python
TIME=/usr/bin/time

cd "$REPO" || exit 2

for path in "$RUN" "$BACKUP" "$VECTORS" "$REPO/churn_baseline/reports/plus_usr/$NAME"; do
    if [ -e "$path" ]; then
        echo "[night] $path уже существует: другое имя или удалите его вручную" >&2
        exit 2
    fi
done

for interpreter in "$PY" "$CHURN_PY" "$TIME"; do
    if [ ! -x "$interpreter" ]; then
        echo "[night] нет $interpreter" >&2
        exit 2
    fi
done

# Второй долгий прогон рядом съел бы память и GPU. [x]-шаблон не находит сам pgrep.
if pgrep -f "[s]rc.generator.emit|[s]rc.mlm.train|[s]rc.pipeline|[s]rc.downstream.finetune" > /dev/null; then
    echo "[night] уже идёт генерация, конвейер или обучение" >&2
    exit 2
fi

mkdir -p "$REPO/data/runs" && mkdir "$RUN" || exit 2

STATUS=$RUN/status

note() {
    echo "$* $(date -Is)" >> "$STATUS"
}

# ---------- снимок рабочего дерева ----------

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

TREE=$(
    export GIT_INDEX_FILE=$TMP/index
    git read-tree HEAD && git add -A && git write-tree
)

if [ -z "$TREE" ]; then
    note "snapshot FAILED"
    exit 1
fi

mkdir "$COPY" && git archive "$TREE" | tar -x -C "$COPY" || { note "archive FAILED"; exit 1; }

rm -rf "$COPY/churn_baseline"
for link in data churn_baseline .venv; do
    if [ -e "$COPY/$link" ] || [ -L "$COPY/$link" ]; then
        note "link FAILED $COPY/$link уже существует"
        exit 1
    fi
    ln -s "$REPO/$link" "$COPY/$link" || { note "link FAILED $link"; exit 1; }
done

{
    echo "start $(date -Is) name=$NAME epochs=$EPOCHS max_steps=${MAX_STEPS:-нет}"
    echo "tree $TREE"
    echo "head $(git rev-parse HEAD)"
    echo "code $COPY"
} > "$STATUS"

export PYTHONPATH=$COPY
export PYTHONUNBUFFERED=1
cd "$COPY" || exit 1

IMPORTED=$("$PY" -c "import src; print(src.__file__)" 2>/dev/null)
case "$IMPORTED" in
    "$COPY"/*) ;;
    *)
        note "freeze FAILED src импортируется из ${IMPORTED:-ничего}"
        exit 1
        ;;
esac

"$PY" -c "from src.generator import config; print('datasets', {k: v.clients for k, v in config.DATASETS.items()}, 'version', config.GENERATOR_VERSION)" >> "$STATUS"

# step <шаг> <каталог> <команда…>: лог <шаг>.log, пиковая память и код — в status, ошибка — стоп.
step() {
    local label=$1 directory=$2
    shift 2
    note "$label start"
    (cd "$directory" && "$TIME" -v -o "$RUN/$label.time" "$@") > "$RUN/$label.log" 2>&1
    local result=$?
    local peak
    peak=$(awk -F': ' '/Maximum resident set size/ {printf "%.2f GB", $2 / 1048576}' "$RUN/$label.time" 2>/dev/null)
    note "$label exit $result peak ${peak:-?}"
    if [ "$result" -ne 0 ]; then
        echo "[night] шаг $label завершился с кодом $result, см. $RUN/$label.log" >&2
        exit "$result"
    fi
}

# ---------- прежние данные — в сторону, без удаления ----------

note "backup start"
mkdir "$BACKUP" || { note "backup FAILED"; exit 1; }
for stage in 01_raw 02_preprocessed 03_vocab 04_tokenized 05_dataset 06_embeddings 07_backbone; do
    if [ -e "$REPO/data/$stage" ]; then
        mv "$REPO/data/$stage" "$BACKUP/$stage" || { note "backup FAILED $stage"; exit 1; }
    fi
done
if [ -e "$REPO/churn_baseline/data/future/train" ]; then
    mkdir -p "$BACKUP/churn_future" && mv "$REPO/churn_baseline/data/future/train" "$BACKUP/churn_future/train" \
        || { note "backup FAILED churn future"; exit 1; }
fi
note "backup done $BACKUP"

# ---------- данные ----------

step gen_train "$COPY" "$PY" -m src.generator.emit train
step gen_val "$COPY" "$PY" -m src.generator.emit val
step gen_test "$COPY" "$PY" -m src.generator.emit test
step future "$COPY" "$PY" -m src.generator.continuation train --days 60 --out churn_baseline/data/future/train
step pipeline "$COPY" "$PY" -m src.pipeline --report "$RUN/pipeline-timing.json"

# ---------- модель и оценка ----------

TRAIN_ARGS=(--epochs "$EPOCHS" --out "data/runs/$NAME")
[ -n "$MAX_STEPS" ] && TRAIN_ARGS+=(--max-steps "$MAX_STEPS")

step train "$COPY" "$PY" -m src.mlm.train "${TRAIN_ARGS[@]}"
step embed "$COPY" "$PY" -m src.downstream.embed --checkpoint "data/runs/$NAME/best_checkpoint.pt" --tag "$NAME"
step churn_build_train "$COPY/churn_baseline" "$CHURN_PY" -m churn.build train
step churn_build_val "$COPY/churn_baseline" "$CHURN_PY" -m churn.build val
step churn_train "$COPY/churn_baseline" "$CHURN_PY" -m churn.train
step plus_usr "$COPY/churn_baseline" "$CHURN_PY" -m churn.plus_usr --embeddings "$VECTORS"
step probe "$COPY" "$PY" -m src.downstream.probe --tag "$NAME"

note "done"
