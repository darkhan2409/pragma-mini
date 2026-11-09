#!/usr/bin/env bash
# LoRA-дообучение эталона под churn_active90 и сравнение: finetune → probe --lora.
#
#   setsid nohup bash audit/2026-11-05-generator-realism/lora.sh <имя> <эталон> [seed] > /dev/null 2>&1 &
#
#   <имя>     имя дообучения: data/runs/<имя> (lora.pt, telemetry), прогнозы data/13_downstream/<имя>
#   <эталон>  прогон предобучения night.sh: data/runs/<эталон>/best_checkpoint.pt и его векторы
#             data/13_downstream/<эталон> (probe сравнивает LoRA с catboost и usr этого эталона)
#   [seed]    seed заплаток, порядка и dropout (по умолчанию 0)
#
# Конфиг дообучения — умолчания FinetuneConfig, как у lora-ref (README, «Сценарий lora»).
# Код — из ЗАМОРОЖЕННОЙ копии рабочего дерева, как в night.sh. Журнал шагов (status, логи, пиковая
# память) — в data/runs/<имя>-chain: каталог data/runs/<имя> пишет и проверяет сам finetune.

set -uo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "использование: lora.sh <имя> <эталон> [seed]" >&2
    exit 2
fi

NAME=$1
BASE=$2
SEED=${3:-0}

for value in "$NAME" "$BASE"; do
    if ! [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "[lora] недопустимое имя: $value" >&2
        exit 2
    fi
done

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
CHAIN=$REPO/data/runs/$NAME-chain
COPY=$CHAIN/code
CHECKPOINT=data/runs/$BASE/best_checkpoint.pt
PY=$REPO/.venv/bin/python
TIME=/usr/bin/time

cd "$REPO" || exit 2

for path in "$CHAIN" "$REPO/data/runs/$NAME" "$REPO/data/13_downstream/$NAME"; do
    if [ -e "$path" ]; then
        echo "[lora] $path уже существует: другое имя или удалите его вручную" >&2
        exit 2
    fi
done

if [ ! -f "$REPO/$CHECKPOINT" ] || [ ! -f "$REPO/data/13_downstream/$BASE/meta.json" ]; then
    echo "[lora] нет чекпойнта $CHECKPOINT или векторов data/13_downstream/$BASE эталона" >&2
    exit 2
fi

if pgrep -f "[s]rc.generator.emit|[s]rc.mlm.train|[s]rc.pipeline|[s]rc.downstream.finetune" > /dev/null; then
    echo "[lora] уже идёт генерация, конвейер или обучение" >&2
    exit 2
fi

mkdir -p "$REPO/data/runs" && mkdir "$CHAIN" || exit 2

STATUS=$CHAIN/status

note() {
    echo "$* $(date -Is)" >> "$STATUS"
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

TREE=$(
    export GIT_INDEX_FILE=$TMP/index
    git read-tree HEAD && git add -A && git write-tree
)

[ -n "$TREE" ] || { note "snapshot FAILED"; exit 1; }

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
    echo "start $(date -Is) name=$NAME base=$BASE seed=$SEED"
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
    *) note "freeze FAILED src импортируется из ${IMPORTED:-ничего}"; exit 1 ;;
esac

step() {
    local label=$1
    shift
    note "$label start"
    "$TIME" -v -o "$CHAIN/$label.time" "$@" > "$CHAIN/$label.log" 2>&1
    local result=$?
    local peak
    peak=$(awk -F': ' '/Maximum resident set size/ {printf "%.2f GB", $2 / 1048576}' "$CHAIN/$label.time" 2>/dev/null)
    note "$label exit $result peak ${peak:-?}"
    if [ "$result" -ne 0 ]; then
        echo "[lora] шаг $label завершился с кодом $result, см. $CHAIN/$label.log" >&2
        exit "$result"
    fi
}

step finetune "$PY" -m src.downstream.finetune --checkpoint "$CHECKPOINT" --name "$NAME" --seed "$SEED"
step probe "$PY" -m src.downstream.probe --tag "$BASE" --lora "$NAME"

note "done"
