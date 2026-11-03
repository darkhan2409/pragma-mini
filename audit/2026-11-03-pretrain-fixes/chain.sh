#!/usr/bin/env bash
# Цепочка одного кандидата аудита 2026-11-03: обучение → embed → plus_usr → probe.
#
#   setsid nohup bash audit/2026-11-03-pretrain-fixes/chain.sh <имя> <train-config> [masking-config] [baseline] &
#
#   <имя>            имя прогона: data/runs/<имя>, тег векторов data/13_downstream/<имя>
#   <train-config>   JSON MlmConfig для --config; "-" — без него (значения по умолчанию)
#   [masking-config] JSON MaskingConfig для --masking-config; "" или "-" — без него
#   [baseline]       тег эталона для probe --baseline (по умолчанию ref-2026-10-09)
#
# Конфиги — относительные пути внутри репозитория: они читаются из замороженной копии.
#
# Код берётся из ЗАМОРОЖЕННОЙ копии рабочего дерева, а не из самого дерева. Правка src/ посреди
# прогона иначе попадала бы в процессы-загрузчики DataLoader, которые заново импортируют модули
# (так 2026-11-03 упал прогон LoRA). Снимок — объект-дерево git без коммита: временный индекс,
# read-tree HEAD, add -A (новые файлы входят, .gitignore соблюдается), write-tree. Из него
# git archive собирает data/runs/<имя>/code; data, churn_baseline и .venv там — ссылки на
# настоящие. churn_baseline ссылкой целиком, а не копией: plus_usr пишет в его reports/ и
# models/, оттуда же probe читает catboost_plus_usr; код churn в этом плане не правится.
#
# В data/runs/<имя>/ ложатся логи шагов (<шаг>.log), status (дерево, копия, коды выхода) и всё,
# что пишет обучение (--out). Первая ошибка останавливает цепочку. Существующий прогон,
# каталог векторов или отчёт plus_usr с тем же именем — отказ: ничего не перезаписывается.

set -uo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    echo "использование: chain.sh <имя> <train-config> [masking-config] [baseline]" >&2
    exit 2
fi

NAME=$1
TRAIN_CONFIG=$2
MASKING_CONFIG=${3:-}
BASELINE=${4:-ref-2026-10-09}

[ "$MASKING_CONFIG" = "-" ] && MASKING_CONFIG=""
[ "$TRAIN_CONFIG" = "-" ] && TRAIN_CONFIG=""

# Имя попадает в пути и теги: только безопасные символы.
if ! [[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "[chain] недопустимое имя прогона: $NAME" >&2
    exit 2
fi

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
RUN=$REPO/data/runs/$NAME
COPY=$RUN/code
VECTORS=$REPO/data/13_downstream/$NAME
PLUS_USR_REPORT=$REPO/churn_baseline/reports/plus_usr/$NAME
PY=$REPO/.venv/bin/python
CHURN_PY=$REPO/churn_baseline/.venv/bin/python

cd "$REPO" || exit 2

for path in "$RUN" "$VECTORS" "$PLUS_USR_REPORT"; do
    if [ -e "$path" ]; then
        echo "[chain] $path уже существует: другое имя или удалите его вручную" >&2
        exit 2
    fi
done

# Конфиги и эталон проверяются до создания прогона: иначе опечатка стоила бы часов обучения.
for config in "$TRAIN_CONFIG" "$MASKING_CONFIG"; do
    # Путь относительно корня репозитория (cd $REPO выше) или абсолютный.
    if [ -n "$config" ] && [ ! -f "$config" ]; then
        echo "[chain] нет файла конфига $config (корень $REPO)" >&2
        exit 2
    fi
done

# probe --baseline читает прогнозы эталона, а не только его каталог.
if [ ! -f "$REPO/data/13_downstream/$BASELINE/predictions.parquet" ]; then
    echo "[chain] нет прогнозов эталона $REPO/data/13_downstream/$BASELINE/predictions.parquet" >&2
    exit 2
fi

if [ ! -f "$REPO/churn_baseline/reports/plus_usr/$BASELINE/metrics.json" ]; then
    echo "[chain] нет отчёта plus_usr эталона $REPO/churn_baseline/reports/plus_usr/$BASELINE/metrics.json" >&2
    exit 2
fi

for interpreter in "$PY" "$CHURN_PY"; do
    if [ ! -x "$interpreter" ]; then
        echo "[chain] нет интерпретатора $interpreter" >&2
        exit 2
    fi
done

# mkdir без -p: если каталог успел появиться, это отказ, а не слияние.
mkdir "$RUN" || exit 2

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
    echo "[chain] не удалось снять рабочее дерево" >&2
    exit 1
fi

mkdir "$COPY" && git archive "$TREE" | tar -x -C "$COPY"
code=$?

if [ "$code" -ne 0 ]; then
    note "archive FAILED exit $code"
    exit "$code"
fi

# Тяжёлое и общее — ссылками на настоящее. churn_baseline из архива заменяется ссылкой.
rm -rf "$COPY/churn_baseline"
for link in data churn_baseline .venv; do
    # Если путь уже есть (например, .venv попал в архив), ln -s положил бы ссылку внутрь него.
    if [ -e "$COPY/$link" ] || [ -L "$COPY/$link" ]; then
        note "link FAILED $COPY/$link уже существует"
        echo "[chain] $COPY/$link уже существует: ссылку класть некуда" >&2
        exit 1
    fi
    ln -s "$REPO/$link" "$COPY/$link" || { note "link FAILED $link"; exit 1; }
done

echo "start $(date -Is) name=$NAME baseline=$BASELINE" > "$STATUS"
echo "tree $TREE" >> "$STATUS"
echo "head $(git rev-parse HEAD)" >> "$STATUS"
echo "code $COPY" >> "$STATUS"

export PYTHONPATH=$COPY
export PYTHONUNBUFFERED=1
cd "$COPY" || exit 1

# Конфиг читается из копии: абсолютный путь внутри репозитория переводится в путь копии.
in_copy() {
    local path=${1#"$REPO"/}
    if [ ! -f "$COPY/$path" ] && [ ! -f "$path" ]; then
        echo "[chain] нет файла конфига $1" >&2
        return 1
    fi
    [ -f "$COPY/$path" ] && echo "$path" || echo "$1"
}

TRAIN_ARGS=()

if [ -n "$TRAIN_CONFIG" ]; then
    TRAIN_CONFIG=$(in_copy "$TRAIN_CONFIG") || { note "config FAILED $TRAIN_CONFIG"; exit 2; }
    TRAIN_ARGS+=(--config "$TRAIN_CONFIG")
fi

if [ -n "$MASKING_CONFIG" ]; then
    MASKING_CONFIG=$(in_copy "$MASKING_CONFIG") || { note "config FAILED $MASKING_CONFIG"; exit 2; }
    TRAIN_ARGS+=(--masking-config "$MASKING_CONFIG")
fi

echo "train-config ${TRAIN_CONFIG:-нет}" >> "$STATUS"
echo "masking-config ${MASKING_CONFIG:-нет}" >> "$STATUS"

# Контроль заморозки: src должен импортироваться из копии, а не из редактируемой установки.
IMPORTED=$("$PY" -c "import src; print(src.__file__)" 2>/dev/null)
case "$IMPORTED" in
    "$COPY"/*) ;;
    *)
        note "freeze FAILED src импортируется из ${IMPORTED:-ничего}"
        echo "[chain] src импортируется не из копии: $IMPORTED" >&2
        exit 1
        ;;
esac

# Действующие конфиги: каждый кандидат несёт только свой флаг, поэтому принятые флаги обязаны
# быть умолчаниями в коде. accepted.json (из копии) — что владелец принял; расхождение с
# действующим значением — отказ до обучения. Что реально применено — в effective-config.json.
"$PY" - "$RUN/effective-config.json" "$COPY/audit/2026-11-03-pretrain-fixes/accepted.json" \
    "${TRAIN_CONFIG:-}" "${MASKING_CONFIG:-}" 2> "$RUN/effective-config.log" <<'PYEOF'
import json
import sys
from pathlib import Path

from src.masking.settings import MaskingConfig
from src.mlm.settings import MlmConfig

out, accepted_path, train_path, masking_path = sys.argv[1:5]

effective = {
    "mlm": MlmConfig.load(Path(train_path) if train_path else None).as_dict(),
    "masking": MaskingConfig.load(Path(masking_path) if masking_path else None).as_dict(),
}
Path(out).write_text(json.dumps(effective, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

accepted = json.loads(Path(accepted_path).read_text(encoding="utf-8"))
problems = []

for section, flags in accepted.items():
    for key, value in flags.items():
        if key not in effective.get(section, {}):
            problems.append(f"{section}.{key}: принято {value!r}, в действующем конфиге ключа нет")
        elif effective[section][key] != value:
            problems.append(f"{section}.{key}: принято {value!r}, действует {effective[section][key]!r}")

if problems:
    print("[chain] принятые флаги не действуют (сделайте их умолчанием или добавьте в конфиг):", file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    sys.exit(1)
PYEOF
if [ "$?" -ne 0 ]; then
    note "effective-config FAILED (расхождение с accepted.json или ошибка конфига)"
    echo "[chain] проверка действующего конфига не прошла, см. $RUN/effective-config.log" >&2
    exit 2
fi

echo "effective-config $RUN/effective-config.json" >> "$STATUS"

# step <имя шага> <каталог запуска> <команда…>: лог — <шаг>.log, код — в status, при ошибке стоп.
step() {
    local label=$1 directory=$2
    shift 2
    note "$label start"
    (cd "$directory" && "$@") > "$RUN/$label.log" 2>&1
    local result=$?
    note "$label exit $result"
    if [ "$result" -ne 0 ]; then
        echo "[chain] шаг $label завершился с кодом $result, см. $RUN/$label.log" >&2
        exit "$result"
    fi
}

step train "$COPY" "$PY" -m src.mlm.train ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"} --out "data/runs/$NAME"
step embed "$COPY" "$PY" -m src.downstream.embed --checkpoint "data/runs/$NAME/best_checkpoint.pt" --tag "$NAME"
step plus_usr "$COPY/churn_baseline" "$CHURN_PY" -m churn.plus_usr --embeddings "$VECTORS"
step probe "$COPY" "$PY" -m src.downstream.probe --tag "$NAME" --baseline "$BASELINE"

note "done"
