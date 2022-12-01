#!/usr/bin/env bash

set -e

LOGFILE="${LOGFILE:-log.txt}"
TB_RUN="${TB_RUN:-}"

usage() {
    echo "Usage: TRAIN_DATA=? DEV_DATA=? [TB_RUN=?] [LOG_FILE=?] $0 [-h|--help]"
    exit 1
}

# Check for help flag and print usage
for arg in "$@"; do
    if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
        usage
    fi
done

# TRAIN_DATA and DEV_DATA are mandatory
if [ -z "$TRAIN_DATA" ] || [ -z "$DEV_DATA" ]; then
    usage
fi

echo date >> "$LOGFILE"

python train.py \
    "$TRAIN_DATA" \
    "$DEV_DATA" \
    0 1 \
    bert-base-cased \
    --separator " " \
    --device "cuda:0" \
    --fine_tune \
    --tb-run "$TB_RUN" \
    2>&1 | tee -a "$LOGFILE"
