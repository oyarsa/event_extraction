#!/usr/bin/env bash

set -e

LOGFILE="${LOGFILE:-train.log}"

RUN_NAME="${RUN_NAME:-}"
LR="${LR:-5e-5}"
LM_NAME="${LM_NAME:-bert-base-cased}"

usage() {
    echo "Usage: TRAIN_DATA=? DEV_DATA=? [TB_RUN=?] [LOG_FILE=?] $0 [-h|--help] [ARGS]"
    printf "\t[ARGS] are passed to train.py\n"
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


printf "\n>>>>>>>>>> %s\n\n" "$(date)" >> "$LOGFILE"

python train.py \
    "$TRAIN_DATA" \
    "$DEV_DATA" \
    0 1 \
    "$LM_NAME" \
    --separator " " \
    --device "cuda:0" \
    --fine_tune \
    --run-name "$RUN_NAME" \
    --logfile "$LOGFILE" \
    --lr "$LR" \
    "$@"
