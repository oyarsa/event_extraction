#!/usr/bin/env fish

set -q LOGFILE; or set LOGFILE "train.log"
set -q RUN_NAME; or set RUN_NAME ""
set -q LR; or set LR 5e-5
set -q LM_NAME; or set LM_NAME bert-base-cased
set -q BATCH_SIZE; or set BATCH_SIZE 32
set -q EPOCHS; or set EPOCHS 4
set -q CRF; or set CRF 0

function usage
    echo "Usage: TRAIN_DATA=? DEV_DATA=? [RUN_NAME=?] [LM_NAME] [LOG_FILE=?] $fish_script [-h|--help] [...ARGS]"
    printf "\t[ARGS] are passed to train.py\n"
    exit 1
end

# Check for help flag and print usage
for arg in $argv
    if test "$arg" = -h -o "$arg" = --help
        usage
    end
end

# TRAIN_DATA and DEV_DATA are mandatory
if not set -q TRAIN_DATA; or not set -q DEV_DATA
    usage
end

test (uname) = Darwin; and set device mps; or set device cuda

printf "\n>>>>>>>>>> %s\n\n" (date) >>$LOGFILE

python train.py \
    $TRAIN_DATA \
    $DEV_DATA \
    0 1 \
    $LM_NAME \
    --separator " " \
    --device $device \
    --save_path models/$RUN_NAME \
    --fine_tune \
    --run-name $RUN_NAME \
    --logfile $LOGFILE \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    (test $CRF -eq 1; and echo "--crf"; or echo "--no-crf") \
    $argv
