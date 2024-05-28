#!/usr/bin/env fish

set -q LM_NAME; or set LM_NAME bert-base-cased
set -q BATCH_SIZE; or set BATCH_SIZE 32

set fish_script (basename (status -f))
function usage
    echo "Usage: TEST_DATA=? MODEL_PATH=? [LM_NAME] [LOG_FILE=?] $fish_script [-h|--help] [...ARGS]"
    printf "\t[ARGS] are passed to predict.py\n"
    exit 1
end

# Check for help flag and print usage
for arg in $argv
    if test "$arg" = -h -o "$arg" = --help
        usage
    end
end

if not set -q TEST_DATA; or not set -q MODEL_PATH
    usage
end

test (uname) = Darwin; and set device mps; or set device cuda

python predict.py \
    $TEST_DATA \
    $MODEL_PATH \
    0 1 \
    $LM_NAME \
    --batch_size $BATCH_SIZE \
    --output_path $MODEL_PATH/predict \
    --device $device \
    $argv
