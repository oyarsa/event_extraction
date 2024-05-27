#!/usr/bin/env fish

if test (count $argv) -ne 2
    echo "Usage: $(basename (status -f)) <data> <name>"
    exit 1
end

set data $argv[1]
set name $argv[2]

python self_critique/rl/run_reward.py \
    --model_path ../evaluation/output/classifier/fcr-deberta-best \
    --data_file $data \
    --run_name $name
