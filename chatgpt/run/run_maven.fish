#!/usr/bin/env fish

if test (count $argv) -ne 4
    echo "Usage: $(basename (status -f)) <model> <input> <output> <cuda|cpu>"
    exit 1
end

set model $argv[1]
set input $argv[2]
set out $argv[3]
set device $argv[4]

if not test $device = cuda -o $device = cpu
    echo "Invalid device: $device"
    exit 1
end

python extraction.py ~/.config/event_extraction/gpt_config.json openai \
    --model $model \
    --input $input \
    --log-file $out/log.jsonl \
    --output $out/output.json \
    --metrics-path $out/metrics.json \
    --args-path $out/args.json \
    --sys-prompt 3 \
    --prompt 6 \
    --mode straight \
    --examples data/extraction/maven/lines/examples.json

set -l root (git rev-parse --show-toplevel 2>/dev/null)
jq 'map({input: .text, output: .pred, gold: .answer})' \
    $out/output.json >$out/output.reshaped.json

python $root/self_critique/self_critique/rl/run_reward.py \
    --model_path $root/evaluation/output/classifier/fcr-deberta-v3-xsmall-combined/ \
    --data_file $out/output.reshaped.json \
    --output_dir $root/self_critique/output/reward/(basename $out)-fcreval/ \
    --mode maven_s \
    --device $device
