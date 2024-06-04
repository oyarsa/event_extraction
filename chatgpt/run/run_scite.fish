#!/usr/bin/env fish

if test (count $argv) -ne 3
    echo "Usage: $(basename (status -f)) <model> <input> <output>"
    exit 1
end

set model $argv[1]
set input $argv[2]
set out $argv[3]

python extraction.py ~/.config/event_extraction/gpt_config.json openai \
    --model $model \
    --input $input \
    --log-file $out/log.jsonl \
    --output $out/output.json \
    --metrics-path $out/metrics.json \
    --args-path $out/args.json \
    --sys-prompt 3 \
    --prompt 5 \
    --mode lines_no_relation \
    --examples data/extraction/scite/lines/examples.json

set -l root (git rev-parse --show-toplevel 2>/dev/null)
$root/chatgpt/scripts/convert_lines_to_tags.py $out/output.json $out/output.tags.json
$root/self_critique/scripts/eval_std.py $out/output.tags.json
