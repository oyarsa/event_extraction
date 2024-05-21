#!/usr/bin/env fish

if test -z "$argv[1]"
    echo "Usage: $(status -f) <input-file>"
    exit 1
end

set file $argv[1]
set output output/extraction/dev/lines/fincausal

python extraction.py ~/.config/event_extraction/gpt_config.json openai \
    --model gpt-3.5-turbo-0125 \
    --input $file \
    --output $output/output.json \
    --log-file $output/log.jsonl \
    --metrics-path $output/metrics.json \
    --args-path $output/args.json \
    --sys-prompt 3 \
    --prompt 5 \
    --mode lines_no_relation \
    --examples data/extraction/fincausal/lines/examples.json
