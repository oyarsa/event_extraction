#!/usr/bin/env fish

argparse 'env=' 'mode=' 'prompt=' 'key=' -- $argv
or return

set -q _flag_env; or set _flag_env dev
set -q _flag_mode; or set _flag_mode lines
set -q _flag_prompt; or set _flag_prompt 0
set -q _flag_key; or set _flag_key personal

if [ $_flag_env = exp ]
    set input_file "extraction_dev_full.json"
    set examples_file "extraction_examples.json"
else if [ $_flag_env = dev ]
    set input_file "extraction_dev_10.json"
    set examples_file "extraction_examples_3.json"
else if [ $_flag_env = debug ]
    set input_file "extraction_dev_2.json"
    set examples_file "extraction_examples_3.json"
else
    echo "Invalid env"
    exit 1
end

set timestamp (date -u +%Y-%m-%dT%H.%M.%SZ)
set input ./data/$_flag_mode/$input_file
set examples ./data/$_flag_mode/$examples_file

set description {$timestamp}_{$_flag_mode}_{$_flag_env}_prompt{$_flag_prompt}
set output_folder ./output/$description
mkdir -p $output_folder

echo $description

.venv/bin/python extraction.py \
    keys.json $_flag_key \
    --prompt $_flag_prompt \
    --input $input \
    --output $output_folder/output.json \
    --metrics-path $output_folder/metrics.json \
    --args-path $output_folder/args.json \
    --log-file $output_folder/log.jsonl \
    --examples $examples \
    --mode $_flag_mode
