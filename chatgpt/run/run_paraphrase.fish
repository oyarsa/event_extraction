#!/usr/bin/env fish

argparse 'env=' 'prompt=' 'key=' 'data=' -- $argv
or return

set -q _flag_env; or set _flag_env dev
set -q _flag_prompt; or set _flag_prompt 0
set -q _flag_key; or set _flag_key kcl

if [ $_flag_env = full ]
    set input_file "paraphrase_dev_full.json"
    set examples_file "paraphrase_examples.json"
else if [ $_flag_env = exp ]
    set input_file "paraphrase_dev_100.json"
    set examples_file "paraphrase_examples.json"
else if [ $_flag_env = dev ]
    set input_file "paraphrase_dev_10.json"
    set examples_file "paraphrase_examples_3.json"
else if [ $_flag_env = debug ]
    set input_file "paraphrase_dev_2.json"
    set examples_file "paraphrase_examples_3.json"
else
    echo "Invalid env"
    exit 1
end

cd (dirname (dirname (realpath (status -f))))

set timestamp (date -u +%Y-%m-%dT%H.%M.%SZ)
if set -q _flag_data
    set input $_flag_data
    set _flag_env full/$_flag_env-(basename $_flag_data)
else
    set input (realpath ./data/paraphrase/$input_file)
end
set examples ./data/paraphrase/$examples_file

echo Input: $input

set description {$timestamp}_paraphrase_{$_flag_mode}_{$_flag_env}_prompt{$_flag_prompt}
set output_folder ./output/{$_flag_env}/$description
mkdir -p $output_folder

echo $description

.venv/bin/python paraphrase.py \
    keys.json $_flag_key \
    --prompt $_flag_prompt \
    --input $input \
    --output $output_folder/output.json \
    --metrics-path $output_folder/metrics.json \
    --args-path $output_folder/args.json \
    --log-file $output_folder/log.jsonl \
    --examples $examples \
    --mode $_flag_mode
