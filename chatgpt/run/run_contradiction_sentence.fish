#!/usr/bin/env fish

argparse 'env=' 'prompt=' 'key=' -- $argv
or return

set -q _flag_env; or set _flag_env dev
set -q _flag_prompt; or set _flag_prompt 0
set -q _flag_key; or set _flag_key kcl

if [ $_flag_env = exp ]
    set input_file "contradiction_dev_100.json"
else if [ $_flag_env = dev ]
    set input_file "contradiction_dev_10.json"
else if [ $_flag_env = debug ]
    set input_file "contradiction_dev_2.json"
else
    echo "Invalid env"
    exit 1
end

cd (dirname (dirname (realpath (status -f))))

set timestamp (date -u +%Y-%m-%dT%H.%M.%SZ)
set input ./data/contradiction-sentence/$input_file
set examples ./data/contradiction-sentence/$examples_file

set description {$timestamp}_contradiction-sentence_{$_flag_env}_prompt{$_flag_prompt}
set output_folder ./output/{$_flag_env}/$description
mkdir -p $output_folder

echo $description

.venv/bin/python contradiction_sentence.py \
    keys.json $_flag_key \
    --prompt $_flag_prompt \
    --input $input \
    --output $output_folder/output.json \
    --metrics-path $output_folder/metrics.json \
    --args-path $output_folder/args.json \
    --log-file $output_folder/log.jsonl
