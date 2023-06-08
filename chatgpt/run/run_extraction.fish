#!/usr/bin/env fish

argparse 'env=' 'mode=' 'key=' 'data=' 'ts=' -- $argv
or return

set -q _flag_env; or set _flag_env dev
set -q _flag_mode; or set _flag_mode lines
set -q _flag_key; or set _flag_key kcl
set -q _flag_ts; or set _flag_ts (date -u +%Y-%m-%dT%H.%M.%SZ)

if ! [ $_flag_mode = tags ] && ! [ $_flag_mode = lines ]
    echo Invalid mode: $_flag_mode
    exit 1
end

if [ $_flag_env = full ]
    set input_file "extraction_dev_full.json"
    set examples_file "extraction_examples.json"
else if [ $_flag_env = exp ]
    set input_file "extraction_dev_100.json"
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

cd (dirname (dirname (realpath (status -f))))

if set -q _flag_data
    set input $_flag_data
else
    set input (realpath ./data/extraction/$_flag_mode/$input_file)
end

set examples ./data/extraction/$_flag_mode/$examples_file

set output_dir ./output/extraction/$_flag_env/$_flag_mode/$_flag_ts
mkdir -p $output_dir

echo Output dir: $output_dir

.venv/bin/python extraction.py \
    keys.json $_flag_key \
    --input $input \
    --output $output_dir/output.json \
    --metrics-path $output_dir/metrics.json \
    --args-path $output_dir/args.json \
    --log-file $output_dir/log.jsonl \
    --examples $examples \
    --mode $_flag_mode

echo "METRICS:"
cat $output_dir/metrics.json
