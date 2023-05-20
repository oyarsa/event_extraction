#!/usr/bin/env fish

argparse 'env=' 'key=' 'data=' 'ts=' -- $argv
or return

set -q _flag_env; or set _flag_env dev
set -q _flag_key; or set _flag_key kcl
set -q _flag_ts; or set _flag_ts (date -u +%Y-%m-%dT%H.%M.%SZ)
if set -q _flag_data
    set input $_flag_data
else
    set input (realpath ./data/paraphrase/$input_file)
end

if test $_flag_env = full
    set input_file "paraphrase_dev_full.json"
else if test $_flag_env = exp
    set input_file "paraphrase_dev_100.json"
else if test $_flag_env = dev
    set input_file "paraphrase_dev_10.json"
else if test $_flag_env = debug
    set input_file "paraphrase_dev_2.json"
else
    echo "Invalid env"
    exit 1
end

cd (dirname (dirname (realpath (status -f))))

set output_dir ./output/paraphrase/{$_flag_env}/{$_flag_ts}
set output_file $output_dir/data/(basename $input)
mkdir -p (dirname $output_file)

echo Input: $input
echo Output dir: $output_dir
echo Output file: $output_file

.venv/bin/python paraphrase.py \
    keys.json $_flag_key \
    --input $input \
    --output $output_file \
    --metrics-path $output_dir/metrics.json \
    --args-path $output_dir/args.json \
    --log-file $output_dir/log.jsonl
