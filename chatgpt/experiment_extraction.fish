#!/usr/bin/env fish

argparse 'env=' -- $argv
or return

if set -q _flag_env
    set environment $_flag_env
else
    set environment dev
end

set timestamp (date -u +%Y-%m-%dT%H.%M.%SZ)

set num_prompts 2
set prompts (seq 0 (math $num_prompts - 1))
set modes tags lines

if [ $environment = exp ]
    set input_file "extraction_dev_full.json"
    set examples_file "extraction_examples.json"
else
    set input_file "extraction_dev_2.json"
    set examples_file "extraction_examples_3.json"
end

for prompt in $prompts
    for mode in $modes
        set input ./data/$mode/$input_file
        set examples ./data/$mode/$examples_file

        set description {$timestamp}_{$mode}_{$environment}_prompt{$prompt}
        set output_folder ./output/$description
        mkdir -p $output_folder

        echo $description

        echo python extraction.py \
            --prompt $prompt \
            --input $input \
            --output $output_folder/output.json \
            --metrics-path $output_folder/metrics.json \
            --args-path $output_folder/args.json \
            --log-file $output_folder/log.jsonl \
            --examples $examples \
            --mode $mode

        string repeat -n 80 '*'
        echo
    end
end
