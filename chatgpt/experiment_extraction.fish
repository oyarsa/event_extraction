#!/usr/bin/env fish

argparse 'e/env=' -- $argv
or return

if set -q _flag_env
    set environment $_flag_env
else
    set environment dev
end

set num_prompts 2
set prompts (seq 0 (math $num_prompts - 1))

set modes tags lines

if [ $environment = dev ]
    set input_file "extraction_dev_2.json"
    set examples_file "extraction_examples_3.json"
else
    set input_file "extraction_dev_full.json"
    set examples_file "extraction_examples.json"
end

mkdir -p output
for prompt in $prompts
    for mode in $modes
        set input ./data/$mode/$input_file
        set examples ./data/$mode/$examples_file
        set output ./output/{$mode}_{$environment}_prompt{$prompt}_output.json
        set metrics ./output/{$mode}_{$environment}_prompt{$prompt}_metrics.json

        echo $prompt
        echo "input $input"
        echo "examples $examples"
        echo "output $output"
        echo python extraction.py \
            --prompt $prompt \
            --input $input \
            --output $output \
            --metrics-path $metrics \
            --examples $examples \
            --mode $mode
        printf "\n\n"
    end
end
