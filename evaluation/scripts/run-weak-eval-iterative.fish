#!/usr/bin/env fish

if test (count $argv) -lt 3; or contains -- $argv -h --help
    echo "Usage: $(basename (status -f)) <name> <filter> <files, ...>"
    echo
    echo "Arguments:"
    echo "  name: name of the experiment"
    echo "  filter: one of 'none', 'average'"
    echo "  files: training files to be used, in order, accumulative"
    exit 1
end

set name $argv[1]
set filter $argv[2]
set files $argv[3..]

set root (git rev-parse --show-toplevel 2>/dev/null)
set eval $root/evaluation
set output_dir output/classifier/$name/$filter

if test "$DEBUG" = 1
    set extra --max_samples 30 --num_epochs 2
else if test "$DEBUG" = 2
    set extra --num_epochs 1
end

set prev $files[1]

for current in $files[2..]
    set_color -b red black
    echo ">>> Train: $prev"
    echo ">>> Infer: $current"
    set_color normal
    printf "\n\n"

    set -l current_name (basename $prev .json)--(basename $current .json)

    python $eval/src/evaluation/classifier.py \
        --config $eval/data/weak/fcr-valid-deberta-v3-xsmall-weak-first.yaml \
        --train_data_path $prev \
        --infer_data_path $current \
        --output_path $output_dir \
        --output_name $current_name \
        $extra

    set -l filtered $current_name.filtered.json
    python $eval/scripts/confidence_filter.py \
        $output_dir/$current_name/inference_results.json \
        $filtered \
        --filter $filter

    set -l combined $current_name.json
    jq -s add $prev $filtered |
        jq -c '.[]' | shuf | jq -s '.' >$combined

    set_color -b blue black
    echo ">>> DATA SUMMARY:"
    echo "prev " (jq length $prev)
    echo "current " (jq length $current)
    echo "filtered " (jq length $filtered)
    echo "combined " (jq length $combined)
    set_color normal
    printf "\n\n"

    set prev $combined
end
