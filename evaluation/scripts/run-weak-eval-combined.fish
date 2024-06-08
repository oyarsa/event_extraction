#!/usr/bin/env fish

if test (count $argv) -ne 2; or contains -- $argv -h --help
    echo "Usage: $(basename (status -f)) <ratio> <filter>"
    echo
    echo "Arguments:"
    echo "  ratio: number between [0, 1]"
    echo "  filter: one of 'none', 'average'"
    exit 1
end

set ratio $argv[1]
set filter $argv[2]

set root (git rev-parse --show-toplevel 2>/dev/null)
set eval $root/evaluation

# Generate split files with evaluation/scripts/split_file_ratio.py
set first $eval/data/weak/training_0-$ratio.json
set second $eval/data/weak/training_$ratio-1.json
if not test -f $first || not test -f $second
    echo "Input ratio files don't exist. Generate them with split_file_ratio.py first."
    exit 1
end

set run f-$filter-r-$ratio

set filtered $eval/data/weak/filtered-$run.json
set filtered_combo $eval/data/weak/filtered-combo-$run.json

set output_dir output/classifier
set output_name_first fcr-deberta-v3-xsmall-weak-$run-first
set output_name_second fcr-deberta-v3-xsmall-weak-$run-second

if test "$DEBUG" = 1
    set extra --max_samples 30 --num_epochs 2
else if test "$DEBUG" = 2
    set extra --num_epochs 1
end

python $eval/src/evaluation/classifier.py \
    --config $eval/data/weak/fcr-valid-deberta-v3-xsmall-weak-first.yaml \
    --train_data_path $first \
    --infer_data_path $second \
    --output_path $output_dir \
    --output_name $output_name_first \
    $extra

python scripts/confidence_filter.py \
    $output_dir/$output_name_first/inference_results.json \
    $filtered \
    --filter $filter

echo ">>>>> Filtered data <<<<<"
echo first (jq length $first)
echo second (jq length $second)
echo filtered (jq length $filtered)
jq -s add $first $filtered >$filtered_combo
echo filtered-combo (jq length $filtered_combo)
echo

python $eval/src/evaluation/classifier.py \
    --config $eval/data/weak/fcr-valid-deberta-v3-xsmall-weak-second.yaml \
    --model_name microsoft/deberta-v3-xsmall \
    --train_data_path $filtered_combo \
    --output_path $output_dir \
    --output_name $output_name_second \
    $extra

command rm -rf $output_dir/$output_name_first
