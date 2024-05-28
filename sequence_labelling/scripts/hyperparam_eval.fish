#!/usr/bin/env fish
source config/bert_nocrf.fish

set models bert-base-cased roberta-base
set batch_sizes 32 128
set learning_rates 5e-5 3e-5
set crf (test $CRF -eq 1; and echo "crf" or echo "no-crf")

for model in $models
    for bs in $batch_sizes
        for lr in $learning_rates
            set model_type (echo $model | awk -F- '{print $1}')
            set run_name {$model_type}_{$crf}_{$EPOCHS}epoch_bs-{$bs}_lr-{$lr}

            if not test -f "models/$run_name"
                echo "ERROR: run $run_name doesn't exist"
                continue
            end
            echo ">>>> MODEL: $model BATCH SIZE: $bs LEARNING RATE: $lr NAME: $run_name"

            echo python predict.py \
                "$TEST_DATA" \
                "models/$run_name" \
                0 1 \
                "$model" \
                --output_path output \
                --output_name "$run_name" \
                --device "cuda:0"

            echo python fgcr_eval.py conll "output/$run_name"

            printf "\n\n"
        end
    end
end
