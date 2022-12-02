. config/bio.bash

models="bert-base-cased roberta-base"
batch_sizes="32 64 128"
learning_rates="3e-4 5e-5 3e-5"

for model in $models; do
    export LM_NAME="$model"

    for bs in $batch_sizes; do
        for lr in $learning_rates; do
            model_type=$(awk -F- '{print $1}' <<< "$model")
            run_name="$model_type-crf-4epoch-bs-${bs}_lr-${lr}"
            echo ">>>> MODEL: $model BATCH SIZE: $bs LEARNING RATE: $lr NAME $run_name"

            bash train.bash \
                --epochs 4 \
                --batch_size "$bs" \
                --lr "$lr" \
                --run-name "$run_name"

            printf "\n\n"
        done
    done
done