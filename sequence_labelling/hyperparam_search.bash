. config/bert_nocrf.bash

models="bert-base-cased roberta-base"
batch_sizes="32 128"
learning_rates="5e-5 3e-5"
crf=$([ "$CRF" -eq 1 ] && echo "crf" || echo "no-crf")

for model in $models; do
    export LM_NAME="$model"

    for bs in $batch_sizes; do
        for lr in $learning_rates; do
            model_type=$(awk -F- '{print $1}' <<< "$model")
            run_name="${model_type}_${crf}_${EPOCHS}epoch_bs-${bs}_lr-${lr}"
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