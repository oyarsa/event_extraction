. config/bert_nocrf.bash

models="bert-base-cased roberta-base"
batch_sizes="32 128"
learning_rates="5e-5 3e-5"
crf=$([ "$CRF" -eq 1 ] && echo "crf" || echo "no-crf")

for model in $models; do
    for bs in $batch_sizes; do
        for lr in $learning_rates; do
            model_type=$(awk -F- '{print $1}' <<< "$model")
            run_name="${model_type}_${crf}_${EPOCHS}epoch_bs-${bs}_lr-${lr}"

            if [ ! -f "models/$run_name" ]; then
                echo "ERROR: run $run_name doesn't exist"
                continue
            fi
            echo ">>>> MODEL: $model BATCH SIZE: $bs LEARNING RATE: $lr NAME: $run_name"

            python predict.py \
                "$TEST_DATA" \
                "models/$run_name" \
                0 1 \
                "$model" \
                --output_path "output" \
                --output_name "$run_name" \
				--device "cuda:0"

            python fgcr_eval.py conll "output/$run_name"

            printf "\n\n"
        done
    done
done
