set -x TRAIN_DATA "../data/sequence_labelling/train.txt"
set -x DEV_DATA "../data/sequence_labelling/dev.txt"
set -x TEST_DATA "../data/sequence_labelling/test.txt"

set -x BATCH_SIZE 32
set -x LR 3e-5
set -x EPOCHS 4
set -x LM_NAME bert-base-cased
set -x CRF 1
set -x RUN_NAME bert_nocrf
