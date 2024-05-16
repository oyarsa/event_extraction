#!/usr/bin/env fish

set overlap 200
set num_splits 3

set root (git rev-parse --show-toplevel)
set files
for name in flan-t5-large_test flan-t5-large_train gpt4_test gpt35_test
    set -a files $root/data/annotation/inputs/$name.json
end

python scripts/filter_data.py $files data/tmp
python scripts/split_files.py data/tmp/to_annotate.json $num_splits data/tmp/split \
    --overlap $overlap
