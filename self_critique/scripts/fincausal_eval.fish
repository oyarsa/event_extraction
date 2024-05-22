#!/usr/bin/env fish

if test (count $argv) -ne 1; or test $argv[1] = --help
    echo "Usage: $(basename (status -f)) <extraction_output.json>"
    exit 1
end

set file $argv[1]
set dir (dirname (realpath $file))
set scripts (git rev-parse --show-toplevel)/self_critique/scripts

python $scripts/convert_fincausal_output.py \
    (jq 'map({input: .context, output, gold: .answer})' $file | psub) \
    --ref $dir/ref.csv --pred $dir/pred.csv
python $scripts/fincausal_eval.py $dir/ref.csv $dir/pred.csv

command rm -f $dir/ref.csv $dir/pred.csv
