#!/usr/bin/env fish

# Script to run the whole data processing pipeline for this annotation run

set files dev.json italo_train.json 2f2b33b6.json 3cc247fd.json xinyu.json

for file in $files
    if not test -e $file
        echo "Required file '$file' does not exist."
        exit 1
    end
end

./shape_italo.fish ./dev.json ./italo_dev.json ./italo_train.json >./italo.json

python clean_data.py --train train.json --dev dev.json \
    --ann 2f2b33b6.json 3cc247fd.json xinyu.json italo.json

python calc_kappa.py result.json -o kappa_order.json
python add_answers.py result.json kappa_order.json ann_answers.json

echo
echo "Output file is 'ann_answers.json'."
