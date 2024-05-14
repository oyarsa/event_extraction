#!/usr/bin/env fish

if test (count $argv) -ne 2
    echo "Usage: $(basename (status -f)) <original-data> <prolific-ids>"
    exit 1
end

set original_data $argv[1]
set prolific_ids $argv[2]
set num_ids (count (cat $prolific_ids))

python scripts/filter_data.py $original_data data/source
python scripts/split_files.py data/source/to_annotate.json $num_ids
python scripts/pick_users.py data/inputs/split_to_user.json $prolific_ids
