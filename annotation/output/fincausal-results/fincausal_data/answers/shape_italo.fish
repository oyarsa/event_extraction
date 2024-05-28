#!/usr/bin/env fish

# This script is used to generate the final "italo" set from Italo's annotations.
# They use a different format for the dev set, as they don't have a username field. I
# have no idea why that is the case. The train set is the same as the rest, so we can
# reshape the dev set to match the train set and then merge them together.

# Command to generate the final "italo" set:
# ./shape_italo.fish dev.json italo_dev.json italo_train.json > italo.json
# "italo" is then used alognside xinyu.json, 2f2b33b6.json etc.

if test (count $argv) -ne 3; or contains -- --help $argv
    echo "Usage: $(basename (status -f)) <dev_ann> <dev_out> <train>"
    echo "Output to stdout"
    exit 1
end

# Path to the dev set file with Italo's annotations
set dev_ann $argv[1]
# Path to output file from transforming the dev set
set dev_out $argv[2]
# Path to the existing Italo train set annotation
set train $argv[3]

# Transform Italo's dev annotations to the same format as other
# annotations
jq '{username: "italo", items: .}' $dev_ann >$dev_out

# Merge italo_dev.json and italo_train.json into a single file
# Information on the data source (i.e. which split is the item from) is lost,
# but will be restored with clean_data.py
jq -s '.[0] * {items: ([.[0].items, .[1].items] | flatten)}' $dev_out $train
