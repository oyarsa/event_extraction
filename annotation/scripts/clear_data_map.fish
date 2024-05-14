#!/usr/bin/env fish

if test -z "$argv[1]"
    echo "Usage: $(basename (status -f)) <split_to_user.json>"
    exit 1
end

set file $argv[1]
# Back up the original file
cp $file (basename $file .json).(date +"%Y%m%d%H%M%S").json
jq 'to_entries | map({key, value: null}) | from_entries' $file | sponge $file
