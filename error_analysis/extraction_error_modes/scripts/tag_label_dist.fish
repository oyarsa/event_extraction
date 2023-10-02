#!/usr/bin/env fish

set data_path $argv[1]
set tags (jq 'map(.tag)|sort[]' -r $data_path | uniq)

for tag in $tags
    set data (jq --arg tag $tag 'map(select(.tag == $tag))' $data_path)
    echo '>' $tag
    echo PRED
    echo $data | ../../self_critique/scripts/label_dist.py pred --compact
    echo GOLD
    echo $data | ../../self_critique/scripts/label_dist.py gold --compact
    echo
end | sed -e 's/\b0\b/invalid/' -e 's/\b1\b/valid/'
