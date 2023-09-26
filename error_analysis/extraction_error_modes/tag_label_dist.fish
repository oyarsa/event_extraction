#!/usr/bin/env fish
set tags (jq 'map(.tag)|sort[]' -r output/weak-test/best-weak-original/test_results.json | uniq)

for tag in $tags
    set data (jq --arg tag $tag 'map(select(.tag == $tag))' output/weak-test/best-weak-original/test_results.json)
    echo '>' $tag
    echo PRED
    ../../self_critique/scripts/label_dist.py --short pred <(echo $data | psub)
    echo GOLD
    ../../self_critique/scripts/label_dist.py --short gold <(echo $data | psub)
    echo
end
