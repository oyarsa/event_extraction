#!/usr/bin/env fish
printf "%-45s %-8s %-7s %-10s %-10s %-10s\n" model epochs acc f1 precision recall
for f in (fd training_metrics.jsonl $argv[1])
    printf "%-45s %-8d %-7.5f %-10.5f %-10.5f %-10.5f\n" \
        (basename (dirname $f)) \
        (wc -l < $f | string trim) \
        (jq -s 'max_by(.accuracy).accuracy' $f) \
        (jq -s 'max_by(.accuracy).f1' $f) \
        (jq -s 'max_by(.accuracy).precision' $f) \
        (jq -s 'max_by(.accuracy).recall' $f)
end | sort -k2
