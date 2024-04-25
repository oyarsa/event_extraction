#!/usr/bin/env fish

set -l help "\
Usage: $(basename (status -f)) <result-dir>

Prints accuracies of the results in the given directory.
- overall: accuracy of the full data
- EM-only: accuracy of only the exact match cases
- non-EM: accuracy of the non-exact match cases
"

if not set -q argv[1]; or contains -- $argv[1] -h --help
    printf $help
    exit (set -q argv[1])
end
set -l dir $argv[1]

set -l filters true '.tag == "em"' '.tag != "em"'
set -l labels overall EM-only non-EM

for split in eval test
    echo Split: $split
    set -l file $dir/{$split}_results.json

    for i in (seq (count $filters))
        set -l filter $filters[$i]
        set -l label $labels[$i]

        set -l matches (
			jq "map(select($filter)) | map(select(.pred == .gold)) | length" $file
        )
        set -l length (jq length $file)

        printf "  %s: %d / %d (%.2f%%)\n" $label $matches $length \
            (math "$matches / $length * 100")
    end

    echo
end
