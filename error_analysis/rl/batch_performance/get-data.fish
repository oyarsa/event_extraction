#!/usr/bin/env fish

if test (count $argv) -lt 3
    echo "Usage: $(status -f) <mode> <file> <n>"
    exit 1
end

set mode (string lower (string trim $argv[1]))
set n $argv[2]
set dir $argv[3]

set cmd 'cd $SELF; .venv/bin/python'
if test $mode = tags
    set -a cmd $dir/count_tags.py $dir
    set label 'no. tags'
else if test $mode = ratios
    set -a cmd 'scripts/get_ratios.py' $dir
    set label 'entailment ratio'
else
    echo "Unknown mode: $mode"
    exit 1
end

python plot.py (ssh vm-free $cmd | psub) $n $label
