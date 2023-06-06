#!/usr/bin/env fish

if test (count $argv) -lt 2
    echo "Usage: $argv[0] <file> <n>"
    exit 1
end

set n $argv[1]
set file $argv[2]

set cmd 'cd $SELF; .venv/bin/python scripts/get_ratios.py' $file
python plot_ratios.py (ssh vm-free $cmd | psub) $n
