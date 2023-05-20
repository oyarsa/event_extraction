#!/usr/bin/env fish

if test (count $argv) -lt 2
    echo "Usage: run_all.fish <script> <data>"
    exit 1
end

set -l script $argv[1]
set -l data $argv[2]
set -l timestamp (date -u +%Y-%m-%dT%H.%M.%SZ)

echo Script: $script
echo Data: $data
echo

cd (dirname (dirname (realpath (status -f))))

for f in $argv[2]/*
    set path (realpath $f)
    fish $script --data $path --env debug --ts $timestamp
    echo
end
