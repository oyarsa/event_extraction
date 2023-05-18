#!/usr/bin/env fish

cd (dirname (dirname (realpath (status -f))))

for f in $argv[1]/*
    set path (realpath $f)
    ./run/run_contradiction_structured.fish --data $path --env full
    echo
end

