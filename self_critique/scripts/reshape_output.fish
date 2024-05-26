#!/usr/bin/env fish


if test $argv[1] = --help
    echo "Usage: $(basename (status -f)) [extraction_output.json]"
    echo
    echo "If no file is provided, reads from stdin."
    echo "Conversion: .context -> .input, .output -> .output, .answer -> .gold"
    exit 0
end

jq 'map({input: .context, output: .output, gold: .answer})' $argv
