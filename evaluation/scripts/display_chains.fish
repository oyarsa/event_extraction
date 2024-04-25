#!/usr/bin/env fish

set -l script_name (basename (status -f))
if not set -q argv[1]; or contains -- $argv[1] -h --help
    echo "Usage: $script_name <chain-file>"
    echo
    echo "Pretty-prints the CoT chains in the file."
    exit (set -q argv[1])
end

jq -r 'map("\(.input)\n\nanswer: \(.answer)\n\nscore: \(.score)\n\ngenerated chain:\n\(.chain)") | join("\n\([range(50) | "-"] | join(""))\n\n")' \
    $argv[1]
