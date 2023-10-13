#!/usr/bin/env fish

set -q argv[1]; and set input $argv[1]; or set input -
cat $input | jq 'map({input: .context, gold: .answers, output: .text_chosen})'
