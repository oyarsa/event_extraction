#!/usr/bin/env fish
jq 'map({input: .context, gold: .answers, output: .text_chosen})'
