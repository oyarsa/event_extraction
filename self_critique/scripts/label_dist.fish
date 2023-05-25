#!/usr/bin/env fish

set file $argv[1]
echo $file
jq 'map(.entailment_label) | sort | .[]' $file | uniq -c
