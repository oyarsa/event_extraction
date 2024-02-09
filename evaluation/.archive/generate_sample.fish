#!/usr/bin/env fish

function sample --argument-names file n
    jq -c '.[]' $file | shuf | jq -s | jq ".[:$n]"
end

sample $argv
