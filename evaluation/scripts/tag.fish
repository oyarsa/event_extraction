#!/usr/bin/env fish

function main -a rule hand
    set rule_tagged (string replace ".json" ".tagged.json" $rule)
    set hand_tagged (string replace ".json" ".tagged.json" $hand)

    jq 'map(.tag = if .valid then "em" else "nonsubstr" end)' $rule >$rule_tagged
    jq 'map(.tag = if .valid then "substr_valid" else "substr_invalid" end)' $hand >$hand_tagged
end

main $argv
