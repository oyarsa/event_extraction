#!/usr/bin/env fish


if test (count $argv) -lt 1
    echo 'usage: ./output_from_log.fish LOG_FILE'
    exit 1
end

set log_file $argv[1]

jq -s 'map({input: .params.messages[-1].content, output: .response.choices[0].message.content})' \
    $log_file
