#!/usr/bin/env fish

if test (count $argv) -lt 2
    echo 'usage: ./skip_completed.fish LOG_FILE DATA_FILE'
    exit 1
end

set log_file $argv[1]
set data_file $argv[2]

jq --argjson start (wc -l < $log_file) \
    '{version, data: .data[$start:]}' \
    $data_file
