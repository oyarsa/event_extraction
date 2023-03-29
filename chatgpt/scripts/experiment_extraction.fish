#!/usr/bin/env fish

argparse 'env=' 'key=' -- $argv
or return

set -q _flag_env; or set _flag_env dev
set -q _flag_key; or set _flag_key personal

set num_prompts 1
set prompts (seq 0 (math $num_prompts - 1))
set modes tags lines

for prompt in $prompts
    for mode in $modes
        fish run_extraction.fish --env $_flag_env --prompt $prompt --mode $mode

        string repeat -n 80 '*'
        echo
    end
end
