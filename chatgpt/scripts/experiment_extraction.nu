#!/usr/bin/env nu

def main [--env: string = dev, --key: string = personal] {
    cd ($env.FILE_PWD | path dirname)

    let num_prompts = 2
    let prompts = (seq 0 ($num_prompts - 1))
    let modes = [tags lines]

    for prompt in $prompts {
        for mode in $modes {
            fish run_extraction.fish --env $env --key $key --mode $mode --prompt $prompt
            print $"('*' * 80)\n"
        }
    }
}
