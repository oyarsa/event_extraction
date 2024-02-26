#!/usr/bin/env fish

# Convert classifier model output containing the gold label to the agreement
# format for human annotation.
# This should be used as the "base" file when comparing other annotators.

if test (count $argv) -ne 2
    echo "Usage: $(status filename) <classifier_output> <human_output>"
    exit 1
end

set classifier_output $argv[1]
set human_output $argv[2]

jq 'map({
	input: .passage,
	valid: (.gold == 1),
	gold: .annotation
})' $classifier_output >$human_output
