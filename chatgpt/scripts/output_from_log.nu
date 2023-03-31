#!/usr/bin/env nu

def main [log_file: string] {
	open $log_file -r | from json -o | each { |r| {
		input: ($r.params.messages | last | get content) 
		output: ($r.response.choices.message.content | first)
	}} | to json
}

