#!/usr/bin/env nu

def main [log_file: string, data_file: string] {
	let start = (open $log_file | lines | length)

	open $data_file | {
		version: ($in | get version)
		data: ($in | get data | skip $start)
	} | to json
}


