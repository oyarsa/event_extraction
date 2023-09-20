#!/usr/bin/env fish

set label_dist (dirname (status --current-filename))/label_dist.py
for f in $argv
    echo $f
    $label_dist $f
    echo
end
