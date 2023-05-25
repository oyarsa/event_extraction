#!/usr/bin/env fish

set label_dist (dirname (status --current-filename))/label_dist.fish
for f in $argv
    $label_dist $f
    echo
end
