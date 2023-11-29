#!/usr/bin/env fish

python (git rev-parse --show-toplevel)/self_critique/scripts/eval.py \
    (fd -ejson . --exclude '*metrics.json') --bleurt --bertscore
