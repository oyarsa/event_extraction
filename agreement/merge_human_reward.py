"""Combine human annotation file and machine prediction file into a single file.

The human annotation file follows the `annotation` format. It's a JSON file with a list
of objects with the following keys:
- input (str): the input passage
- gold (str): the annotation extraction
- output (str): the model output
- tag (str, optional): instance tag, e.g. 'em'
- valid (bool): a boolean indicating whether the human annotated the output as valid

The machine prediction follows the one from `self_critique`'s `run_reward` and RL
outputs.
- input (str): the input passage. It can include the whole evaluator input after a
  newline, but only the first line is used.
- gold (str): the annotation extraction
- rl_extract_txt (str): the model output
- reward_label (str): the evaluator's output label, e.g. 'VALID'

The output file is a JSON file with a list of objects with the following keys:
- gold (int): 1 if the human annotated the output as valid, 0 otherwise
- pred (int): 1 if the evaluator output is 'VALID', 0 otherwise
- passage (str): the input passage
- output (str): the model output
- tag (str, optional): the instance tag
"""

import argparse
import json
import sys
from typing import Any, TextIO

human_file = sys.argv[1]
machine_file = sys.argv[2]


def hash_data(data: dict[str, Any], keys: list[str]) -> int:
    return hash("".join(data[k] for k in keys))


def main(human_file: TextIO, machine_file: TextIO, outfile: TextIO) -> None:
    human = json.load(human_file)
    machine = json.load(machine_file)

    machine_clean = [
        m | {"input": m["input"].splitlines()[0], "output": m["rl_extract_txt"]}
        for m in machine
    ]

    keys = ["input", "output", "gold"]
    machine_idx = {hash_data(m, keys): m for m in machine_clean}

    out = [
        {
            "gold": int(h["valid"]),
            "pred": int(m["reward_label"] == "VALID"),
            "passage": h["input"],
            "output": h["output"],
            "tag": h["tag"],
        }
        for h in human
        if (m := machine_idx.get(hash_data(h, keys)))
    ]
    json.dump(out, outfile, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("human_file", type=argparse.FileType("r"), help="Human file")
    parser.add_argument(
        "machine_file", type=argparse.FileType("r"), help="Machine file"
    )
    parser.add_argument("outfile", type=argparse.FileType("w"), help="Output file")
    args = parser.parse_args()
    main(args.human_file, args.machine_file, args.outfile)
