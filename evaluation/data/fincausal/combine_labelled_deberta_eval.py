#!/usr/bin/env python3
"""Combine human-labelled with model-evaluated data.

Both human and model-evaluated data are JSON lists of objects.
Human-labelled objects have this shape:
- input: str
- output: str
- gold: str
- valid: bool
- tag: str

Model-evaluated objects have this shape:
- input: str
- rl_extract_txt: str
- gold: str
- reward_label: str
"""

import argparse
import json
from pathlib import Path


def main(labelled_file: Path, eval_file: Path) -> None:
    labelled = json.loads(labelled_file.read_text())
    evaluated = json.loads(eval_file.read_text())

    keys = ["passage", "output", "annotation"]

    labelled_reshaped = sorted(
        (
            {
                "passage": ll["input"],
                "output": ll["output"],
                "annotation": ll["gold"],
                "gold": ll["valid"],
                "tag": ll["tag"],
            }
            for ll in labelled
        ),
        key=lambda x: "".join(x[k] for k in keys),
    )

    eval_reshaped = sorted(
        (
            {
                # FIX: Model evaluation leaves the whole prompt in the input. It shouldn't.
                "passage": ev["input"].split("\n")[0],
                "output": ev["rl_extract_txt"],
                "annotation": ev["gold"],
                "pred": ev["reward_label"] == "VALID",
            }
            for ev in evaluated
        ),
        key=lambda x: "".join(x[k] for k in keys),
    )

    out = [
        ll | {"pred": ev["pred"]}
        for ll, ev in zip(labelled_reshaped, eval_reshaped)
        if all(ll[k] == ev[k] for k in keys)
    ]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="\n".join(__doc__.splitlines()[1:]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("labelled_file", type=Path, help="Data with human labels")
    parser.add_argument("eval_file", type=Path, help="Data with model evaluations")
    args = parser.parse_args()
    main(args.labelled_file, args.eval_file)
