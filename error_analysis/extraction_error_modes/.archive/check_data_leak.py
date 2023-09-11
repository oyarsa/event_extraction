import json
import sys
from pathlib import Path

import typer


def main(file1: Path, file2: Path) -> None:  # sourcery skip: extract-method
    data1 = json.loads(file1.read_text())
    data2 = json.loads(file2.read_text())

    for d1 in data1:
        for d2 in data2:
            if d1["passage"] == d2["input"] and d1["annotation"] == d2["gold"]:
                print(d1["passage"])
                print(d1["annotation"])
                print(d2["input"])
                print(d2["gold"])
                print()
                sys.exit(1)

    print("No leak found")


if __name__ == "__main__":
    typer.run(main)
