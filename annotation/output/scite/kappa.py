"""Calculate Cohen's kappa for a set of annotations. They must have the same examples.

The input files must be in the annotation tool format, which is a JSON file with the
following structure:

- username (str): user of the annotator
- items (list): list of items
  - id (str): identifier of the item
  - data (dict): data of the item
    - text (str): text to be annotated (plain text)
    - reference (str): reference answer (tag format)
    - model (str): model answer (tag format)
  - answer (str): answer of the annotator
"""

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from typing import Any, TextIO

from sklearn.metrics import cohen_kappa_score  # type: ignore


@dataclass(frozen=True)
class Data:
    text: str
    reference: str
    model: str


@dataclass(frozen=True)
class Item:
    id: str
    data: Data
    answer: str


@dataclass(frozen=True)
class Annotation:
    username: str
    items: list[Item]


def calc_kappa(data1: list[Item], data2: list[Item]) -> float:
    if not all(x.id == y.id for x, y in zip(data1, data2)):
        raise ValueError("Incompatible annotations. The items must be the same.")

    return cohen_kappa_score(
        [d.answer for d in data1],
        [d.answer for d in data2],
    )


def parse(data: dict[str, Any]) -> Annotation:
    return Annotation(
        username=data["username"],
        items=sorted(
            [
                Item(
                    id=item["id"],
                    data=Data(
                        text=item["data"]["text"],
                        reference=item["data"]["reference"],
                        model=item["data"]["model"],
                    ),
                    answer=item["answer"],
                )
                for item in data["items"]
            ],
            key=lambda x: x.id,
        ),
    )


def main(input_files: list[TextIO]) -> None:
    annotations = [parse(json.load(f)) for f in input_files]

    kappas = {
        (ann1.username, ann2.username): calc_kappa(ann1.items, ann2.items)
        for ann1, ann2 in combinations(annotations, 2)
    }

    for (u1, u2), kappa in kappas.items():
        print(f"{u1} vs {u2}: {kappa}")

    print("Average kappa:", sum(kappas.values()) / len(kappas))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        type=argparse.FileType("r"),
        help="Input JSON files (annotation tool format)",
    )
    args = parser.parse_args()
    main(args.input_files)
