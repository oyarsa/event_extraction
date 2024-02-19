import json
from collections import Counter
from pathlib import Path

import typer

from metrics import StructureFormat, parse_instance


def get_relation(instance: str, mode: StructureFormat) -> str:
    relation = parse_instance(instance, mode)[1] or "<invalid>"
    return relation.lower().strip()


def main(path: Path, mode: StructureFormat = StructureFormat.LINES) -> None:
    valid_relations = {"enable", "cause", "prevent"}
    data: list[dict[str, str]] = json.loads(path.read_text())

    relations = [get_relation(d["pred"], mode) for d in data]

    relation_freq = Counter(relations)
    max_len = max(len(rel) for rel in relation_freq)
    max_len = min(max_len, 40)

    relations_set = set(relations)
    ordered_relations = sorted(valid_relations & relations_set) + sorted(
        relations_set - valid_relations, key=lambda x: -relation_freq[x]
    )

    print("idx", "relation".rjust(max_len + 1), "   #")
    for i, rel in enumerate(ordered_relations, start=1):
        freq = relation_freq[rel]
        rel = rel[:max_len].replace("\n", " ")  # noqa: PLW2901
        print(str(i).ljust(3), rel.rjust(max_len + 2), " ", freq)


if __name__ == "__main__":
    typer.run(main)
