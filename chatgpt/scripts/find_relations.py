import json
import sys
from collections import Counter
from pathlib import Path


def get_relation(s: str) -> str:  # sourcery skip: use-next
    for line in s.splitlines():
        if line.lower().strip().startswith("relation:"):
            return line.split(":")[1].strip()
    return ""


def clean_relation(s: str) -> str:
    return s if s in {"cause", "enable", "prevent"} else "other"


def relation_frequency(data: list[dict[str, str]], field: str) -> None:
    relations = (clean_relation(get_relation(d[field])) for d in data)
    counter = Counter(relations)
    print(field)
    print(json.dumps(counter, indent=2))


def main() -> None:
    data = json.loads(Path(sys.argv[1]).read_text())
    relation_frequency(data, "answer")
    relation_frequency(data, "pred")


if __name__ == "__main__":
    main()
