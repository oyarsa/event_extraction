import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Relation:
    type: str
    causes: list[str]
    effects: list[str]


@dataclass
class Example:
    id: int
    text: str
    relations: list[Relation]


@dataclass
class ProcessedRelation:
    relation: Relation
    sentence: str


@dataclass
class ProcessedExample:
    id: int
    text: int
    relations: list[ProcessedRelation]


def read_data(data_json: list[dict[str, str]]) -> list[Example]:
    processed = []

    for example in data_json:
        id = example["tid"]
        text = example["info"]
        relations = []

        for relation in example["labelData"]:
            type = relation["type"]
            causes = [text[start:end] for start, end in relation["reason"]]
            effects = [text[start:end] for start, end in relation["result"]]
            relations.append(
                Relation(
                    type=type,
                    causes=causes,
                    effects=effects,
                )
            )

        processed.append(
            Example(
                id=id,
                text=text,
                relations=relations,
            )
        )

    return processed


def process_relation(relation: Relation, text: str) -> str:
    start_index = int(float(1e9))
    end_index = -1

    for clause in relation.causes + relation.effects:
        start = text.index(clause)
        end = start + len(clause)

        start_index = min(start_index, start)
        end_index = max(end_index, end)

    assert start_index != int(float(1e9)) and end_index != -1
    return text[start_index:end_index].strip()


def process_example(example: Example) -> ProcessedExample:
    return ProcessedExample(
        id=example.id,
        text=example.text,
        relations=[
            ProcessedRelation(
                relation=relation,
                sentence=process_relation(relation, example.text),
            )
            for relation in example.relations
        ],
    )


def convert_file(infile: Path, outfile: Path) -> None:
    with open(infile) as f:
        data = read_data(json.load(f))

    processed = [process_example(example) for example in data]
    processed_json = [asdict(example) for example in processed]

    with open(outfile, "w") as f:
        json.dump(processed_json, f, indent=2)


def main() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--src",
        default="data/raw",
        help="Path to the folder containing the raw data",
    )
    argparser.add_argument(
        "--dst", default="data/reconstruct", help="Path to the output folder"
    )
    args = argparser.parse_args()

    raw_folder = Path(args.src)
    new_folder = Path(args.dst)

    splits = ["dev", "test", "train"]
    for split in splits:
        raw_path = raw_folder / f"event_dataset_{split}.json"
        new_path = new_folder / f"{split}.json"
        convert_file(raw_path, new_path)


if __name__ == "__main__":
    main()
