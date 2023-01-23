import collections
import json
from dataclasses import dataclass
from typing import TextIO

import click


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


def process_data(data_json: list[dict]) -> list[Example]:
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


def _render_example(ex: Example) -> str:
    out = [
        f"# ID: {ex.id}",
        ex.text,
        "",
        "# Relations",
        "",
    ]
    for i, rel in enumerate(ex.relations):
        out.extend(
            [
                f"## {i+1}: {rel.type}",
                "### Causes",
                "\n".join(f"{i+1}. {x}" for i, x in enumerate(rel.causes)),
                "",
                "### Effects",
                "\n".join(f"{i+1}. {x}" for i, x in enumerate(rel.effects)),
                "",
            ]
        )
    return "\n".join(out)


def render_example(ex: Example | list[Example]) -> str:
    if not isinstance(ex, list):
        ex = [ex]

    sep = "-" * 10
    return f"\n{sep}\n\n".join(_render_example(e) for e in ex)


@click.command
@click.argument("data_file", type=click.File("r"))
@click.option(
    "--limit",
    type=int,
    default=20,
    help="Limit the number of examples to print",
)
@click.option("--min-causes", type=int, default=1, help="Minimum number of causes")
@click.option("--min-effects", type=int, default=1, help="Minimum number of effects")
@click.option("--max-causes", type=int, default=3, help="Maximum number of causes")
@click.option("--max-effects", type=int, default=3, help="Maximum number of effects")
@click.option("--stats", is_flag=True, help="Print statistics")
def main(
    data_file: TextIO,
    limit: int,
    min_causes: int,
    min_effects: int,
    max_causes: int,
    max_effects: int,
    stats: bool,
) -> None:
    data_json = json.load(data_file)

    processed = process_data(data_json)

    if stats:
        print_stats(processed)
        return

    processed = [
        p
        for p in processed
        if any(
            min_causes <= len(r.causes) <= max_causes
            and min_effects <= len(r.effects) <= max_effects
            for r in p.relations
        )
    ]
    processed = processed[:limit]

    print(render_example(processed))


# Find number of relations by number of causes and effects
def calculate_stats(data: list[Example]) -> list[tuple[tuple[int, int], int]]:
    stats: dict[tuple[int, int], int] = collections.defaultdict(int)

    for ex in data:
        for rel in ex.relations:
            num_causes = len(rel.causes)
            num_effects = len(rel.effects)
            stats[(num_causes, num_effects)] += 1

    return sorted(stats.items(), key=lambda x: x[1], reverse=True)


def print_stats(data: list[Example]) -> None:
    num_examples = len(data)
    num_relations = sum(len(ex.relations) for ex in data)
    num_causes = sum(len(r.causes) for ex in data for r in ex.relations)
    num_effects = sum(len(r.effects) for ex in data for r in ex.relations)
    other_stats = calculate_stats(data)

    print("# Statistics")
    print(f"- Number of examples: {num_examples}")
    print(f"- Number of relations: {num_relations}")
    print(f"- Number of causes: {num_causes}")
    print(f"- Number of effects: {num_effects}")

    print("\n| num cases, num effects | count |")
    print("| --- | --- |")
    for (num_causes, num_effects), count in other_stats:
        print(f"| ({num_causes}, {num_effects}) | {count} |")


if __name__ == "__main__":
    main()
