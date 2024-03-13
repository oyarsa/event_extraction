from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


def tag_sort_key(tag: str) -> tuple[int, str]:
    """Sort tags by the following order: Cause, Relation, Effect.
    Tags with the same type are sorted alphabetically.

    Args:
        tag (str): The tag to sort.

    Returns:
        tuple[int, str]: A pair of tag-based index and the tag itself.
    """
    for i, c in enumerate("CRE"):
        if tag.startswith("[" + c):
            return i, tag
    assert False, f"Unknown tag: {tag}"


_ANSWER_SEPATOR = " | "


def generate_answer_combined_tags(
    events: dict[str, list[str]],
    label_map: dict[str, str],
    relation: str | None = None,
) -> str:
    out = []
    for ev_type, evs in events.items():
        event = f"[{label_map[ev_type]}] " + _ANSWER_SEPATOR.join(evs)
        out.append(event.strip())
    if relation:
        out.append(f"[Relation] {relation}")
    return " ".join(sorted(out, key=tag_sort_key))


def generate_answer_separate_tags(
    events: dict[str, list[str]], label_map: dict[str, str], relation: str | None = None
) -> str:
    out = []
    for ev_type, evs in events.items():
        for e in evs:
            out.append(f"[{label_map[ev_type]}] {e}")
    if relation:
        out.append(f"[Relation] {relation}")
    return " ".join(sorted(out, key=tag_sort_key))


def hash_instance(d: dict[str, str]) -> str:
    return hashlib.sha1(str(d).encode("utf-8")).hexdigest()[:8]


def convert_file_qa(
    infile: Path,
    outfile: Path,
    *,
    convert_instance: Callable[[dict[str, Any]], list[dict[str, str]]],
) -> None:
    with infile.open() as f:
        dataset = json.load(f)

    nested_instances = [convert_instance(instance) for instance in dataset]
    instances = deduplicate(item for sublist in nested_instances for item in sublist)
    transformed = {"version": "v1.0", "data": instances}

    outfile.parent.mkdir(exist_ok=True, parents=True)
    with outfile.open("w") as f:
        json.dump(transformed, f)


def deduplicate(list: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Remove duplicate instances from the list based on `id`."""
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []

    for instance in list:
        if instance["id"] in seen:
            continue

        seen.add(instance["id"])
        deduped.append(instance)

    return deduped


def extract_relation_span(causes: list[str], effects: list[str], text: str) -> str:
    start_index = int(1e9)
    end_index = -1

    for clause in causes + effects:
        start = text.index(clause)
        end = start + len(clause)

        start_index = min(start_index, start)
        end_index = max(end_index, end)

    assert start_index != int(1e9) and end_index != -1
    return text[start_index:end_index].strip()
