import difflib
import json
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from readchar import readkey


def overlap(s1: str, s2: str) -> int:
    """
    Returns the length of the longest overlapping sequence of words between two strings.
    """
    words1 = s1.split()
    words2 = s2.split()
    matcher = difflib.SequenceMatcher(None, words1, words2)
    _, _, size = matcher.find_longest_match(0, len(words1), 0, len(words2))
    return size


def show(entry: dict[str, Any], use_excess: bool, use_overlap: bool) -> str:
    out: list[str] = [
        "PASSAGE",
        entry["input"],
        "\n",
    ]

    if entry["gold_cause"] != entry["pred_cause"]:
        out.extend(
            (
                "CAUSE",
                "gold:",
                f"  -> {entry['gold_cause']!r}",
                "pred:",
                f"  -> {entry['pred_cause']!r}",
            )
        )

        if use_excess:
            out.append(f"excess count: {entry['cause_excess_count']}")

        if use_overlap:
            is_substr_passage = entry["gold_cause"] in entry["input"]
            out.extend(
                (
                    f"substr passage: {is_substr_passage}",
                    f"overlap: {overlap(entry['gold_cause'], entry['pred_cause'])}",
                )
            )
    else:
        out.extend(
            (
                "CAUSE",
                f"  {entry['gold_cause']!r}",
            )
        )

    out.append("\n")

    if entry["gold_effect"] != entry["pred_effect"]:
        out.extend(
            (
                "EFFECT",
                "gold:",
                f"  -> {entry['gold_effect']!r}",
                "pred:",
                f"  -> {entry['pred_effect']!r}",
            )
        )

        if use_excess:
            out.append(f"excess count: {entry['effect_excess_count']}")

        if use_overlap:
            is_substr_passage = entry["gold_effect"] in entry["input"]
            out.extend(
                (
                    f"substr passage: {is_substr_passage}",
                    f"overlap: {overlap(entry['gold_effect'], entry['pred_effect'])}",
                )
            )
    else:
        out.extend(
            (
                "EFFECT",
                f"  {entry['gold_effect']!r}",
            )
        )

    out.append("\n")

    return "\n".join(out)


def label_entry(
    entry: dict[str, Any], use_excess: bool, use_overlap: bool
) -> dict[str, Any] | None:
    "Gets user input for an entry label. Returns None if user quits."

    if use_overlap:
        overlap_cause = overlap(entry["gold_cause"], entry["pred_cause"])
        overlap_effect = overlap(entry["gold_effect"], entry["pred_effect"])

        if overlap_cause <= 1 or overlap_effect <= 1:
            print(
                f"Skipping due to low overlap (C:{overlap_cause} E:{overlap_effect}))"
            )
            return {**entry, "valid": False}

    print(show(entry, use_excess, use_overlap))

    while True:
        print("Valid extraction? y/n/q: ", end="", flush=True)

        answer = readkey().lower()
        if answer == "q":
            return None
        if answer not in ["y", "n"]:
            print("Invalid answer")
            continue

        valid = answer == "y"
        print("valid" if valid else "invalid")
        return {**entry, "valid": valid}


def label(
    data: list[dict[str, Any]], use_excess: bool, use_overlap: bool
) -> list[dict[str, Any]]:
    labelled_data: list[dict[str, Any]] = []
    for i, entry in enumerate(data):
        if "valid" in entry:
            labelled_data.append(entry)
            continue

        print(f"ENTRY {i+1}/{len(data)}")

        new_data = label_entry(entry, use_excess, use_overlap)
        if new_data is None:
            break

        labelled_data.append(new_data)

        print("\n###########\n")
    return labelled_data


def merge(
    previous: list[dict[str, Any]], new: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    previous_entries = {entry["input"]: entry for entry in previous}
    merged: list[dict[str, Any]] = []

    for entry in new:
        if entry["input"] in previous_entries:
            merged.append(previous_entries[entry["input"]])
        else:
            merged.append(entry)

    return merged


def main(
    data_path: Path,
    output_path: Annotated[Optional[Path], typer.Option("--output", "-o")] = None,
    max_samples: Annotated[Optional[int], typer.Option("--max-samples", "-n")] = None,
    use_excess: Annotated[bool, typer.Option("--use-excess", "-x")] = False,
    use_overlap: Annotated[bool, typer.Option("--use-overlap", "-O")] = False,
) -> None:
    data = json.loads(data_path.read_text())[:max_samples]
    print("Loaded", len(data), "samples")

    output_path = output_path or data_path.with_suffix(".labelled.json")
    if output_path.exists():
        previous = json.loads(output_path.read_text())
        data = merge(previous, data)

        n_skipped = sum("valid" in entry for entry in data)
        print(f"Skipping {n_skipped} already labelled samples")
    print()

    if labelled_data := label(data, use_excess, use_overlap):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(labelled_data, indent=2))


if __name__ == "__main__":
    typer.run(main)
