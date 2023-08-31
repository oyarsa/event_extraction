import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from readchar import readkey


def show(entry: dict[str, Any]) -> str:
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
                "  " + repr(entry["gold_cause"]),
                "pred:",
                "  " + repr(entry["pred_cause"]),
                f"excess count: {entry['cause_excess_count']}",
            )
        )
    else:
        out.append("CAUSE: same")

    out.append("\n")

    if entry["gold_effect"] != entry["pred_effect"]:
        out.extend(
            (
                "EFFECT",
                "gold:",
                "  " + repr(entry["gold_effect"]),
                "pred:",
                "  " + repr(entry["pred_effect"]),
                f"excess count: {entry['effect_excess_count']}",
            )
        )
    else:
        out.append("EFFECT: same")

    out.append("\n")

    return "\n".join(out)


def label_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    print(show(entry))

    while True:
        print("Valid extraction? y/n/q: ", end="")
        sys.stdout.flush()

        answer = readkey().lower()
        if answer == "q":
            return None
        if answer not in ["y", "n"]:
            print("Invalid answer")
            continue

        valid = answer == "y"
        print("valid" if valid else "invalid")
        return {**entry, "valid": valid}


def label(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labelled_data: list[dict[str, Any]] = []
    for i, entry in enumerate(data):
        if "valid" in entry:
            labelled_data.append(entry)
            continue

        print(f"ENTRY {i+1}/{len(data)}")

        new_data = label_entry(entry)
        if new_data is None:
            break
        labelled_data.append(new_data)

        print("\n###########\n")
    return labelled_data


def merge(
    previous: list[dict[str, Any]], new: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    previous_hashes = {entry["input"]: entry for entry in previous}
    merged: list[dict[str, Any]] = []

    for entry in new:
        if entry["input"] in previous_hashes:
            merged.append(previous_hashes[entry["input"]])
        else:
            merged.append(entry)

    return merged


def main(data_path: Path, output_path: Path, max_samples: Optional[int] = None) -> None:
    data = json.loads(data_path.read_text())[:max_samples]
    n_samples = len(data)
    print("Loaded", n_samples, "samples")
    if output_path.exists():
        previous = json.loads(output_path.read_text())
        data = merge(previous, data)
        n_skipped = sum("valid" in entry for entry in data)
        print(f"Skipping {n_skipped} already labelled samples")
    print()

    labelled_data = label(data)
    output_path.write_text(json.dumps(labelled_data, indent=2))


if __name__ == "__main__":
    typer.run(main)
