#!/usr/bin/env python3

"""Calculate the percentage of non-exact match valid examples in a JSON file.

The input is a JSON file with a list of objects, each with the following shape:
- input (str): input context
- gold (str): gold annotation response (tagged)
- rl_extract_txt (str): extracted response from a model (tagged)
- reward_label (str): label of the example (VALID or INVALID)

If the data is not in this shape, you can use the --rename option to rename the keys.
Each argument is a pair `old:new`, where the `old` key is renamed to `new`. The
validation is done after renaming the keys.
"""

import argparse
import json
from typing import Any, TextIO

from beartype.door import is_bearable


def build_renames(rename: list[str]) -> dict[str, str]:
    renames: dict[str, str] = {}
    for item in rename:
        old, new = item.split(":")
        renames[old] = new
    return renames


def rename_keys(item: dict[str, Any], renames: dict[str, str]) -> dict[str, Any]:
    new_item = item.copy()
    for old, new in renames.items():
        if old in new_item:
            new_item[new] = new_item.pop(old)
    return new_item


def main(input: TextIO, rename: list[str]) -> None:
    data = json.load(input)
    if not is_bearable(data, list[dict[str, Any]]):
        raise ValueError("Invalid JSON format. Expected a list of objects.")

    renames = build_renames(rename)
    data = [rename_keys(item, renames) for item in data]

    data_keys = {"input", "gold", "rl_extract_txt", "reward_label"}
    for item in data:
        if missing := data_keys - item.keys():
            raise SystemExit(f"Invalid JSON format. Missing keys: {missing}.")

    non_em_valid = sum(
        item["gold"] != item["rl_extract_txt"] and item["reward_label"] == "VALID"
        for item in data
    )

    print(f"Valid non-EM: {non_em_valid}/{len(data)} ({non_em_valid / len(data):.2%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=argparse.FileType("r"), help="Input JSON file")
    parser.add_argument(
        "--rename",
        type=str,
        nargs="+",
        default=[],
        help="Rename keys in the JSON. Format: old:new",
    )
    args = parser.parse_args()
    main(args.input, args.rename)
