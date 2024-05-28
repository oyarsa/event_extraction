"""Add answers to the input file by aggregating the answers from multiple annotators.

The input file should be a JSON file with a list of objects with the following keys:
TODO

The order file should be a JSON file with a list of usernames orderd by highest average
pairwise kappa score. This is used in case of ties, where it will use the answer from
the first username in the list that answered that question.
"""

import argparse
import json
from typing import Any, TextIO


def get_answer(item: dict[str, Any], users_ordered: list[str]) -> bool:
    user_to_answer = {
        a["name"]: a["answer"] == "valid"
        for a in item["answers"]
        if a["answer"] is not None
    }
    answers = list(user_to_answer.values())

    count_true = answers.count(True)
    count_false = answers.count(False)

    # If we have a clean majority, use that
    if count_true != count_false:
        return count_true > count_false

    # Tie breaker: find first username in the ordered list that answered the question
    for user in users_ordered:
        if answer := user_to_answer.get(user):
            return answer

    raise ValueError(f"None of the usernames answered the question: {item['id']}")


def ensure_users(data: list[dict[str, Any]], usernames_ordered: list[str]) -> None:
    """Ensure that all users who answered are in the order file.

    If they aren't, we can't break ties correctly.
    """
    answers = {a["name"] for d in data for a in d["answers"]}
    ordered = set(usernames_ordered)

    missing = answers - ordered
    if missing:
        raise ValueError(f"Users who answered but are not in the order file: {missing}")


def main(infile: TextIO, order_file: TextIO, outfile: TextIO) -> None:
    usernames_ordered = json.load(order_file)
    data = json.load(infile)
    ensure_users(data, usernames_ordered)

    for item in data:
        item["valid"] = get_answer(item, usernames_ordered)

    json.dump(data, outfile, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("infile", type=argparse.FileType("r"), help="The input file")
    parser.add_argument(
        "order_file", type=argparse.FileType("r"), help="The order file"
    )
    parser.add_argument("outfile", type=argparse.FileType("w"), help="The output file")
    args = parser.parse_args()
    main(args.infile, args.order_file, args.outfile)
