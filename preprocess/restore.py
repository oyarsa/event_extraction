from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from colorama import Fore, Style


@dataclass
class Entry:
    token: str
    gold: str


def parse_ace(path: Path) -> list[list[Entry]]:
    sentences: list[list[Entry]] = []
    sentence: list[Entry] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                sentences.append(sentence)
                sentence = []
            else:
                token, gold = line.split()
                sentence.append(Entry(token, gold))

    return sentences


def reconstruct_sentence(sentence: list[Entry]) -> str:
    sio = StringIO()

    for entry in sentence:
        label = entry.gold
        if label == "O":
            sio.write(Style.RESET_ALL)
            sio.write(entry.token + " ")
            continue

        _, kind = label.split("-")
        in_tag = False
        if not in_tag:
            in_tag = True

        colour = Fore.RED if kind == "Cause" else Fore.GREEN
        sio.write(colour)

        sio.write(entry.token + " ")

    return sio.getvalue()


def analyse_ace_(dataset: list[list[Entry]]) -> list[str]:
    analyses: list[str] = [reconstruct_sentence(s) for s in dataset]
    return analyses


def analyse_ace(path: Path) -> list[str]:
    dataset = parse_ace(path)
    analyses = analyse_ace_(dataset)
    return analyses


analysis_funcs = {
    "ace": analyse_ace,
}


def main() -> None:
    supported_models = list(analysis_funcs)

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "model", help=f"Model to run error analysis. One of {supported_models}."
    )
    argparser.add_argument("path", help="Path to predictions file or folder.")
    argparser.add_argument("search")

    args = argparser.parse_args()

    path = Path(args.path)
    assert path.exists(), f"Path doesn't exist: {path}"

    model = args.model.strip().lower()
    if model not in analysis_funcs:
        raise ValueError(f"Invalid model: {model}. Choose one of {supported_models}.")

    analysis_func = analysis_funcs[model]
    analyses = analysis_func(path)

    for an in analyses:
        if args.search in an:
            print(an)
            print()


if __name__ == "__main__":
    main()
