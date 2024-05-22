#!/usr/bin/env python
"""Scoring program for Fincausal 2020 Task 2."""

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass

import nltk
from sklearn import metrics


def build_token_index(text: str) -> tuple[list[str], dict[str, list[int]]]:
    """Build dictionary of text tokens with their respective positions in the text.

    E.g. "this is a basic example of a basic method" returns
     {'this': [0], 'is': [1], 'a': [2, 6], 'basic': [3, 7], 'example': [4], 'of': [5],
      'method': [8]}

    :param text: reference text to index
    :return: dict of text token, each token with their respective position(s) in the
        text
    """
    tokens = nltk.word_tokenize(text)
    token_index: dict[str, list[int]] = defaultdict(list)

    for position, token in enumerate(tokens):
        token_index[token].append(position)

    return tokens, token_index


_discarded = 0


def get_longest_tokens_sequence(
    text: str, token_index: dict[str, list[int]]
) -> list[int]:
    """Find longest sequence of tokens in the reference text matching the input text."""
    tokens = nltk.word_tokenize(text)
    # build list of possible position for each token
    positions: list[list[int]] = []

    for token in tokens:
        if pos := token_index.get(token):
            positions.append(pos)
            continue

        # Special case when '.' is not tokenized properly
        alt_token = token + "."
        if pos := token_index.get(alt_token):
            logging.debug(f'tokenize fix ".": {alt_token}')
            positions.append(pos)
            continue

        global _discarded  # noqa: PLW0603
        _discarded += 1

    # No matching ? stop here
    if not positions:
        return []

    # recursively process the list of token positions to return combinations of
    # consecutive tokens
    seqs = _get_sequences(*positions)
    # Note: several sequences can possibly be found in the reference text, when similar
    # text patterns are repeated
    # always return the longest
    return max(seqs, key=len)


def _get_sequences(
    *args, value: int | None = None, path: list[int] | None = None
) -> list[list[int]]:
    """Select sequences of tokens using their position relative to the reference text.

    A sequence is the list of successive indexes in the tokenized reference text.

    :param args: list of list of positions
    :param value: position of the previous token (i.e. next token position must be in
        range [value+1, value+3]
    :param path: debugging - current sequence
    :return:
    """
    # end of recursion
    if len(args) == 1:
        if value is not None:
            # return items matching constraint (i.e. within range with previous token)
            return [x for x in args[0] if x > value and (x < value + 3)]
        else:
            # Special case where text is restricted to a single token
            # return all positions on first call (i.e. value is None)
            return [args[0]]

    # iterate over current token possible positions and combine with other tokens from recursive call
    # result is a list of explored sequences (i.e. list of list of positions)
    result = []
    for x in args[0]:
        # <Debug> keep track of current explored sequence
        p = [x] if path is None else list([*path, x])
        # </Debug>

        if value is None or (x > value and (x < value + 3)):
            seqs = _get_sequences(*args[1:], value=x, path=p)
            # when recursion returns empty list and current position match constraint (either only value
            # or value within range) add current position as a single result
            if len(seqs) == 0 and (value is None or (x > value and (x < value + 3))):
                result.append([x])
            else:
                # otherwise combine current position with recursion results (whether returned sequences are list
                # or single number) and add to the list of results for this token position
                for s in seqs:
                    res = [x, *s] if isinstance(s, list) else [x, s]
                    result.append(res)
    return result


def encode_causal_tokens(text: str, cause: str, effect: str) -> list[str]:
    """Encode text, cause and effect tokens as their respective classes ('-', 'C', 'E').

    :param text: reference text
    :param cause: causal substring in reference text
    :param effect: effect substring in reference text
    :return: text string converted as a list of tuple(token, label)
    """
    # Get reference text tokens and token index
    words, wi = build_token_index(text)

    # init labels with default class label
    labels = ["-" for _ in range(len(words))]

    # encode cause using token index
    cause_seq = get_longest_tokens_sequence(cause, wi)
    for position in cause_seq:
        labels[position] = "C"

    # encode effect using token index
    effect_seq = get_longest_tokens_sequence(effect, wi)
    for position in effect_seq:
        labels[position] = "E"

    return labels


@dataclass
class Task2Data:
    index: str
    text: str
    cause: str
    effect: str
    labels: list[str]


def evaluate(
    truth: list[Task2Data], predict: list[Task2Data], classes: list[str]
) -> tuple[float, float, float, float]:
    """Fincausal 2020 Task 2 evaluation

    Calculates precision, recall, F1 and exact match comparing submitting data to
    reference data.

    :param truth: reference data set
    :param predict: submission data set
    :param classes: list of classes
    :return: evaluation metrics
    """
    exact_match = 0
    y_truth: list[str] = []
    y_predict: list[str] = []
    multi: dict[str, list[list[list[str]]]] = {}

    # First pass - process text sections with single causal relations and store others in `multi` dict()
    for t, p in zip(truth, predict):
        # Process Exact Match
        exact_match += all([x == y for x, y in zip(t.labels, p.labels)])

        # PRF: Text section with multiple causal relationship ?
        if t.index.count(".") == 2:
            # extract root index and add to the list to be processed later
            root_index = ".".join(t.index.split(".")[:-1])
            if root_index in multi:
                multi[root_index][0].append(t.labels)
                multi[root_index][1].append(p.labels)
            else:
                multi[root_index] = [[t.labels], [p.labels]]
        else:
            # Accumulate data for precision, recall, f1 scores
            y_truth.extend(t.labels)
            y_predict.extend(p.labels)

    # Second pass - deal with text sections having multiple causal relations
    for _, (possible, candidates) in multi.items():
        # for each possible combination of truth labels - try to find the best match in predicted labels
        # then repeat, removing this match from the list of remaining predicted labels

        for t in possible:
            best = None
            for p in candidates:
                f1 = metrics.f1_score(
                    t, p, labels=classes, average="weighted", zero_division=0
                )
                if best is None or f1 > best[1]:
                    best = (p, f1)
            assert best, "No candidate found for multi-label evaluation"

            # Use best to add to global evaluation
            y_truth.extend(t)

            y_predict.extend(best[0])
            # Remove best from list of candidate for next iteration
            candidates.remove(best[0])

        # Ensure all candidate predictions have been reviewed
        assert not candidates

    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_truth, y_predict, labels=classes, average="weighted", zero_division=0
    )
    return precision, recall, f1, exact_match / len(truth)


def get_data(csv_lines: list[str]) -> list[Task2Data]:
    """Load data from CSV with separator ';'.

    This should not contain the header line. Each line comprises 4 text fields: index,
    text, cause and effect.

    :param csv_lines: ;-separator lines, each line contains 4 fields, without header
    :return: each line converted to Task2Data
    """
    result: list[Task2Data] = []
    for line in csv_lines:
        index, text, cause, effect = line.strip().split(";")[:4]

        text = text.strip()
        cause = cause.strip()
        effect = effect.strip()
        labels = encode_causal_tokens(text, cause, effect)

        result.append(Task2Data(index, text, cause, effect, labels))

    return result


def evaluate_files(
    gold_file: str, submission_file: str, output_file: str | None = None
) -> None:
    """
    Evaluate Precision, Recall, F1 scores between gold_file and submission_file
    If output_file is provided, scores are saved in this file and printed to std output.

    :param gold_file: path to reference data
    :param submission_file: path to submitted data
    :param output_file: path to output file as expected by Codalab competition framework
    :return:
    """
    with open(gold_file, encoding="utf-8") as fp:
        ref_csv = fp.readlines()
    with open(submission_file, encoding="utf-8") as fp:
        sub_csv = fp.readlines()

    # Get data (skipping headers)
    y_true = get_data(ref_csv[1:])
    y_pred = get_data(sub_csv[1:])

    assert len(y_true) == len(
        y_pred
    ), "Different number of lines in reference and submission"
    assert all(
        [x.text == y.text for x, y in zip(y_true, y_pred)]
    ), "Input text mismatch"

    logging.debug("Discarded tokens: %d" % _discarded)

    # Process data using classes: -, C & E
    precision, recall, f1, exact_match = evaluate(y_true, y_pred, ["-", "C", "E"])

    scores = {
        "F1": f1,
        "Recall": recall,
        "Precision": precision,
        "ExactMatch": exact_match,
    }
    logging.info(
        "Evaluation metrics\n"
        + "\n".join([f"{k:<12} : {v}" for k, v in scores.items()])
    )

    if output_file:
        with open(output_file, "w") as fp:
            json.dump(scores, fp, indent=2)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, filename=None, format="%(levelname)-7s| %(message)s"
    )

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ref_file", help="reference file")
    parser.add_argument("pred_file", help="prediction file to evaluate")
    parser.add_argument(
        "--score-file",
        type=str,
        help="path to output score file (or stdout if not provided)",
    )

    args = parser.parse_args()
    evaluate_files(args.ref_file, args.pred_file, args.score_file)


if __name__ == "__main__":
    main()
