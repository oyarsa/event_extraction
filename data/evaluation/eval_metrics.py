"""Calculate element-wise metrics for a given dataset.

The script calculates the following metrics:
- Exact match (EM)
- ROUGE-L
- BLEU
- BLEURT
- BERTScore
- F1

For all of them, the metrics are calculated element-wise, i.e., for each item in the
dataset. This is so these metrics can be compared with the human evaluation.

The input JSON file should have the following structure:
- input (str): the input text
- output (str): the model's output
- annotation (str): the human annotation for the extraction
- gold (str): the human evaluation of the model output with respect to the annotation
"""

import argparse
import contextlib
import gzip
import io
import itertools
import json
import os
import re
import string
import warnings
from collections import Counter
from collections.abc import Iterable
from typing import Any, TypeVar

import evaluate
import transformers
from tqdm import tqdm

T = TypeVar("T")


def batched(iterable: Iterable[T], n: int) -> list[list[T]]:
    if n < 1:
        raise ValueError("n must be at least one")

    it = iter(iterable)
    batches: list[list[T]] = []
    while batch := list(itertools.islice(it, n)):
        batches.append(batch)
    return batches


def exact_match(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item | {"pred": int(item["output"] == item["annotation"])} for item in data]


def rougel(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rouge = evaluate.load("rouge")
    scores: list[float] = []

    for item in tqdm(data):
        result = rouge.compute(
            predictions=[item["output"]], references=[item["annotation"]]
        )
        assert result is not None, "ROUGE failed to compute the score."
        scores.append(result["rougeL"])

    return [item | {"pred": score} for item, score in zip(data, scores)]


def bleu(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bleu = evaluate.load("bleu")
    scores: list[float] = []

    for item in tqdm(data):
        result = bleu.compute(
            predictions=[item["output"]], references=[[item["annotation"]]]
        )
        assert result is not None, "BLEU failed to compute the score."
        scores.append(result["bleu"])

    return [item | {"pred": score} for item, score in zip(data, scores)]


def bleurt(data: list[dict[str, Any]], batch_size: int = 16) -> list[dict[str, Any]]:
    try:
        bleurt = evaluate.load("bleurt", "BLEURT-20-D12")
    except ImportError as e:
        raise ImportError(
            "BLEURT is not installed. Install from the repository: "
            "https://github.com/google-research/bleurt/tree/master"
        ) from e

    golds = [item["annotation"] for item in data]
    preds = [item["output"] for item in data]
    batched_golds = batched(golds, batch_size)
    batched_preds = batched(preds, batch_size)

    scores: list[float] = []
    for batched_gold, batched_pred in tqdm(
        zip(batched_golds, batched_preds), total=len(batched_golds)
    ):
        result = bleurt.compute(predictions=batched_pred, references=batched_gold)
        assert result, "BLEURT failed to compute the score."
        scores.extend(result["scores"])

    return [item | {"pred": score} for item, score in zip(data, scores)]


# F1 {{{


def get_tokens(s: str) -> list[str]:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.split()


def parse_instance(answer: str) -> dict[str, list[str]]:
    matches = re.findall(r"\[Cause\](.*?)\[Relation\].*?\[Effect\](.*?)$", answer)
    if not matches:
        return {"Cause": [], "Effect": []}

    causes, effects = matches[0]
    causes = sorted(c.strip() for c in causes.split("|") if c.strip())
    effects = sorted(e.strip() for e in effects.split("|") if e.strip())

    return {"Cause": causes, "Effect": effects}


def calc_f1_sentence(gold: list[str], pred: list[str]) -> float:
    gold_toks = get_tokens(" ".join(gold))
    pred_toks = get_tokens(" ".join(pred))

    if not gold_toks or not pred_toks:
        return 0

    common = Counter(gold_toks) & Counter(pred_toks)
    precision = sum(common.values()) / len(pred_toks)
    recall = sum(common.values()) / len(gold_toks)

    if precision + recall != 0:
        return (2 * precision * recall) / (precision + recall)
    else:
        return 0


def calc_f1(sentence1: str, sentence2: str) -> float:
    entities1 = parse_instance(sentence1)
    entities2 = parse_instance(sentence2)

    f1_cause = calc_f1_sentence(entities1["Cause"], entities2["Cause"])
    f1_effect = calc_f1_sentence(entities1["Effect"], entities2["Effect"])

    return (f1_cause + f1_effect) / 2


def f1(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item | {"pred": calc_f1(item["output"], item["annotation"])} for item in data
    ]


# F1 }}}


def bertscore(data: list[dict[str, Any]], batch_size: int = 16) -> list[dict[str, Any]]:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore", module="transformers.convert_slow_tokenizer")
    transformers.logging.set_verbosity_error()

    bertscore = evaluate.load("bertscore")
    scores: list[float] = []

    preds = [item["output"] for item in data]
    golds = [item["annotation"] for item in data]

    with contextlib.redirect_stderr(io.StringIO()):
        batched_preds = batched(preds, batch_size)
        batched_golds = batched(golds, batch_size)

        for batch_pred, batch_gold in tqdm(
            zip(batched_preds, batched_golds), total=len(batched_preds)
        ):
            result = bertscore.compute(
                predictions=batch_pred, references=batch_gold, lang="en"
            )
            assert result, "BERTScore failed to compute the score."
            scores.extend(result["f1"])

    return [item | {"pred": score} for item, score in zip(data, scores)]


def main(
    input_file: str,
    use_rougel: bool,
    use_bleu: bool,
    use_bleurt: bool,
    use_bertscore: bool,
    all_metrics: bool,
    compress: bool,
) -> None:
    metrics = {
        "em": exact_match,
        "f1": f1,
        "rougel": rougel,
        "bleu": bleu,
        "bleurt": bleurt,
        "bertscore": bertscore,
    }
    if all_metrics:
        metrics_to_use = set(metrics.keys())
    else:
        metrics_to_use = {"em", "f1"}

    if use_rougel:
        metrics_to_use.add("rougel")
    if use_bleu:
        metrics_to_use.add("bleu")
    if use_bleurt:
        metrics_to_use.add("bleurt")
    if use_bertscore:
        metrics_to_use.add("bertscore")

    with open(input_file) as f:
        data = json.load(f)

    for name in metrics_to_use:
        print(f">>> Computing {name}...")

        new_data = metrics[name](data)
        path, ext = os.path.splitext(input_file)
        out_name = f"{path}.{name}{ext}"
        with gzip.open(out_name, "wt") if compress else open(out_name, "w") as f:
            json.dump(new_data, f, indent=2)

        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_file", type=str, help="Path to the input JSON file.")
    parser.add_argument(
        "--rougel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute ROUGE-L metric. Default: %(default)s.",
    )
    parser.add_argument(
        "--bleu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute BLEU metric. Default: %(default)s.",
    )
    parser.add_argument(
        "--bleurt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute BLEURT metric. Default: %(default)s.",
    )
    parser.add_argument(
        "--bertscore",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute BERTScore metric. Default: %(default)s.",
    )
    parser.add_argument(
        "--all",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Turn on all metrics. Default: %(default)s.",
    )
    parser.add_argument(
        "--compress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to compress the output files using gzip. Default: %(default)s.",
    )
    args = parser.parse_args()
    main(
        args.input_file,
        args.rougel,
        args.bleu,
        args.bleurt,
        args.bertscore,
        args.all,
        args.compress,
    )
