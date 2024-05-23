"""Evaluate model answers using SentenceTransformer and cosine similarity.

Requires the input JSON file to have the following structure:
- gold (original text reference)
- output (extraction model output)
- valid (boolean indicating if the output is valid from human annotation)

Uses the reference `valid` field from the input to determine the best classification
threshold.
"""

import argparse
import json
from pathlib import Path
from typing import TextIO

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def find_threshold(valids: list[bool], cosine: list[float]) -> tuple[float, float]:
    thresholds = set(cosine)
    best_threshold = 0
    best_agreement = 0

    for threshold in thresholds:
        predictions = (c >= threshold for c in cosine)
        correct = sum(p == v for p, v in zip(predictions, valids))
        accuracy = correct / len(valids)

        if accuracy > best_agreement:
            best_agreement = accuracy
            best_threshold = threshold

    return best_threshold, best_agreement


def classify(
    model: SentenceTransformer, gold: list[str], pred: list[str]
) -> list[float]:
    """Classify if pred is valid with SentenceTransformer + cosine similarity to gold."""
    gold_emb = model.encode(gold)
    pred_emb = model.encode(pred)

    cosine_sims = cosine_similarity(gold_emb, pred_emb)
    return [float(sim) + 1 / 2 for sim in cosine_sims.diagonal()]


def main(
    input_file: TextIO,
    output_file: Path | None,
    model_name: str,
    num_samples: int | None,
) -> None:
    model = SentenceTransformer(model_name)
    data = json.load(input_file)[:num_samples]

    cosine = classify(model, [d["gold"] for d in data], [d["output"] for d in data])

    threshold, agreement = find_threshold([d["valid"] for d in data], cosine)
    result = [c >= threshold for c in cosine]
    num_valid = sum(result)

    print(
        f"Model: {model_name}",
        f"Threshold: {threshold}",
        f"Valid: {num_valid}/{len(result)} ({num_valid/len(result):.2%})",
        f"Agreement: {agreement:.2%}",
        sep="\n",
    )

    data = [
        d | {"cosine": c, "pred": int(r), "gold": int(d["valid"])}
        for d, c, r in zip(data, cosine, result)
    ]

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(data, indent=2))
    else:
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        help="Input JSON file containing gold and predicted answers. If '-' is given,"
        " read from stdin.",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        help="Output JSON file containing cosine similarity and validity. If not given,"
        " will write to stdout",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Name of the SentenceTransformer model to use. Default: %(default)s.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to print. Default: all.",
    )
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.model, args.num_samples)
