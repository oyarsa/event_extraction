import argparse
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


# sourcery skip: avoid-single-character-names-functions
def p(*args: Any, **kwargs: Any) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | ", *args, **kwargs)


def get_device() -> str:
    "Get the device to use for SentenceTransformer, including MPS for Apple Silicon."
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def suppress_warnings() -> None:
    "Remove annoying messages about tokenisers."
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore", module="transformers.convert_slow_tokenizer")


def load_data(file_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    data = json.loads(file_path.read_text())
    sentences = [item["input"] for item in data]
    return data, sentences


def encode_sentences(sentences: list[str], model_name: str, device: str) -> npt.NDArray:
    p(3.1)
    model = SentenceTransformer(model_name, device=device)
    p(3.2)
    return cast(npt.NDArray, model.encode(sentences))


def cluster_vectors(
    vectors: npt.NDArray, k: int, seed: int
) -> tuple[npt.NDArray, npt.NDArray]:
    """Cluster vectors using KMeans."""
    kmeans = KMeans(n_clusters=k, random_state=seed).fit(vectors)
    return cast(npt.NDArray, kmeans.labels_), kmeans.cluster_centers_


def get_cluster_centers(
    data: list[dict[str, Any]],
    vectors: npt.NDArray,
    labels: npt.NDArray,
    centers: npt.NDArray,
) -> list[dict[str, Any]]:
    clusters = [[] for _ in range(max(labels) + 1)]

    for item, vector, label in zip(data, vectors, labels):
        distance = np.linalg.norm(vector - centers[label])
        clusters[label].append((distance, item))

    return [max(cluster, key=lambda x: x[0])[1] for cluster in clusters]


def main(
    input_file: Path, k: int, output_file: Path, model: str, seed: int, device: str
) -> None:
    p(1)
    suppress_warnings()
    p(2)
    data, sentences = load_data(input_file)
    p(3)
    vectors = encode_sentences(sentences, model, device)
    p(4)
    cluster_labels, cluster_center_vectors = cluster_vectors(vectors, k, seed)
    p(5)
    cluster_center_items = get_cluster_centers(
        data, vectors, cluster_labels, cluster_center_vectors
    )
    p(6)

    output_file = output_file or Path(f"{input_file.stem}_{k}_clustered.json")
    print(f"Saving output to {output_file}")
    output_file.write_text(json.dumps(cluster_center_items, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster sentences using SentenceBERT and K-means",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Path to the input JSON file")
    parser.add_argument("-k", type=int, default=8, help="Number of clusters")
    parser.add_argument("--output", type=Path, help="Path to the output JSON file")
    parser.add_argument(
        "--model",
        type=str,
        default="all-mpnet-base-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for KMeans clustering"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=get_device(),
        help="Device for SentenceTransformer",
    )
    args = parser.parse_args()

    main(args.input, args.k, args.output, args.model, args.seed, args.device)
