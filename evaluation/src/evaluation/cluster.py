import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


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


def encode_sentences(
    sentences: list[str], model_name: str, device: str
) -> list[list[float]]:
    model = SentenceTransformer(model_name, device=device)
    vectors = model.encode(sentences)
    return cast(npt.NDArray, vectors).tolist()


def cluster_vectors(
    vectors: list[list[float]], k: int, seed: int
) -> tuple[list[int], list[list[float]]]:
    """Cluster vectors using KMeans.

    Returns a tuple of cluster labels (each int is a label) and cluster centers
    (each item in the list is a vector in list form).
    """
    kmeans = KMeans(n_clusters=k, random_state=seed)
    cluster_labels = kmeans.fit_predict(vectors)
    cluster_centers = kmeans.cluster_centers_
    return cluster_labels.tolist(), cluster_centers.tolist()


def calculate_distance(vector: list[float], center: list[float]) -> float:
    dist = np.linalg.norm(np.array(vector) - np.array(center))
    return float(dist)


def get_cluster_centers(
    data: list[dict[str, Any]],
    vectors: list[list[float]],
    cluster_labels: list[int],
    cluster_centers: list[list[float]],
) -> list[dict[str, Any]]:
    num_clusters = max(cluster_labels) + 1
    clusters: list[list[tuple[float, dict[str, Any]]]] = [
        [] for _ in range(num_clusters)
    ]

    for item, vector, label in zip(data, vectors, cluster_labels):
        distance = calculate_distance(vector, cluster_centers[label])
        clusters[label].append((distance, item))

    return [max(cluster, key=lambda x: x[0])[1] for cluster in clusters]


def main(
    input_file: Path,
    k: int,
    output_file: Path | None,
    model: str,
    seed: int,
    device: str,
) -> None:
    suppress_warnings()

    data, sentences = load_data(input_file)

    vectors = encode_sentences(sentences, model, device)
    cluster_labels, cluster_centers_vectors = cluster_vectors(vectors, k, seed)
    cluster_center_items = get_cluster_centers(
        data, vectors, cluster_labels, cluster_centers_vectors
    )

    output_file = output_file or Path(f"{input_file.stem}_{k}_clustered.json")
    print(f"Saving output to {output_file}")
    output_file.write_text(json.dumps(cluster_center_items, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster sentences using SentenceBERT and K-means",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Path to the input JSON file")
    parser.add_argument("--k", type=int, default=8, help="Number of clusters")
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Model name for SentenceTransformer",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for KMeans clustering",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=get_device(),
        help="Device to use for SentenceTransformer",
    )
    args = parser.parse_args()

    main(args.input, args.k, args.output, args.model, args.seed, args.device)
