import argparse
import json
import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import tiktoken
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def load_data(file_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    data = json.loads(file_path.read_text())
    sentences = [item["input"] for item in data]
    return data, sentences


def encode_sentences(sentences: list[str], model_name: str) -> list[list[float]]:
    model = SentenceTransformer(model_name)
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


def get_item_clusters(
    data: list[dict[str, Any]],
    vectors: list[list[float]],
    cluster_labels: list[int],
    cluster_centers: list[list[float]],
) -> list[list[dict[str, Any]]]:
    num_clusters = max(cluster_labels) + 1
    clusters: list[list[tuple[float, dict[str, Any]]]] = [
        [] for _ in range(num_clusters)
    ]

    for item, vector, label in zip(data, vectors, cluster_labels):
        distance = calculate_distance(vector, cluster_centers[label])
        clusters[label].append((distance, item))

    for cluster in clusters:
        cluster.sort(key=lambda x: x[0])

    return [[item for _, item in cluster] for cluster in clusters]


def num_tokens(text: str) -> int:
    "Use tiktoken to count the number of tokens in the text with the GPT-3.5 tokeniser."
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    return len(tokens)


def auto_cot_filter(input: str) -> bool:
    """
    Based on Zhang et al. (2022), "Automatic Chain of Thought Prompting in Large
    Language Models."

    We ignore the number of steps and only consider the number of tokens.
    """
    return num_tokens(input) <= 60


def get_cluster_representatives(
    clusters: list[list[dict[str, Any]]], filter_func: Callable[[str], bool]
) -> list[dict[str, Any]]:
    """Select the representative item for each cluster.

    The representative item is the first item in the cluster that satisfies the filter
    function.

    """
    representatives: list[dict[str, Any]] = []
    for cluster in clusters:
        items = [item for item in cluster if filter_func(item["input"])]
        selected = items[0] if items else cluster[0]
        representatives.append(selected)
    return representatives


def suppress_warnings() -> None:
    "Remove annoying messages about tokenisers."
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore", module="transformers.convert_slow_tokenizer")


def main(
    input_file: Path, k: int, output_file: Path | None, model: str, seed: int
) -> None:
    suppress_warnings()

    data, sentences = load_data(input_file)

    vectors = encode_sentences(sentences, model)
    cluster_labels, cluster_centers = cluster_vectors(vectors, k, seed)
    item_clusters = get_item_clusters(data, vectors, cluster_labels, cluster_centers)
    cluster_representatives = get_cluster_representatives(
        item_clusters, lambda x: len(x) > 50
    )

    output_file = output_file or Path(f"{input_file.stem}_{k}_reps.json")
    print(f"Saving output to {output_file}")
    output_file.write_text(json.dumps(cluster_representatives, indent=2))


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
    args = parser.parse_args()

    main(args.input, args.k, args.output, args.model, args.seed)
