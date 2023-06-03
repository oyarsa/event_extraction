import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import typer


def levensthein_distance(str1: str, str2: str) -> int:
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i, c1 in enumerate(str1, 1):
        for j, c2 in enumerate(str2, 1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + (c1 != c2),
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )

    return dp[m][n]


def levenshtein_similarity(str1: str, str2: str) -> float:
    len_sum = len(str1) + len(str2)
    dist = levensthein_distance(str1, str2)
    return 1 - dist / len_sum


def set_to_map(s: set[str]) -> dict[str, int]:
    return {k: i for i, k in enumerate(s)}


def word_vector(vocab: dict[str, int], words: list[str]) -> list[int]:
    vec = [0] * len(vocab)
    for w in words:
        if w in vocab:
            vec[vocab[w]] += 1
    return vec


def bag_of_words(vocab: dict[str, int], s: str) -> list[int]:
    return word_vector(vocab, s.split())


def make_vocab(corpus: list[str]) -> dict[str, int]:
    vocab: set[str] = set()
    for s in corpus:
        vocab.update(s.split())
    return set_to_map(vocab)


def jaccard(vector1: list[int], vector2: list[int]) -> float:
    sum_min = sum(min(v1, v2) for v1, v2 in zip(vector1, vector2))
    sum_max = sum(max(v1, v2) for v1, v2 in zip(vector1, vector2))
    return sum_min / sum_max


def norm(vector: list[int]) -> float:
    return sum(v**2 for v in vector) ** 0.5


def cosine(vector1: list[int], vector2: list[int]) -> float:
    dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
    return dot_product / (norm(vector1) * norm(vector2))


def jaccard_similarity(str1: str, str2: str) -> float:
    vocab = make_vocab([str1, str2])
    vec1 = bag_of_words(vocab, str1)
    vec2 = bag_of_words(vocab, str2)
    return jaccard(vec1, vec2)


def cosine_similarity(str1: str, str2: str) -> float:
    vocab = make_vocab([str1, str2])
    vec1 = bag_of_words(vocab, str1)
    vec2 = bag_of_words(vocab, str2)
    return cosine(vec1, vec2)


SIMILARITY_THRESHOLD = 0.75


def is_similar(
    str1: str, str2: str, sim_func: Callable[[str, str], float], threshold: float
) -> bool:
    return sim_func(str1, str2) >= threshold


def is_symmetric_substring(str1: str, str2: str) -> bool:
    return str1 in str2 or str2 in str1


def classify_pair(
    str1: str, str2: str, sim_func: Callable[[str, str], float], threshold: float
) -> str:
    if is_symmetric_substring(str1, str2):
        return "ENTAILMENT"
    if is_similar(str1, str2, sim_func, threshold):
        return "CONTRADICTION"
    return "NEUTRAL"


def classifiy(
    data: list[dict[str, str]], sim_func: Callable[[str, str], float], threshold: float
) -> list[str]:
    return [
        classify_pair(d["sentence1"], d["sentence2"], sim_func, threshold) for d in data
    ]


def calc_accuracy(
    data: list[dict[str, str]], sim_func: Callable[[str, str], float], threshold: float
) -> float:
    gold = [d["label"] for d in data]
    pred = classifiy(data, sim_func, threshold)
    correct = sum(g == p for g, p in zip(gold, pred))
    return correct / len(gold)


def get_sim_func(s: str) -> Callable[[str, str], float]:
    if s == "jaccard":
        return jaccard_similarity
    if s == "cosine":
        return cosine_similarity
    if s == "levenshtein":
        return levenshtein_similarity
    raise ValueError(f"Unknown similarity function: {s}")


def main(
    file: Path,
    similarity: str = "jaccard",
    max_samples: Optional[int] = None,
    threshold: float = SIMILARITY_THRESHOLD,
) -> None:
    data = json.loads(file.read_text())[:max_samples]
    score = calc_accuracy(data, get_sim_func(similarity), threshold)
    print(f"Accuracy: {score:.2%}")


if __name__ == "__main__":
    if not hasattr(sys, "ps1"):  # not in interactive mode
        typer.run(main)


def rcf():
    "..."
    str1 = "hey there man hey"
    str2 = "hey there girl there there"
    vocab = make_vocab([str1, str2])
    bow1 = bag_of_words(vocab, str1)
    bow2 = bag_of_words(vocab, str2)
    print(bow1, "->", str1)
    print(bow2, "->", str2)
