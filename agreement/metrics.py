from typing import Any

import krippendorff
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

AVAILABLE_METRICS = ["agreement", "krippendorff", "kdf", "spearman", "cohen"]


def calculate_metric(metric: str, x: list[Any], y: list[Any]) -> float:
    x = [int(value) for value in x]
    y = [int(value) for value in y]

    match metric:
        case "agreement":
            return sum(a == b for a, b in zip(x, y)) / len(x)
        case "krippendorff" | "kdf":
            return krippendorff.alpha([x, y], level_of_measurement="nominal")
        case "spearman":
            return spearmanr(x, y)[0]
        case "cohen":
            return cohen_kappa_score(x, y)
        case _:
            raise ValueError(f"Unknown metric: {metric}")
