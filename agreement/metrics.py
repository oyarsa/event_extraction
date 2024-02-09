from typing import SupportsInt

import krippendorff
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

AVAILABLE_METRICS = ["agreement", "krippendorff", "spearman", "cohen"]


def calculate_metric(
    metric: str, xs: list[SupportsInt], ys: list[SupportsInt]
) -> float:
    if metric not in AVAILABLE_METRICS:
        raise ValueError(f"Unknown metric: {metric}")

    x = [int(value) for value in xs]
    y = [int(value) for value in ys]

    match metric:
        case "agreement":
            return sum(a == b for a, b in zip(x, y)) / len(x)
        case "krippendorff":
            return krippendorff.alpha([x, y], level_of_measurement="nominal")
        case "spearman":
            return spearmanr(x, y)[0]
        case "cohen":
            return cohen_kappa_score(x, y)
        case _:
            raise ValueError(f"Metric not implmeneted: {metric}")
