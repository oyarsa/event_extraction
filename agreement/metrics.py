import warnings
from collections.abc import Iterable
from typing import SupportsFloat

import krippendorff
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

AVAILABLE_METRICS = ["agreement", "krippendorff", "spearman", "cohen", "pearson"]


def are_numbers_ints(xs: list[float]) -> bool:
    return all(x.is_integer() for x in xs)


def agreement(x: list[float], y: list[float]) -> float:
    if not are_numbers_ints(x) or not are_numbers_ints(y):
        warnings.warn("Agreement only works with integer or boolean values")
        return float("nan")
    return sum(a == b for a, b in zip(x, y)) / len(x)


def calculate_metric(
    metric: str, xs: Iterable[SupportsFloat], ys: Iterable[SupportsFloat]
) -> float:
    if metric not in AVAILABLE_METRICS:
        raise ValueError(f"Unknown metric: {metric}")

    x = [float(value) for value in xs]
    y = [float(value) for value in ys]

    match metric:
        case "agreement":
            return agreement(x, y)
        case "krippendorff":
            return krippendorff.alpha([x, y], level_of_measurement="nominal")
        case "spearman":
            return spearmanr(x, y)[0]
        case "pearson":
            return pearsonr(x, y)[0]
        case "cohen":
            return cohen_kappa_score(x, y)
        case _:
            raise ValueError(f"Metric not implmeneted: {metric}")
