import contextlib
import dataclasses
import logging
import multiprocessing
import multiprocessing.sharedctypes
import platform
import queue
import re
import subprocess
import time
import warnings
from collections.abc import Callable, Generator
from typing import Any, cast

import krippendorff
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
)


@dataclasses.dataclass
class EvaluationResult:
    golds: list[int]
    preds: list[int]
    passages: list[str]
    outputs: list[str]
    annotations: list[str]
    loss: float
    tags: list[str] | None = None


def spearman(x: list[int], y: list[int]) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        return spearmanr(x, y)[0]


def calc_metrics(
    results: EvaluationResult, average: str = "binary", mse: bool = False
) -> dict[str, float]:
    x, y = results.golds, results.preds

    acc = accuracy_score(x, y)
    prec, rec, f1, _ = precision_recall_fscore_support(
        x,
        y,
        average=average,
        zero_division=0,  # type: ignore
    )
    krippendorff_ = krippendorff.alpha(
        [x, y], level_of_measurement="ordinal" if mse else "nominal"
    )

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "agreement": sum(a == b for a, b in zip(x, y)) / len(x),
        "krippendorff": krippendorff_,
        "spearman": spearman(x, y),
        "cohen": cohen_kappa_score(x, y),
        "eval_loss": results.loss,
    }
    if mse:
        metrics["mse"] = sum((x_i - y_i) ** 2 for x_i, y_i in zip(x, y)) / len(x)
    return metrics


def report_metrics(
    logger: logging.Logger, metrics: dict[str, float], desc: str, mse: bool = False
) -> None:
    logger.info(
        f"{desc} results\n"
        f"    Accuracy      : {metrics['accuracy']:.4f}\n"
        f"    Precision     : {metrics['precision']:.4f}\n"
        f"    Recall        : {metrics['recall']:.4f}\n"
        f"    F1            : {metrics['f1']:.4f}\n"
        f"    Agreement     : {metrics['agreement']:.4f}\n"
        f"    Krippendorff  : {metrics['krippendorff']:.4f}\n"
        f"    Spearman      : {metrics['spearman']:.4f}\n"
        f"    Cohen         : {metrics['cohen']:.4f}\n"
        f"    Eval Loss     : {metrics['eval_loss']:.4f}\n"
        + (f"    MSE           : {metrics['mse']:.4f}\n" if mse else "")
    )


@dataclasses.dataclass
class UsageResult:
    used: int
    free: int
    total: int


def get_nvidia_memory_usage() -> UsageResult:
    nvidia_smi_output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    match = re.findall(r"(\d+),(\s+\d+),(\s+\d+)", nvidia_smi_output)
    if len(match) != 1:
        raise ValueError(f"Could not parse nvidia-smi output:\n{nvidia_smi_output}")
    used, free, total = map(int, match[0])

    return UsageResult(used, free, total)


def get_mac_memory_usage() -> UsageResult:
    vm_stat_output = subprocess.check_output(["vm_stat"], text=True)

    # Parse page size
    page_size_match = re.search(r"page size of (\d+) bytes", vm_stat_output)
    if page_size_match is None:
        raise ValueError(
            f"Could not parse page size from vm_stat output:\n{vm_stat_output}"
        )
    page_size = int(page_size_match[1]) / 1024 / 1024  # Convert bytes to MiB

    memory_stats = {"free": 0, "active": 0, "inactive": 0, "speculative": 0}

    # Parse memory statistics
    for line in vm_stat_output.splitlines():
        for key in memory_stats:
            if key in line and (m := re.search(r"(\d+)\.", line)):
                value = int(m[1])
                memory_stats[key] = int(value * page_size)

    used_memory = (
        memory_stats["active"] + memory_stats["inactive"] + memory_stats["speculative"]
    )
    free_memory = memory_stats["free"]
    total_memory = used_memory + free_memory

    return UsageResult(used_memory, free_memory, total_memory)


def get_memory_usage() -> UsageResult:
    try:
        subprocess.check_output(["nvidia-smi"])
        return get_nvidia_memory_usage()
    except FileNotFoundError as e:
        if platform.system() == "Darwin":
            return get_mac_memory_usage()
        raise ValueError("Unsupported platform: need nvidia-smi or macOS") from e


def memory_usage_process(
    stop_signal: multiprocessing.sharedctypes.Synchronized,
    result_queue: multiprocessing.Queue,
    sleep_s: int = 1,
) -> None:
    memory_usage = get_memory_usage()

    while not stop_signal.value:
        new_memory_usage = get_memory_usage()
        memory_usage.used = max(memory_usage.used, new_memory_usage.used)
        memory_usage.free = min(memory_usage.free, new_memory_usage.free)

        time.sleep(sleep_s)

    result_queue.put(memory_usage)


@contextlib.contextmanager
def track_memory(sleep_s: int = 1) -> Generator[UsageResult, None, None]:
    stop_signal = cast(
        multiprocessing.sharedctypes.Synchronized, multiprocessing.Value("i", 0)
    )
    result_queue = multiprocessing.Queue()

    process = multiprocessing.Process(
        target=memory_usage_process, args=(stop_signal, result_queue, sleep_s)
    )
    process.start()

    stats = UsageResult(-1, -1, -1)
    try:
        yield stats
    finally:
        stop_signal.value = 1
        process.join()

        with contextlib.suppress(queue.Empty):
            new_stats = result_queue.get(timeout=sleep_s + 1)
            stats.used = new_stats.used
            stats.free = new_stats.free
            stats.total = new_stats.total


def report_gpu_memory(
    main: Callable[..., Any], logger: logging.Logger, sleep_s: int = 1
) -> None:
    with track_memory(sleep_s) as stats:
        main()

    output = [
        "----------------",
        "GPU MEMORY USAGE",
        "----------------",
        f"Max usage: {stats.used}/{stats.total} ({stats.free} free) MiB",
    ]
    logger.info("\n".join(output))
