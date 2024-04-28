import contextlib

import dataclasses
import logging
import multiprocessing
import multiprocessing.sharedctypes
import os
import platform
import queue
import random
import re
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Generator
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import torch
import torch.backends.mps
import transformers
from transformers import PreTrainedTokenizer

if TYPE_CHECKING:
    from trl.models.modeling_base import PreTrainedModelWrapper


def get_root(module: str) -> str:
    files = resources.files(module)
    with resources.as_file(files) as path:
        return path.parent.resolve()


PROJECT_ROOT = get_root("self_critique")


def resolve_path(path: str | Path) -> str:
    return str(PROJECT_ROOT / path)


def get_device() -> torch.device:
    "Returns MPS if available, CUDA if available, otherwise CPU device."
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return torch.device(device)


def is_git_repo_dirty() -> bool:
    try:
        output = subprocess.check_output(["git", "status", "--porcelain"])
        return output.strip() != b""
    except subprocess.CalledProcessError:
        return False


def get_current_commit() -> str:
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        return "unknown"
    else:
        if is_git_repo_dirty():
            git_hash += " (dirty)"
        return git_hash


class ColourFormatter(logging.Formatter):
    """Logging colored formatter.

    Adapted from https://stackoverflow.com/a/56944256/3638629
    """

    formats: ClassVar[dict[int, str]] = {
        logging.DEBUG: "\x1b[38;21m",  # grey
        logging.INFO: "\x1b[37m",  # white
        logging.WARNING: "\x1b[38;5;226m",  # yellow
        logging.ERROR: "\x1b[38;5;196m",  # red
        logging.CRITICAL: "\x1b[31;1m",  # bold red
    }
    reset: ClassVar[str] = "\x1b[0m"

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if colour := self.formats.get(record.levelno):
            return colour + msg + self.reset
        else:
            return msg


def setup_logger(
    logger: logging.Logger, output_dir: Path, log_level: str = "info"
) -> None:
    logger.setLevel(logging.getLevelName(log_level.upper()))

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColourFormatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(console_handler)


def save_model(
    model: "PreTrainedModelWrapper", tokeniser: PreTrainedTokenizer, output_dir: Path
) -> None:
    model.config.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    tokeniser.save_pretrained(output_dir)


def suppress_transformers_warnings() -> None:
    "Remove annoying messages about tokenisers and unititialised weights."
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore", module="transformers.convert_slow_tokenizer")
    transformers.logging.set_verbosity_error()


def set_seed(seed: int) -> None:
    "Set random seed for reproducibility."
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_metrics(
    metrics: dict[str, float], desc: str | None, logger: logging.Logger | None = None
) -> None:
    if logger is None:
        logger = logging.getLogger(__name__)

    desc = desc or "metrics"
    logger.info(f">>>> {desc.upper()}")

    padding = max(len(k) for k in metrics)
    for k, v in metrics.items():
        logger.info(f"    {k:>{padding}}: {v}")


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
