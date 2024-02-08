import contextlib
import json
import multiprocessing
import multiprocessing.sharedctypes
import queue
import re
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import typer


def git_root() -> Path:
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )


CLASSIFIER_SCRIPT = Path("classifier.py")
EVALUATOR_SCRIPT = git_root() / "agreement" / "calc_more.py"


def run_classifier(config_path: Path, output_path: Path, output_name: str) -> None:
    args = [
        sys.executable,
        str(CLASSIFIER_SCRIPT),
        "--config",
        str(config_path),
        "--output_path",
        str(output_path),
        "--output_name",
        output_name,
    ]
    subprocess.run(args, check=True)


def get_gpu_memory_usage() -> dict[str, int]:
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

    return {
        "used": used,
        "free": free,
        "total": total,
    }


def gpu_memory_usage_process(
    stop_signal: multiprocessing.sharedctypes.Synchronized,
    result_queue: multiprocessing.Queue,
    sleep_s: int = 1,
) -> None:
    memory_usage = get_gpu_memory_usage()

    while not stop_signal.value:
        new_memory_usage = get_gpu_memory_usage()
        memory_usage["used"] = max(memory_usage["used"], new_memory_usage["used"])
        memory_usage["free"] = min(memory_usage["free"], new_memory_usage["free"])

        time.sleep(sleep_s)

    result_queue.put(memory_usage)


@contextlib.contextmanager
def track_memory(sleep_s: int = 1) -> Generator[dict[str, int], None, None]:
    stop_signal = cast(
        multiprocessing.sharedctypes.Synchronized, multiprocessing.Value("i", 0)
    )
    result_queue = multiprocessing.Queue()

    process = multiprocessing.Process(
        target=gpu_memory_usage_process, args=(stop_signal, result_queue, sleep_s)
    )
    process.start()

    stats: dict[str, int] = defaultdict(lambda: -1)
    try:
        yield stats
    finally:
        stop_signal.value = 1
        process.join()

        with contextlib.suppress(queue.Empty):
            stats |= result_queue.get(timeout=sleep_s + 1)


def classify(
    config: Path, output_path: Path, output_name: str, sleep_s: int = 1
) -> None:
    with track_memory(sleep_s) as stats:
        run_classifier(config, output_path, output_name)
    print(
        f"Max GPU memory usage: {stats['used']}/{stats['total']}"
        f" ({stats['free']} free) MiB"
    )


def run_evaluator(metric: str, base_file: Path, eval_file: Path) -> None:
    args = [
        sys.executable,
        str(EVALUATOR_SCRIPT),
        metric,
        str(base_file),
        f"{eval_file},valid",
    ]
    subprocess.run(args, check=True)


def evaluate(human_file: Path, output_file: Path) -> None:
    print()
    print(">>>> EVALUATING")
    for metric in ["agreement", "krippendorff", "spearman", "cohen"]:
        run_evaluator(metric, human_file, output_file)
        print()


def transform_data(data: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Transforms the data from the classifier into the format expected by the
    agreement evaluation script.

    The classifier format is:
    - passage: str
    - pred: int
    - annotation: str

    The agreement evaluation format is:
    - input: str
    - reward_label: str
    - gold: str
    """
    return [
        {
            "input": item["passage"],
            "reward_label": "VALID" if item.get("pred") == 1 else "INVALID",
            "gold": item["annotation"],
        }
        for item in data
    ]


def transform(input_file: Path, output_file: Path) -> None:
    input_data = json.loads(input_file.read_text())
    output_data = transform_data(input_data)
    output_file.write_text(json.dumps(output_data, indent=2))


def main(
    dir_name: Path = typer.Argument(help="Path to output directory"),
    run_name: str = typer.Argument(help="Name of the run"),
    config: Path = typer.Argument(help="Path to the config file"),
    human_file: Path = typer.Argument(help="Path to the human base file"),
) -> None:
    "Train classifier and evaluate output for agreement."
    run_path = dir_name / run_name
    input_file = run_path / "test_results.json"
    output_file = run_path / "knowwhy_valid.json"

    classify(config, dir_name, run_name)
    transform(input_file, output_file)
    evaluate(human_file, output_file)


if __name__ == "__main__":
    typer.run(main)
