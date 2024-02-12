import contextlib
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
from typing import cast

import typer


def git_root() -> Path:
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )


def run_classifier(
    classifier_script: Path,
    config_path: Path,
    output_path: Path,
    output_name: str,
    debug: bool,
) -> None:
    args = [
        sys.executable,
        str(classifier_script),
        "--config",
        str(config_path),
        "--output_path",
        str(output_path),
        "--output_name",
        output_name,
        "--do_inference",
        "false",
    ]
    if debug:
        args.extend(["--num_epochs", "2", "--max_samples", "100"])
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
    classifier_script: Path,
    config: Path,
    output_path: Path,
    output_name: str,
    sleep_s: int = 1,
    debug: bool = False,
) -> None:
    with track_memory(sleep_s) as stats:
        run_classifier(classifier_script, config, output_path, output_name, debug=debug)
    print(
        f"Max GPU memory usage: {stats['used']}/{stats['total']}"
        f" ({stats['free']} free) MiB"
    )


def run_evaluator(evaluator_script: Path, metric: str, data_file: Path) -> None:
    args = [
        sys.executable,
        str(evaluator_script),
        metric,
        str(data_file),
    ]
    subprocess.run(args, check=True)


def evaluate(evaluator_script: Path, output_file: Path) -> None:
    print()
    print(">>>> EVALUATING")
    for metric in ["agreement", "krippendorff", "spearman", "cohen"]:
        run_evaluator(evaluator_script, metric, output_file)
        print()


def main(
    dir_name: Path = typer.Argument(help="Path to output directory."),
    run_name: str = typer.Argument(help="Name of the run."),
    config: Path = typer.Argument(help="Path to the config file."),
    output_file_name: str = typer.Option(
        "test_results.json", help="Name of the the output file."
    ),
    classifier: Path = typer.Option(
        "evaluation/classifier.py", help="Path to the classifier script."
    ),
    evaluator: Path = typer.Option(
        git_root() / "agreement" / "calc.py", help="Path to the evaluator script."
    ),
    debug: bool = typer.Option(
        False,
        help="Use only a subset of the data (100 samples) and 2 epcohs for a faster"
        " debug cycle.",
    ),
) -> None:
    "Train classifier and evaluate output for agreement and correlation statistics."
    run_path = dir_name / run_name
    output_file = run_path / output_file_name

    classify(classifier, config, dir_name, run_name, debug=debug)
    evaluate(evaluator, output_file)


if __name__ == "__main__":
    typer.run(main)
