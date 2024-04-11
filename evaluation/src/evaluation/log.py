import inspect
import logging
import subprocess
import sys
from pathlib import Path
from types import FrameType
from typing import Any


def setup_logger(
    logger: logging.Logger,
    output_dir: Path,
    file_name: str = "train.log",
    mode: str = "a",
) -> None:
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(output_dir / file_name, mode=mode)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def make_serialisable(obj: Any) -> Any:
    """Convert an object to a JSON serialisable form.

    This means that objects that aren't serialisable (e.g. Path objects) are converted
    to strings.
    """
    match obj:
        case dict():
            return {k: make_serialisable(v) for k, v in obj.items()}
        case list() | tuple():
            return [make_serialisable(v) for v in obj]
        case int() | float() | bool() | str() | None:
            return obj
        case _:
            return str(obj)


def get_func_params(
    current_frame: FrameType | None, serialise: bool = False
) -> dict[str, Any]:
    """Get the parameters of the function that called this function (names and values).

    Optionally, ensure the values are JSON serialisable.
    """
    if current_frame is None or current_frame.f_back is None:
        return {}

    arg_names, _, _, arg_values = inspect.getargvalues(current_frame.f_back)
    args = {name: arg_values[name] for name in arg_names} | arg_values.get("kwargs", {})

    if serialise:
        return make_serialisable(args)
    return args


def get_current_commit_shorthash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        return "unknown"
