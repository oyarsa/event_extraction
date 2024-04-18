import inspect
import logging
import subprocess
import sys
from enum import Enum
from pathlib import Path
from types import FrameType
from typing import Any, ClassVar


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    def to_std(self) -> int:
        return getattr(logging, self.name)


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
    logger: logging.Logger,
    output_dir: Path,
    file_name: str = "train.log",
    mode: str = "a",
    level: LogLevel = LogLevel.INFO,
) -> None:
    logger.setLevel(level.to_std())

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    file_handler = logging.FileHandler(output_dir / file_name, mode=mode)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColourFormatter(fmt=fmt, datefmt=datefmt))
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
