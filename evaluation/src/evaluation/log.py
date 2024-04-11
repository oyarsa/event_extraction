import logging
import sys
from pathlib import Path


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
