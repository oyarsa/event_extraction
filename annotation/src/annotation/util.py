import hashlib
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import yaml
from typing_extensions import override

logger = logging.getLogger("annotation.util")


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def check_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed


def check_admin_password(password: str) -> bool:
    admin_password = get_config().admin_password
    return check_password(password, admin_password)


@dataclass
class Config:
    log_path: Path
    annotation_dir: Path
    answer_dir: Path
    split_to_user_file: Path
    instructions_file: Path
    admin_password: str
    completion_code: str


def get_config() -> Config:
    """Loads the configuration from a YAML file.

    The path to the file can be set with the ANNOTATION_CONFIG_PATH environment variable.
    The default is config/config.yaml, relative to the current directory of where the
    script is run.

    The paths listed in the configuration file are relative to current directory too.
    """
    config_path = os.environ.get("ANNOTATION_CONFIG_PATH", "config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return Config(
        log_path=Path(config["log_path"]),
        annotation_dir=Path(config["data"]["input"]),
        answer_dir=Path(config["data"]["answers"]),
        split_to_user_file=Path(config["data"]["split_to_user"]),
        instructions_file=Path(config["instructions"]),
        admin_password=config["admin_password"],
        completion_code=config["completion_code"],
    )


class ColourFormatter(logging.Formatter):
    """Logging colored formatter.

    Adapted from https://stackoverflow.com/a/56944256/3638629
    """

    formats: ClassVar[dict[int, str]] = {
        logging.DEBUG: "\x1b[37m",  # grey
        logging.INFO: "\x1b[0m",  # no colour
        logging.WARNING: "\x1b[33m",  # yellow
        logging.ERROR: "\x1b[35m",  # purple
        logging.CRITICAL: "\x1b[4;31m",  # red underline
    }
    reset: ClassVar[str] = "\x1b[0m"

    @override
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
    level: str = "info",
    colour: bool = True,
) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.setLevel(logging.getLevelName(level.upper()))

    fmt = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    file_handler = logging.FileHandler(output_dir / file_name, mode=mode)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(file_handler)

    fmt_cls = ColourFormatter if colour else logging.Formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt_cls(fmt=fmt, datefmt=datefmt))
    logger.addHandler(console_handler)


def backup_and_write(file: Path, txt: str) -> None:
    """Back up a text file with the timestamp, then overwrite the original."""
    if file.exists():
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_file = file.with_suffix(f".{ts}{file.suffix}")
        shutil.copy(file, backup_file)
        logger.info(f"Backed up {file} to {backup_file}")
    file.write_text(txt)
