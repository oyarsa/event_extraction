import subprocess
from importlib import resources
from pathlib import Path

import torch
import torch.backends.mps


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


def get_current_commit_shorthash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        return "unknown"
