from importlib import resources
from pathlib import Path


def get_root(module: str) -> str:
    files = resources.files(module)
    with resources.as_file(files) as path:
        return path.parent.resolve()


PROJECT_ROOT = get_root("self_critique")


def resolve_path(path: str | Path) -> str:
    return str(PROJECT_ROOT / path)
