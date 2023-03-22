import json
from pathlib import Path
from typing import Any


class ExchangeLogger:
    def __init__(self) -> None:
        self.file: Path | None = None
        self.print_log = False

    def config(self, file: Path, print_log: bool) -> None:
        self.file = file
        self.print_log = print_log

    def log_exchange(self, params: dict[str, Any], response: dict[str, Any]) -> None:
        assert self.file is not None, "Must call config() before logging exchanges."

        log = {"params": params, "response": response}

        with self.file.open("a") as f:
            json.dump(log, f)
            f.write("\n")

        if self.print_log:
            print(json.dumps(log, indent=2))
            print()


logger = ExchangeLogger()
