import json
import random
from pathlib import Path

import typer


def main(data_file: Path, output_dir: Path, percentage: float, seed: int = 0) -> None:
    random.seed(seed)

    data = json.loads(data_file.read_text())
    random.shuffle(data)

    split_index = int(len(data) * percentage)
    to_label = data[:split_index]
    unlabelled = data[split_index:]

    print("To label:", len(to_label))
    print("Unlabelled:", len(unlabelled))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "unlabelled.json").write_text(json.dumps(unlabelled))
    (output_dir / "to_label.json").write_text(json.dumps(to_label))


if __name__ == "__main__":
    typer.run(main)
