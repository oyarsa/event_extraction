import json
import random
from pathlib import Path

import typer


def main(input_dir: Path, output_dir: Path, splits: str) -> None:
    train_pct, dev_pct, _test_pct = (int(x) / 100 for x in splits.split("/"))

    data: list[dict[str, str]] = []
    for file in input_dir.glob("*.json"):
        data.extend(json.loads(file.read_text()))

    random.shuffle(data)

    size = len(data)
    train_end = int(train_pct * size)
    dev_end = int(train_end + dev_pct * size)

    splits_data = {
        "train": data[:train_end],
        "dev": data[train_end:dev_end],
        "test": data[dev_end:],
    }

    print(f"{len(splits_data['train'])=}")
    print(f"{len(splits_data['dev'])=}")
    print(f"{len(splits_data['test'])=}")

    output_dir.mkdir(exist_ok=True, parents=True)
    for split_name, split_data in splits_data.items():
        path = output_dir / f"{split_name}.json"
        path.write_text(json.dumps(split_data))


if __name__ == "__main__":
    typer.run(main)
