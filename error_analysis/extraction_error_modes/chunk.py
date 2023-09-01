import json
from pathlib import Path
from typing import Any

import typer


def generate_chunks(
    data: list[Any],
    chunk_size: int,
) -> list[tuple[int, int, list[Any]]]:
    return [
        (i, min(i + chunk_size, len(data)), data[i : i + chunk_size])
        for i in range(0, len(data), chunk_size)
    ]


def save_chunks(
    input_file_path: Path, output_dir: Path, chunks: list[tuple[int, int, list[Any]]]
) -> None:
    base_name = input_file_path.stem

    for start, end, chunk in chunks:
        filename = f"{base_name}.{start}-{end - 1}.json"
        (output_dir / filename).write_text(json.dumps(chunk, indent=2))

    print(f"Successfully chunked '{input_file_path}' into {len(chunks)} files.")


def main(input_file: Path, output_dir: Path, chunk_size: int) -> None:
    data = json.loads(input_file.read_text())
    chunks = generate_chunks(data, chunk_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_chunks(input_file, output_dir, chunks)


if __name__ == "__main__":
    typer.run(main)
