import argparse
import hashlib
import json
from pathlib import Path

from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chatgpt-output", type=Path, required=True)
    parser.add_argument("--chatgpt-input", type=Path, required=True)
    parser.add_argument("--original-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    chat_output = json.loads(args.chatgpt_output.read_text())
    chat_input = json.loads(args.chatgpt_input.read_text())["data"]
    original_data = json.loads(args.original_data.read_text())["data"]

    final: list[dict[str, str]] = []
    for input, output in tqdm(zip(chat_input, chat_output), total=len(chat_input)):
        for original in original_data:
            if input["answers"] in original["context"]:
                inst = {
                    "sentence1": original["context"],
                    "sentence2": output,
                    "label": "CONTRADICTION",
                }
                inst["id"] = hashlib.sha1(str(inst).encode("utf-8")).hexdigest()[:8]
                final.append(inst)
                break

    args.output.write_text(json.dumps(final))


if __name__ == "__main__":
    main()
