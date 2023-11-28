import json
import statistics
import sys

from transformers import AutoTokenizer


def load_json(file_path: str) -> list[dict]:
    """Load JSON data from a file."""
    with open(file_path) as file:
        data = json.load(file)
    return data


def count_tokens(data: list[dict], tokenizer: AutoTokenizer) -> dict[str, float]:
    """Count tokens in the 'output' and 'answer' fields and compute average."""
    tokens: list[int] = []
    for item in data:
        output_tokens = tokenizer.tokenize(item.get("output", ""))
        gold_tokens = tokenizer.tokenize(item.get("gold", ""))
        tokens.append(len(output_tokens) + len(gold_tokens))

    return {
        "max": max(tokens),
        "min": min(tokens),
        "mean": statistics.mean(tokens),
        "median": statistics.median(tokens),
    }


def main(json_file_path: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")

    # Load JSON data
    data = load_json(json_file_path)

    # Compute the average number of tokens
    avg_tokens = count_tokens(data, tokenizer)
    print(json.dumps(avg_tokens, indent=2))


# Example usage
if __name__ == "__main__":
    json_file_path = sys.argv[1]
    main(json_file_path)
