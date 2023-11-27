import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import typer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationMixin


@dataclass
class Config:
    num_beams: int = 1
    max_seq_length: int = 256
    max_samples: int | None = None
    model_1_data_path: str | None = None
    model_2_data_path: str | None = None

    @classmethod
    def from_json(cls, json_data: str) -> "Config":
        "Create instance from dict, ignoring unknown fields."
        return cls(
            **{
                k: v
                for k, v in json.loads(json_data).items()
                if k in inspect.signature(cls).parameters
            }
        )


def generate(
    model: GenerationMixin, input_ids: torch.Tensor, config: Config
) -> torch.Tensor:
    return model.generate(
        input_ids, num_beams=config.num_beams, max_length=config.max_seq_length
    )


def read_json_dataset(
    path: Path, key: str | None = None, max_n: int | None = None
) -> list[dict[str, str]]:
    data = json.loads(path.read_text())
    if key is not None:
        data = data[key]
    return data[:max_n]


def main(
    model_1_path: Path = typer.Argument(..., help="Path to the first model"),
    model_2_path: Path = typer.Argument(..., help="Path to the second model"),
    config_path: Path = typer.Argument(..., help="Path to the configuration JSON file"),
    model_1_data_path: Optional[Path] = typer.Argument(
        None, help="JSON file with data for the first model"
    ),
    model_2_data_path: Optional[Path] = typer.Argument(
        None, help="JSON file with data for the second model"
    ),
) -> None:
    """
    Generate output texts using two seq2seq models in a pipeline.
    """
    tokenizer_1 = AutoTokenizer.from_pretrained(model_1_path)
    model_1 = AutoModelForSeq2SeqLM.from_pretrained(model_1_path)
    tokenizer_2 = AutoTokenizer.from_pretrained(model_2_path)
    model_2 = AutoModelForSeq2SeqLM.from_pretrained(model_2_path)

    config = Config.from_json(config_path.read_text())

    model_1_data_path = model_1_data_path or (
        Path(config.model_1_data_path) if config.model_1_data_path else None
    )
    if model_1_data_path is None:
        raise ValueError("You must specify the path for the first model's data")

    model_2_data_path = model_2_data_path or (
        Path(config.model_2_data_path) if config.model_2_data_path else None
    )
    if model_2_data_path is None:
        raise ValueError("You must specify the path for the second model's data")

    model_1_data = read_json_dataset(model_1_data_path, "data", config.max_samples)
    model_2_data = read_json_dataset(model_2_data_path, "data", config.max_samples)

    input_texts = [entry["context"] for entry in model_1_data]
    expected_outputs = [entry["answers"] for entry in model_2_data]

    input_ids_1 = tokenizer_1(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).input_ids
    output_ids_1 = generate(model_1, input_ids_1, config)[:, 1:]

    output_ids_2 = generate(model_2, output_ids_1, config)
    output_texts = tokenizer_2.batch_decode(output_ids_2, skip_special_tokens=True)

    for input_text, expected_output, output_text in zip(
        input_texts, expected_outputs, output_texts
    ):
        print("INPUT TEXT:")
        print(input_text)
        print("=" * 80)
        print("EXPECTED OUTPUT:")
        print(expected_output)
        print("=" * 80)
        print("OUTPUT TEXT:")
        print(output_text)
        print()


if __name__ == "__main__":
    typer.run(main)
