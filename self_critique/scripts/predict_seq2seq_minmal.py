import inspect
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Self

import torch
import typer
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append(str(Path(__file__).parents[1]))
from metric import FGCRCls  # noqa: E402


@dataclass
class Config:
    max_seq_length: int = 256
    generation_num_beams: int | None = None
    per_device_test_batch_size: int = 32
    device: str = "cuda:0"

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> Self:
        "Create instance from dict, ignoring unknown fields."
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def format_input(d: dict[str, str]) -> str:
    context = d["context"].lstrip()
    question = d["question"].lstrip()
    return f"{context}\n{question}"


def do_train(model, tokeniser, data: list[dict[str, str]], config: Config):
    "TODO: untested. Probably doesn't work."
    source_texts = [format_input(d) for d in data]
    target_texts = [d["answers"] for d in data]

    input_ids = tokeniser(
        source_texts,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_seq_length,
    ).input_ids
    labels = tokeniser(
        text_target=target_texts,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_seq_length,
    ).input_ids

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokeniser.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    model.train()
    # TODO: add batching
    # TODO: add metrics
    # TODO: add evaluation at the end of every epoch
    for i in range(len(input_ids)):
        outputs = model(input_ids[i].unsqueeze(0))
        logits = outputs.logits

        loss = criterion(logits.view(-1, logits.shape[-1]), labels[i])
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        logging.info(f"Step {i+1}, loss: {loss.item()}")

    return model


def calculate_metrics(
    data: list[dict[str, str]], predictions: list[str]
) -> dict[str, float]:
    references = [
        {
            "id": inp["id"],
            "answers": inp["answers"],
            "question_type": inp["question_type"],
        }
        for inp in data
    ]
    predictions = [
        {
            "id": inp["id"],
            "prediction_text": out,
        }
        for inp, out in zip(data, predictions)
    ]

    return FGCRCls()._compute(predictions=predictions, references=references)


def do_predict(
    model, tokeniser, data: list[dict[str, str]], config: Config
) -> tuple[list[dict[str, str]], dict[str, float]]:
    """Perform prediction on data.

    `model` must be a Seq2Seq model and compatible with `tokeniser`.
    """
    logging.info("*** Predicting ***")
    model.eval()

    logging.info("Tokenising input")

    input_texts = [format_input(d) for d in data]
    input_ids = tokeniser(
        input_texts,
        return_tensors="pt",
        padding=True,
        max_length=config.max_seq_length,
        truncation=True,
    ).input_ids.to(config.device)

    logging.info("Generating output")

    batch_size = config.per_device_test_batch_size
    num_beams = config.generation_num_beams or model.config.num_beams

    predicted_ids: list[torch.Tensor] = []
    for i in tqdm(range(0, len(input_ids), batch_size)):
        batch_input_ids = input_ids[i : i + batch_size]
        batch_predicted_ids = model.generate(
            batch_input_ids,
            num_beams=num_beams,
            max_length=config.max_seq_length,
        )
        predicted_ids.extend(batch_predicted_ids)

    logging.info("Decoding output")
    predicted_texts = tokeniser.batch_decode(predicted_ids, skip_special_tokens=True)

    logging.info("Calculating metrics")
    metrics = calculate_metrics(data, predicted_texts)
    output = [
        {"input": inp, "output": out} for inp, out in zip(input_texts, predicted_texts)
    ]
    return output, metrics


def main(
    model_path: Path,
    config_path: Path,
    data_path: Path,
    output_dir: Optional[Path],
    train: bool = False,
    predict: bool = True,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    config = Config.from_dict(json.loads(config_path.read_text()))
    data = json.loads(data_path.read_text())["data"]

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(config.device)
    tokeniser = AutoTokenizer.from_pretrained(model_path)

    if train:
        model = do_train(model, tokeniser, data, config)

    if predict:
        output, metrics = do_predict(model, tokeniser, data, config)
        if output_dir is not None:
            output_dir.mkdir(exist_ok=True, parents=True)
            logging.info("Saving results to: %s", output_dir.resolve())
            (output_dir / "predict_output.json").write_text(json.dumps(output))
            (output_dir / "predict_metrics.json").write_text(json.dumps(metrics))


if __name__ == "__main__":
    typer.run(main)
