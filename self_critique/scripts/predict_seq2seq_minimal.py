import argparse
import inspect
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

sys.path.append(str(Path(__file__).parents[1]))
from metric import FGCRCls  # noqa: E402


@dataclass
class Config:
    model_name_or_path: str
    max_seq_length: int = 256
    generation_num_beams: int | None = None
    per_device_test_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    per_device_train_batch_size: int = 32
    device: str = "cuda:0"
    num_train_epochs: int = 20
    learning_rate: float = 5e-4
    output_dir: Path | None = None
    train_file: Path | None = None
    validation_file: Path | None = None
    test_file: Path | None = None
    do_predict: bool = True
    do_train: bool = True
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    max_predict_samples: int | None = None

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> Self:
        "Create instance from dict, ignoring unknown fields."

        def convert(key: str, value: Any) -> Any:
            if any(sub in key for sub in ["file", "path", "dir"]):
                return Path(value)
            return value

        return cls(
            **{
                k: convert(k, v)
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def log_metrics(metrics: dict[str, float], desc: str) -> None:
    logging.info(f">>>> {desc.upper()}")

    padding = max(len(k) for k in metrics)
    for k, v in metrics.items():
        logging.info(f"  {k:>{padding}}: {v}")


def format_input(d: dict[str, str]) -> str:
    context = d["context"].lstrip()
    question = d["question"].lstrip()
    return f"{context}\n{question}"


def preprocess_data(
    tokeniser,
    data: list[dict[str, str]],
    config: Config,
    shuffle: bool,
    batch_size: int,
) -> DataLoader:
    source_texts = [format_input(d) for d in data]
    target_texts = [d["answers"] for d in data]

    input_ids = tokeniser(
        source_texts,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_seq_length,
    ).input_ids.to(config.device)
    labels = tokeniser(
        target_texts,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_seq_length,
    ).input_ids.to(config.device)

    dataset = TensorDataset(input_ids, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def pad_labels(
    label_batch: torch.Tensor, tokeniser: PreTrainedTokenizer
) -> torch.Tensor:
    return torch.cat(
        [
            label_batch[:, :-1],
            tokeniser.pad_token_id
            * torch.ones((label_batch.shape[0], 1), dtype=torch.long),
        ],
        dim=-1,
    )


@dataclass
class EvalResult:
    loss: float
    metrics: dict[str, float]
    predictions: list[str]


def do_eval(
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizer,
    epoch: int,
    loader: DataLoader,
    data: list[dict[str, str]],
    desc: str | None = None,
) -> None:
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokeniser.pad_token_id)
    loss = 0
    num_batches = 0
    all_predictions: list[str] = []

    with torch.no_grad():
        for input_batch, label_batch in tqdm(loader, desc=desc):
            decoder_input_ids = pad_labels(label_batch, tokeniser)
            outputs = model(input_batch, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits

            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_texts = tokeniser.batch_decode(
                predicted_ids, skip_special_tokens=True
            )
            metrics = calculate_metrics(data, predicted_texts)
            log_metrics(metrics, desc)

            loss = criterion(
                logits.view(-1, logits.shape[-1]),
                label_batch.view(-1),
            )

            all_predictions.extend(predicted_texts)
            loss += loss.item()
            num_batches += 1

    avg_loss = loss / num_batches
    return EvalResult(
        loss=avg_loss,
        metrics=metrics,
        predictions=all_predictions,
    )


def do_train(
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizer,
    train_data: list[dict[str, str]],
    eval_data: list[dict[str, str]] | None,
    config: Config,
) -> PreTrainedModel:
    train_data = train_data[: config.max_train_samples]
    train_loader = preprocess_data(
        tokeniser,
        train_data,
        config,
        shuffle=True,
        batch_size=config.per_device_train_batch_size,
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokeniser.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    model.train()

    eval_loader = None
    if eval_data:
        eval_data = eval_data[: config.max_eval_samples]
        eval_loader = preprocess_data(
            tokeniser,
            eval_data,
            config,
            shuffle=False,
            batch_size=config.per_device_eval_batch_size,
        )

    for epoch in range(config.num_train_epochs):
        model.train()

        total_loss = 0
        num_batches = 0

        for input_batch, label_batch in tqdm(
            train_loader, desc=f"Epoch {epoch+1} training"
        ):
            decoder_input_ids = pad_labels(label_batch, tokeniser)
            outputs = model(input_batch, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits

            loss = criterion(logits.view(-1, logits.shape[-1]), label_batch.view(-1))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        logging.info(f"Epoch {epoch+1}, loss: {avg_loss}")

        if eval_data and eval_loader is not None:
            eval_result = do_eval(
                model,
                tokeniser,
                epoch,
                eval_loader,
                eval_data,
                desc=f"Epoch {epoch+1} evaluation",
            )
            logging.info(f"Epoch {epoch+1}, evaluation loss: {eval_result.loss}")

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
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizer,
    data: list[dict[str, str]],
    config: Config,
) -> tuple[list[dict[str, str]], dict[str, float]]:
    """Perform prediction on data.

    `model` must be a Seq2Seq model and compatible with `tokeniser`.
    """
    logging.info("*** Predicting ***")
    model.eval()

    logging.info("Tokenising input")

    data = data[: config.max_predict_samples]
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
    dataset = TensorDataset(input_ids)
    loader = DataLoader(dataset, batch_size=batch_size)
    for input_batch in tqdm(loader, desc="Predicting"):
        batch_predicted_ids = model.generate(
            input_batch[0].to(config.device),
            num_beams=num_beams,
            max_length=config.max_seq_length,
        )
        predicted_ids.extend(batch_predicted_ids)

    logging.info("Decoding output")
    predicted_texts = tokeniser.batch_decode(predicted_ids, skip_special_tokens=True)

    logging.info("Calculating metrics")
    metrics = calculate_metrics(data, predicted_texts)
    log_metrics(metrics, "Prediction")
    output = [
        {"input": inp, "output": out, "gold": d["answers"]}
        for inp, out, d in zip(input_texts, predicted_texts, data)
    ]
    return output, metrics


def log_config(config: Config) -> None:
    logging.info(">>>> CONFIGURATON")
    for key, value in asdict(config).items():
        logging.info(f"{key}: {value}")


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path, help="Path to config file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = Config.from_dict(json.loads(args.config_path.read_text()))
    log_config(config)

    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path).to(
        config.device
    )
    tokeniser = AutoTokenizer.from_pretrained(
        config.model_name_or_path, model_max_length=config.max_seq_length
    )

    if config.do_train:
        if config.train_file is None:
            raise ValueError("train_data_path must be specified when training")

        train_data = json.loads(config.train_file.read_text())["data"]
        eval_data = (
            json.loads(config.validation_file.read_text())["data"]
            if config.validation_file is not None
            else None
        )

        model = do_train(model, tokeniser, train_data, eval_data, config)
        if config.output_dir is not None:
            logging.info("Saving model to: %s", config.output_dir.resolve())
            model.save_pretrained(config.output_dir)
            tokeniser.save_pretrained(config.output_dir)

    if config.do_predict:
        predict_data = json.loads(config.test_file.read_text())["data"]
        output, metrics = do_predict(model, tokeniser, predict_data, config)
        if config.output_dir is not None:
            config.output_dir.mkdir(exist_ok=True, parents=True)
            logging.info("Saving results to: %s", config.output_dir.resolve())
            (config.output_dir / "predict_output.json").write_text(json.dumps(output))
            (config.output_dir / "predict_metrics.json").write_text(json.dumps(metrics))


if __name__ == "__main__":
    main()
