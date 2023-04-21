# pyright: basic
import json
import logging
import os
import random
import sys
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import simple_parsing
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.append(str(Path(__file__).parents[1]))
import metric  # noqa: E402


@dataclass
class Config:
    """Configuration for the model training, evaluation and prediction."""

    # Model name from the HuggingFace model hub, or path to a local model saved
    # with `model.save_pretrained`.
    model_name_or_path: str
    # Directory to save the model, tokeniser, metrics and predictions
    output_dir: Path
    # Maximum length of the input sequence
    max_seq_length: int = 256
    # Number of beams to use for beam search
    generation_num_beams: int | None = None
    # Batch size for training
    per_device_train_batch_size: int = 32
    # Batch size for evaluation
    per_device_eval_batch_size: int = 32
    # Batch size for prediction
    per_device_test_batch_size: int = 32
    # Device to use
    device: str = "cuda:0"
    # Number of training epochs
    num_train_epochs: int = 20
    # Learning rate
    learning_rate: float = 5e-4
    # Path to the training data
    train_file: Path | None = None
    # Path to the validation data
    validation_file: Path | None = None
    # Path to the test data
    test_file: Path | None = None
    # Whether to run prediction at the end
    do_predict: bool = True
    # Whether to run evaluation
    do_eval: bool = True
    # Whether to run training
    do_train: bool = True
    # Maximum number of samples used for training
    max_train_samples: int | None = None
    # Maximum number of samples used for evaluation
    max_eval_samples: int | None = None
    # Maximum number of samples used for prediction
    max_predict_samples: int | None = None
    # Level for logging. Most messages are INFO.
    log_level: str = "info"
    # Load the best model by token F1 at the end of training
    load_best_model_at_end: bool = True
    # Early stopping patience
    early_stopping_patience: int = 5
    # Random seed for reproducibility
    seed: int = 0

    def __init__(self, **kwargs: Any) -> None:
        "Ignore unknown arguments"
        for f in fields(self):
            if f.name in kwargs:
                setattr(self, f.name, kwargs[f.name])


@dataclass
class Seq2SeqEntry:
    id: str
    context: str
    question: str
    answers: str
    question_type: str


def load_data(file_path: Path) -> list[Seq2SeqEntry]:
    data = json.loads(file_path.read_text())
    if "data" in data:
        data = data["data"]
    return [Seq2SeqEntry(**d) for d in data]


def log_metrics(metrics: dict[str, float], desc: str | None) -> None:
    desc = desc or "metrics"
    logging.info(f">>>> {desc.upper()}")

    padding = max(len(k) for k in metrics)
    for k, v in metrics.items():
        logging.info(f"    {k:>{padding}}: {v}")


def preprocess_data(
    tokeniser: PreTrainedTokenizer,
    data: list[Seq2SeqEntry],
    config: Config,
    shuffle: bool,
    batch_size: int,
    desc: str | None = None,
) -> DataLoader:
    desc = desc or ""
    logging.info(f"Preprocessing {desc} data")
    source_texts = [f"{d.question.lstrip()}\n{d.context.lstrip()}" for d in data]
    target_texts = [d.answers for d in data]

    model_inputs = tokeniser(
        source_texts,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        max_length=config.max_seq_length,
    )
    labels = tokeniser(
        text_target=target_texts,
        padding="max_length",
        return_tensors="pt",
        max_length=config.max_seq_length,
        truncation=True,
    )

    dataset = Seq2SeqDataset(
        input_tokens=model_inputs,
        target_tokens=labels,
        data=data,
        device=config.device,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


class Seq2SeqDatasetEntry(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    id: str
    answers: str
    question_type: str
    context: str


class Seq2SeqDatasetSeries(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    id: list[str]
    answers: list[str]
    question_type: list[str]
    context: list[str]


@dataclass
class Seq2SeqDataset(Dataset):
    input_tokens: Mapping[str, torch.Tensor]
    target_tokens: Mapping[str, torch.Tensor]
    data: list[Seq2SeqEntry]
    device: str

    def __len__(self) -> int:
        return self.input_tokens["input_ids"].size(0)

    def __getitem__(self, idx: int) -> Seq2SeqDatasetEntry:
        return {
            "input_ids": self.input_tokens["input_ids"][idx].to(self.device),
            "attention_mask": self.input_tokens["attention_mask"][idx].to(self.device),
            "labels": self.target_tokens["input_ids"][idx].to(self.device),
            "id": self.data[idx].id,
            "answers": self.data[idx].answers,
            "question_type": self.data[idx].question_type,
            "context": self.data[idx].context,
        }


@dataclass
class EvalResult:
    loss: float
    metrics: dict[str, float]
    predictions: list[str]


def eval(
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizer,
    loader: DataLoader,
    desc: str | None = None,
) -> EvalResult:
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokeniser.pad_token_id)
    total_loss = 0
    num_batches = 0

    all_predictions: list[torch.Tensor] = []
    all_data: list[Seq2SeqDatasetSeries] = []
    with torch.no_grad():
        for inputs in tqdm(loader, desc=desc):
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )

            # CrossEntropy wants [batch * seq_len, num_classes] and [batch * seq_len]
            loss = criterion(
                # [batch, seq_len, vocab_size] -> [batch * seq_len, vocab_size]
                outputs.logits.reshape(-1, outputs.logits.size(-1)),
                # [batch, seq_len] -> [batch * seq_len]
                inputs["labels"].reshape(-1),
            )
            total_loss += loss.item()

            predicted_ids = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predicted_ids)
            all_data.append(inputs)

            num_batches += 1

    logging.info("Decoding output")
    predicted_texts = tokeniser.batch_decode(all_predictions, skip_special_tokens=True)

    model_data = collect_model_data(all_data)
    logging.info("Calculating metrics")
    metrics = calculate_metrics(model_data, predicted_texts)
    log_metrics(metrics, desc)
    avg_loss = total_loss / num_batches
    return EvalResult(
        loss=avg_loss,
        metrics=metrics,
        predictions=predicted_texts,
    )


def collect_model_data(
    all_data: list[Seq2SeqDatasetSeries],
) -> list[Seq2SeqDatasetEntry]:
    model_data: list[Seq2SeqDatasetEntry] = []
    for d in all_data:
        for i in range(len(d["id"])):
            entry = {k: d[k][i] for k in d}
            model_data.append(cast(Seq2SeqDatasetEntry, entry))
    return model_data


def train(
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizer,
    train_data: list[Seq2SeqEntry],
    eval_data: list[Seq2SeqEntry] | None,
    config: Config,
) -> PreTrainedModel:
    train_data = train_data[: config.max_train_samples]
    train_loader = preprocess_data(
        tokeniser,
        train_data,
        config,
        shuffle=True,
        batch_size=config.per_device_train_batch_size,
        desc="training",
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokeniser.pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 0, config.num_train_epochs * len(train_loader)
    )

    model.train()

    eval_loader = None
    if eval_data:
        eval_data = eval_data[: config.max_eval_samples]
        eval_loader = preprocess_data(
            tokeniser,
            eval_data,
            config,
            shuffle=True,
            batch_size=config.per_device_eval_batch_size,
            desc="evaluation",
        )

    best_f1 = -1.0
    early_stopping_counter = 0

    for epoch in range(config.num_train_epochs):
        model.train()

        total_loss = 0
        num_batches = 0

        for inputs in tqdm(train_loader, desc=f"Epoch {epoch+1} training"):
            optimizer.zero_grad()

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
            logits = outputs.logits

            loss = criterion(
                logits.view(-1, logits.shape[-1]), inputs["labels"].view(-1)
            )
            loss.backward()

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        logging.info(f"Epoch {epoch+1}, training loss: {avg_loss}")

        if eval_data and eval_loader is not None:
            eval_result = eval(
                model,
                tokeniser,
                eval_loader,
                desc=f"Epoch {epoch+1} evaluation",
            )
            logging.info(f"Epoch {epoch+1}, evaluation loss: {eval_result.loss}")

            if eval_result.metrics["f1"] > best_f1:
                best_f1 = eval_result.metrics["f1"]
                early_stopping_counter = 0

                logging.info(
                    "New best model! Saving to: %s", config.output_dir.resolve()
                )
                save_model(model, tokeniser, config.output_dir)
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= config.early_stopping_patience:
                logging.info(
                    f"Early stopping: {early_stopping_counter} epochs without improvement"
                )
                break

    # Either we're not saving based on eval f1, or we're at the end of training
    # and we haven't saved yet
    if best_f1 == -1:
        save_model(model, tokeniser, config.output_dir)

    return model


def save_model(
    model: PreTrainedModel, tokeniser: PreTrainedTokenizer, output_dir: Path
) -> None:
    model.config.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    tokeniser.save_pretrained(output_dir)


def calculate_metrics(
    data: list[Seq2SeqDatasetEntry], output: list[str]
) -> dict[str, float]:
    references: list[metric.MetricReference] = [
        {
            "id": entry["id"],
            "answers": entry["answers"],
            "question_type": entry["question_type"],
        }
        for entry in data
    ]
    predictions: list[metric.MetricPrediction] = [
        {
            "id": entry["id"],
            "prediction_text": out,
        }
        for entry, out in zip(data, output)
    ]

    return metric.FGCRCls()._compute(predictions=predictions, references=references)


@dataclass
class InferenceResult:
    predictions: list[dict[str, str]]
    metrics: dict[str, float]


def infer(
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizer,
    data: list[Seq2SeqEntry],
    config: Config,
    desc: str,
    max_samples: int | None = None,
) -> InferenceResult:
    """Perform prediction on data.

    `model` must be a Seq2Seq model and compatible with `tokeniser`.
    """
    desc = desc.capitalize()
    logging.info("*** %s ***", desc)
    model.eval()

    data = data[:max_samples]
    logging.info("%d samples", len(data))
    loader = preprocess_data(
        tokeniser,
        data,
        config,
        shuffle=False,
        batch_size=config.per_device_train_batch_size,
        desc=desc,
    )

    predicted_ids: list[torch.Tensor] = []
    all_data: list[Seq2SeqDatasetSeries] = []

    for inputs in tqdm(loader, desc=desc):
        batch_predicted_ids = model.generate(
            inputs["input_ids"],
            num_beams=config.generation_num_beams or model.config.num_beams,
            max_length=config.max_seq_length,
        )
        predicted_ids.extend(batch_predicted_ids)
        all_data.append(inputs)

    predicted_texts = tokeniser.batch_decode(predicted_ids, skip_special_tokens=True)

    model_data = collect_model_data(all_data)
    metrics = calculate_metrics(model_data, predicted_texts)
    log_metrics(metrics, desc)

    output = [
        {"input": d["context"], "output": out, "gold": d["answers"]}
        for out, d in zip(predicted_texts, model_data)
    ]
    return InferenceResult(predictions=output, metrics=metrics)


def log_config(config: Config) -> None:
    logging.info(">>>> CONFIGURATON")
    for key, value in asdict(config).items():
        logging.info(f"  {key}: {value}")


def load_model(
    model_name_or_path: str | Path, max_seq_length: int, device: str
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    logging.info("Loading model from %s", model_name_or_path)

    model_config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path, config=model_config
    ).to(device)
    tokeniser = AutoTokenizer.from_pretrained(
        model_name_or_path, model_max_length=max_seq_length
    )
    model.resize_token_embeddings(len(tokeniser))
    return model, tokeniser


def set_seed(seed: int) -> None:
    "Set random seed for reproducibility."
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_results(desc: str, output_dir: Path, result: InferenceResult) -> None:
    desc = desc.lower()
    logging.info("Saving %s results to: %s", desc, output_dir.resolve())
    (output_dir / f"{desc}_output.json").write_text(json.dumps(result.predictions))
    (output_dir / f"{desc}_metrics.json").write_text(json.dumps(result.metrics))


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = simple_parsing.parse(Config, add_config_path_arg=True)
    set_seed(config.seed)

    logging.basicConfig(
        level=logging.getLevelName(config.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_config(config)

    warnings.filterwarnings("ignore", module="transformers.convert_slow_tokenizer")
    transformers.logging.set_verbosity_error()

    config.output_dir.mkdir(exist_ok=True, parents=True)

    model, tokeniser = load_model(
        config.model_name_or_path, config.max_seq_length, config.device
    )

    train_data, eval_data, predict_data = None, None, None
    if config.train_file is not None:
        train_data = load_data(config.train_file)
    if config.validation_file is not None:
        eval_data = load_data(config.validation_file)
    if config.test_file is not None:
        predict_data = load_data(config.test_file)

    if config.do_train:
        if train_data is None:
            raise ValueError("train_file must be specified when training")
        model = train(model, tokeniser, train_data, eval_data, config)
        if config.load_best_model_at_end:
            logging.info("Loading best model from %s", config.output_dir.resolve())
            model, tokeniser = load_model(
                config.output_dir, config.max_seq_length, config.device
            )

    if config.do_eval:
        if eval_data is None:
            raise ValueError(
                "validation_file must be specified when evaluating training"
            )
        result = infer(
            model, tokeniser, eval_data, config, "evaluation", config.max_eval_samples
        )
        save_results("evaluation", config.output_dir, result)

    if config.do_predict:
        if predict_data is None:
            raise ValueError("test_file must be specified when training")
        result = infer(
            model,
            tokeniser,
            predict_data,
            config,
            "prediction",
            config.max_predict_samples,
        )
        save_results("prediction", config.output_dir, result)


if __name__ == "__main__":
    main()
