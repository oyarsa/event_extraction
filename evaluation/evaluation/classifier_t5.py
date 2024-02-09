import collections
import copy
import dataclasses
import json
import logging
import os
import random
import sys
import warnings
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import simple_parsing
import torch
import torch.backends.mps
import transformers
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger("classifier_t5")


@dataclasses.dataclass
class Config:
    # Classifier model name or path
    model_name: str
    # Path to data. Will be used for both train and dev.
    train_data_path: Path
    # Evaluation data
    eval_data_path: Path
    # Learning rate for AdamW optimizer
    learning_rate: float
    # Number of epochs to train
    num_epochs: int
    # Maximum length for model output (tokens)
    max_seq_length: int
    # Output directory for model Batch size
    batch_size: int
    # Maximum number of epochs without improvement
    early_stopping_patience: int
    # Number of samples to use from data
    max_samples: int | None = None
    # Path to the directory where the output will be saved
    output_path: Path = Path("output")
    # Random seed for reproducibility
    seed: int = 0
    # Name of the output directory. If unspecified, will be generated the current
    # ISO timestamp.
    output_name: str | None = None
    # Do train
    do_train: bool = True
    # Do test
    do_test: bool = False
    # Do inference
    do_inference: bool = False
    # Do evaluation
    do_evaluation: bool = False
    # Test data
    test_data_path: Path | None = None
    # Inference data
    infer_data_path: Path | None = None
    # Dropout percentage for transformer model
    dropout: float | None = None

    def __init__(self, **kwargs: Any) -> None:
        "Ignore unknown arguments"
        for f in dataclasses.fields(self):
            if f.name in kwargs:
                setattr(self, f.name, kwargs[f.name])

    def __str__(self) -> str:
        config_lines = [">>>> CONFIGURATION"]
        for key, val in dataclasses.asdict(self).items():
            value = val.resolve() if isinstance(val, Path) else val
            config_lines.append(f"  {key}: {value}")
        return "\n".join(config_lines)


def init_model(
    model_name: str, max_seq_length: int, dropout: float | None
) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    config = AutoConfig.from_pretrained(model_name)
    if dropout is not None:
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=max_seq_length
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config
    )
    return tokenizer, model


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    desc: str,
) -> float:
    total_loss = 0

    model.train()
    for batch in tqdm(train_loader, desc=desc):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


@dataclasses.dataclass
class EvaluationResult:
    golds: list[int]
    preds: list[int]
    passages: list[str]
    outputs: list[str]
    annotations: list[str]
    loss: float


def evaluate(
    model: PreTrainedModel,
    val_loader: DataLoader,
    device: torch.device,
    desc: str,
) -> EvaluationResult:
    preds: list[int] = []
    golds: list[int] = []
    passages: list[str] = []
    outputs: list[str] = []
    annotations: list[str] = []
    loss: float = 0

    model.eval()
    for batch in tqdm(val_loader, desc=desc):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            model_outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            preds.extend(torch.argmax(model_outputs.logits, dim=1).tolist())
            golds.extend(labels.tolist())
            loss += model_outputs.loss.item()

            passages.extend(batch["input"])
            outputs.extend(batch["output"])
            annotations.extend(batch["gold"])

    return EvaluationResult(
        golds=golds,
        preds=preds,
        passages=passages,
        outputs=outputs,
        annotations=annotations,
        loss=loss / len(val_loader),
    )


@dataclasses.dataclass
class InferenceResult:
    preds: list[int]
    passages: list[str]
    outputs: list[str]
    annotations: list[str]


def infer(
    model: PreTrainedModel,
    val_loader: DataLoader,
    device: torch.device,
    desc: str,
) -> InferenceResult:
    preds: list[int] = []
    passages: list[str] = []
    outputs: list[str] = []
    annotations: list[str] = []

    model.eval()
    for batch in tqdm(val_loader, desc=desc):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            model_outputs = model(input_ids, attention_mask=attention_mask)

            preds.extend(torch.argmax(model_outputs.logits, dim=1).tolist())

            passages.extend(batch["input"])
            outputs.extend(batch["output"])
            annotations.extend(batch["gold"])

    return InferenceResult(
        preds=preds,
        passages=passages,
        outputs=outputs,
        annotations=annotations,
    )


def calc_metrics(results: EvaluationResult) -> dict[str, float]:
    acc = accuracy_score(results.golds, results.preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        results.golds, results.preds, average="binary", zero_division=0  # type: ignore
    )

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "eval_loss": results.loss,
    }


def report_metrics(metrics: dict[str, float], desc: str) -> None:
    logger.info(
        f"{desc} results\n"
        f"    Accuracy      : {metrics['accuracy']:.4f}\n"
        f"    Precision     : {metrics['precision']:.4f}\n"
        f"    Recall        : {metrics['recall']:.4f}\n"
        f"    F1            : {metrics['f1']:.4f}\n"
        f"    Eval Loss     : {metrics['eval_loss']:.4f}\n"
    )


def train(
    model: PreTrainedModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    early_stopping_patience: int,
    metrics_file: Path,
) -> PreTrainedModel:
    model = model.to(device)  # type: ignore
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    best_f1 = 0
    best_model = copy.deepcopy(model)
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        avg_train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            desc=f"Training ({epoch + 1}/{num_epochs})",
        )
        logger.info(f"Epoch {epoch + 1}/{num_epochs} -> Loss: {avg_train_loss}")

        results = evaluate(
            model, val_loader, device, desc=f"Evaluation ({epoch + 1}/{num_epochs})"
        )

        metrics = calc_metrics(results)
        report_metrics(metrics, "Train evaluation")
        save_metrics(metrics_file, **metrics, train_loss=avg_train_loss)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = copy.deepcopy(model)
            logger.info(f"New best: {best_f1:.4f} F1")
        else:
            epochs_without_improvement += 1
            logger.info(f"{epochs_without_improvement} epochs without improvement")
            if epochs_without_improvement >= early_stopping_patience:
                logger.info("Early stopping")
                break

    return best_model


def save_metrics(metrics_file: Path, **kwargs: float) -> None:
    with metrics_file.open("a") as f:
        f.write(json.dumps(kwargs) + "\n")


def get_device() -> torch.device:
    "Returns MPS if available, CUDA if available, otherwise CPU device."
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return torch.device(device)


@dataclasses.dataclass
class DataEntry:
    input: str
    output: str
    gold: str
    valid: bool


@dataclasses.dataclass
class ClassifierDataset(Dataset):
    input_tokens: Mapping[str, torch.Tensor]
    labels: torch.Tensor | None
    data: list[DataEntry]

    def __len__(self) -> int:
        return self.input_tokens["input_ids"].size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        d = {
            "input_ids": self.input_tokens["input_ids"][idx],
            "attention_mask": self.input_tokens["attention_mask"][idx],
            "input": self.data[idx].input,
            "output": self.data[idx].output,
            "gold": self.data[idx].gold,
        }
        if self.labels is not None:
            d["labels"] = self.labels[idx]
        return d


def load_json(path: Path, n: int | None) -> list[DataEntry]:
    data = json.loads(path.read_text())
    assert isinstance(data, list), "JSON file should be a list of objects"
    assert isinstance(data[0], dict), "JSON list should contain objects"
    return [
        DataEntry(
            input=d["input"],
            output=d["output"],
            gold=d["gold"],
            valid=d["valid"],
        )
        for d in data[:n]
    ]


def preprocess_data(
    data: list[DataEntry],
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    batch_size: int,
    has_labels: bool,
) -> DataLoader:
    text = [
        f"is the extraction valid for the passage? extraction: {d.output}."
        f" passage: {d.input}"
        for d in data
    ]

    model_inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=max_seq_length,
    )
    if has_labels:
        labels = torch.tensor([int(d.valid) for d in data])
    else:
        labels = None

    dataset = ClassifierDataset(input_tokens=model_inputs, labels=labels, data=data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def save_model(
    model: PreTrainedModel, tokeniser: PreTrainedTokenizer, output_dir: Path
) -> None:
    model.config.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    tokeniser.save_pretrained(output_dir)


def setup_logger(output_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(output_dir / "train.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def suppress_transformers_warnings() -> None:
    "Remove annoying messages about tokenisers and unititialised weights."
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore", module="transformers.convert_slow_tokenizer")
    transformers.logging.set_verbosity_error()


def set_seed(seed: int) -> None:
    "Set random seed for reproducibility."
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_eval_results(
    results: EvaluationResult, metrics: dict[str, float], output_dir: Path, desc: str
) -> None:
    r = [
        {
            "gold": results.golds[i],
            "pred": results.preds[i],
            "passage": results.passages[i],
            "output": results.outputs[i],
            "annotation": results.annotations[i],
        }
        for i in range(len(results.golds))
    ]
    (output_dir / f"{desc}_results.json").write_text(json.dumps(r, indent=2))
    (output_dir / f"{desc}_metrics.json").write_text(json.dumps(metrics, indent=2))


def save_inference_results(
    results: InferenceResult, output_dir: Path, desc: str = "inference"
) -> None:
    r = [
        {
            "valid": bool(results.preds[i]),
            "input": results.passages[i],
            "output": results.outputs[i],
            "gold": results.annotations[i],
        }
        for i in range(len(results.preds))
    ]
    (output_dir / f"{desc}_results.json").write_text(json.dumps(r, indent=2))


def run_training(
    model: PreTrainedModel,
    config: Config,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    output_dir: Path,
) -> PreTrainedModel:
    logger.info(">>>> TRAINING <<<<")
    train_data = load_json(config.train_data_path, config.max_samples)
    train_loader = preprocess_data(
        train_data,
        tokenizer,
        config.max_seq_length,
        config.batch_size,
        has_labels=True,
    )

    eval_data = load_json(config.eval_data_path, config.max_samples)
    eval_loader = preprocess_data(
        eval_data,
        tokenizer,
        config.max_seq_length,
        config.batch_size,
        has_labels=True,
    )

    metrics_file = output_dir / "training_metrics.jsonl"
    metrics_file.unlink(missing_ok=True)
    metrics_file.touch()
    trained_model = train(
        model,
        train_loader,
        eval_loader,
        device,
        config.num_epochs,
        config.learning_rate,
        config.early_stopping_patience,
        metrics_file,
    )
    save_model(trained_model, tokenizer, output_dir)

    results = evaluate(trained_model, eval_loader, device, desc="Final evaluation")
    metrics = calc_metrics(results)
    report_metrics(metrics, "Final train evaluation")
    save_eval_results(results, metrics, output_dir, desc="eval")

    return trained_model


def report_distribution(preds: list[int], desc: str) -> None:
    c = collections.Counter(preds)
    logger.info(f"{desc.capitalize()} results:")
    logger.info(f"  Valid: {c[True]} ({c[True] / len(preds):.2%})")
    logger.info(f"  Invalid: {c[False]} ({c[False] / len(preds):.2%})")


def run_evaluation(
    model: PreTrainedModel,
    data_path: Path,
    config: Config,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    output_dir: Path,
    desc: str,
) -> None:
    model = model.to(device)  # type: ignore

    logger.info(f">>>> {desc.upper()} <<<<")

    data = load_json(data_path, config.max_samples)
    loader = preprocess_data(
        data,
        tokenizer,
        config.max_seq_length,
        config.batch_size,
        has_labels=True,
    )

    results = evaluate(model, loader, device, desc=f"{desc.capitalize()} evaluation")
    metrics = calc_metrics(results)
    report_metrics(metrics, desc)
    report_distribution(results.preds, desc)
    save_eval_results(results, metrics, output_dir, desc=desc)


def run_inference(
    model: PreTrainedModel,
    data_path: Path,
    config: Config,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    output_dir: Path,
) -> None:
    model = model.to(device)  # type: ignore

    logger.info(">>>> INFERENCE <<<<")

    data = load_json(data_path, config.max_samples)
    loader = preprocess_data(
        data,
        tokenizer,
        config.max_seq_length,
        config.batch_size,
        has_labels=False,
    )

    results = infer(model, loader, device, desc="Inference")
    report_distribution(results.preds, desc="Inference")
    save_inference_results(results, output_dir)


def main() -> None:
    config = simple_parsing.parse(Config, add_config_path_arg=True)

    set_seed(config.seed)
    suppress_transformers_warnings()

    output_name = config.output_name or datetime.now().isoformat()
    output_dir = config.output_path / output_name
    output_dir.mkdir(exist_ok=True, parents=True)

    setup_logger(output_dir)
    logger.info(f"\n{config}")
    logger.info(f"Output directory: {output_dir.resolve()}\n")
    (output_dir / "args.json").write_text(
        json.dumps(dataclasses.asdict(config), default=str, indent=2)
    )

    tokenizer, model = init_model(
        config.model_name, config.max_seq_length, config.dropout
    )
    device = get_device()

    if config.do_train:
        model = run_training(model, config, tokenizer, device, output_dir)

    if config.do_evaluation:
        if config.eval_data_path is None:
            raise ValueError("Eval data path must be specified for evaluation.")

        run_evaluation(
            model,
            config.eval_data_path,
            config,
            tokenizer,
            device,
            output_dir,
            desc="eval",
        )

    if config.do_test:
        if config.test_data_path is None:
            raise ValueError("Test data path must be specified for testing.")

        run_evaluation(
            model,
            config.test_data_path,
            config,
            tokenizer,
            device,
            output_dir,
            desc="test",
        )

    if config.do_inference:
        if config.infer_data_path is None:
            raise ValueError("Inference data path must be specified for inference.")

        run_inference(
            model, config.infer_data_path, config, tokenizer, device, output_dir
        )


if __name__ == "__main__":
    main()
