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
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger("classifier")


@dataclasses.dataclass
class Config:
    # Classifier model name or path
    model_name: str
    # Path to data. Will be used for both train and dev.
    data_path: Path
    # Learning rate for AdamW optimizer
    learning_rate: float
    # Percentage of data to use for training
    split_percentage: float
    # Number of epochs to train
    num_epochs: int
    # Maximum length for model output (tokens)
    max_seq_length: int
    # Output directory for model Batch size
    batch_size: int
    # Number of samples to use from data
    max_samples: int | None = None
    # Path to the directory where the output will be saved
    output_path: Path = Path("output")
    # Random seed for reproducibility
    seed: int = 0
    # Name of the output directory. If unspecified, will be generated the current
    # ISO timestamp.
    output_name: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        "Ignore unknown arguments"
        for f in dataclasses.fields(self):
            if f.name not in kwargs:
                continue

            value = kwargs[f.name]
            if f.name in ["data_path", "output_dir"]:
                value = Path(value)
            setattr(self, f.name, value)

    def __str__(self) -> str:
        config_lines = [">>>> CONFIGURATION"]
        for key, val in dataclasses.asdict(self).items():
            value = val.resolve() if isinstance(val, Path) else val
            config_lines.append(f"  {key}: {value}")
        return "\n".join(config_lines)


@dataclasses.dataclass
class ClassifierDataset(Dataset):
    input_tokens: Mapping[str, torch.Tensor]
    labels: torch.Tensor
    data: list[dict[str, Any]]

    def __len__(self) -> int:
        return self.input_tokens["input_ids"].size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_tokens["input_ids"][idx],
            "attention_mask": self.input_tokens["attention_mask"][idx],
            "labels": self.labels[idx],
            "input": self.data[idx]["input"],
            "output": self.data[idx]["output"],
            "gold": self.data[idx]["gold"],
        }


def init_model(model_name: str) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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

    model.eval()
    for batch in tqdm(val_loader, desc=desc):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            model_outputs = model(input_ids, attention_mask=attention_mask)

            preds.extend(torch.argmax(model_outputs.logits, dim=1).tolist())
            golds.extend(labels.tolist())

            passages.extend(batch["input"])
            outputs.extend(batch["output"])
            annotations.extend(batch["gold"])

    return EvaluationResult(
        golds=golds,
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
    }


def report_metrics(metrics: dict[str, float]) -> None:
    logger.info(
        "Evaluation results\n"
        f"    Accuracy : {metrics['accuracy']:.4f}\n"
        f"    Precision: {metrics['precision']:.4f}\n"
        f"    Recall   : {metrics['recall']:.4f}\n"
        f"    F1       : {metrics['f1']:.4f}\n",
    )


def train(
    model: PreTrainedModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
) -> PreTrainedModel:
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    best_f1 = 0
    best_model = model

    for epoch in range(num_epochs):
        avg_train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            desc=f"Training ({epoch + 1}/{num_epochs})",
        )
        results = evaluate(
            model, val_loader, device, desc=f"Evaluation ({epoch + 1}/{num_epochs})"
        )

        logger.info(f"Epoch {epoch + 1}/{num_epochs} -> Loss: {avg_train_loss}")

        metrics = calc_metrics(results)
        report_metrics(metrics)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = copy.deepcopy(model)

    return best_model


def get_device() -> torch.device:
    "Returns MPS if available, CUDA if available, otherwise CPU device."
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return torch.device(device)


def preprocess_data(
    data: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    split_percentage: float,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    model_inputs = tokenizer(
        # [d["passage"] for d in data],
        # [d["gold"] + " [SEP] " + d["output"] for d in data],
        [d["gold"] for d in data],
        [d["output"] for d in data],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=max_seq_length,
    )
    labels = [int(d["valid"]) for d in data]
    dataset = ClassifierDataset(
        input_tokens=model_inputs, labels=torch.tensor(labels), data=data
    )
    train_size = int(split_percentage * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


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


def save_results(
    results: EvaluationResult, metrics: dict[str, float], output_dir: Path
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
    (output_dir / "results.json").write_text(json.dumps(r, indent=2))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


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

    tokenizer, model = init_model(config.model_name)
    device = get_device()

    data = json.loads(config.data_path.read_text())[: config.max_samples]
    train_loader, val_loader = preprocess_data(
        data,
        tokenizer,
        config.max_seq_length,
        config.split_percentage,
        config.batch_size,
    )

    trained_model = train(
        model,
        train_loader,
        val_loader,
        device,
        config.num_epochs,
        config.learning_rate,
    )
    save_model(trained_model, tokenizer, output_dir)

    results = evaluate(model, val_loader, device, desc="Final evaluation")
    metrics = calc_metrics(results)
    report_metrics(metrics)
    save_results(results, metrics, output_dir)


if __name__ == "__main__":
    main()