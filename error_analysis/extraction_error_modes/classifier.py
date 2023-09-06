import argparse
import copy
import dataclasses
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
import torch.backends.mps
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AdamW,
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
    # Batch size
    batch_size: int
    # Number of samples to use from data
    max_samples: int | None = None
    # Output directory for model
    output_dir: Path = Path("output")

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


class CustomDataset(Dataset):
    def __init__(
        self, data: list[dict[str, Any]], tokenizer: PreTrainedTokenizer, max_len: int
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data[idx]
        text = row["input"]
        text_pair = row["gold"] + " [SEP] " + row["output"]

        encoding = self.tokenizer(
            text=text,
            text_pair=text_pair,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
        )
        input_ids = cast(torch.Tensor, encoding["input_ids"])
        attention_mask = cast(torch.Tensor, encoding["attention_mask"])
        token_type_ids = cast(torch.Tensor, encoding["token_type_ids"])
        label = torch.tensor(row["valid"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "label": label,
        }


def init_model(model_name: str) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config, num_labels=2
    )
    model.config.label2id = {True: 1, False: 0}
    model.config.id2label = {1: True, 0: False}
    return tokenizer, model


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(
    model: torch.nn.Module, val_loader: DataLoader, device: torch.device
) -> tuple[list[int], list[int]]:
    model.eval()
    preds: list[int] = []
    golds: list[int] = []
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).tolist())
            golds.extend(labels.tolist())

    return golds, preds


def train(
    model: PreTrainedModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
) -> PreTrainedModel:
    device = get_device()
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    best_f1 = 0
    best_model = model

    for epoch in range(num_epochs):
        avg_train_loss = train_epoch(model, train_loader, optimizer, device)
        true_labels, preds = evaluate(model, val_loader, device)

        acc = accuracy_score(true_labels, preds)
        prec = precision_score(true_labels, preds)
        rec = recall_score(true_labels, preds)
        f1 = f1_score(true_labels, preds)

        print(
            f"Epoch {epoch + 1}/{num_epochs} -> Loss: {avg_train_loss},"
            f" Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1: {f1}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_model = copy.deepcopy(model)

    return best_model


def get_device() -> torch.device:
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
    dataset = CustomDataset(data, tokenizer, max_len=max_seq_length)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model using the config file")
    parser.add_argument("config_path", type=Path, help="Path to the config JSON file")
    args = parser.parse_args()

    config = Config(**json.loads(args.config_path.read_text()))

    output_dir = config.output_dir / datetime.now().isoformat()
    output_dir.mkdir(exist_ok=True, parents=True)
    setup_logger(output_dir)

    logger.info(f"\n{config}")
    logger.info(f"\nOutput directory: {output_dir}\n")
    (output_dir / "args.json").write_text(
        json.dumps(dataclasses.asdict(config), default=str, indent=2)
    )

    tokenizer, model = init_model(config.model_name)

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
        config.num_epochs,
        config.learning_rate,
    )
    save_model(trained_model, tokenizer, output_dir)


if __name__ == "__main__":
    main()
