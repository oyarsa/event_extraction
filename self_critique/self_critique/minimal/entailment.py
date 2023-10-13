# pyright: basic
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import simple_parsing
import torch
import torch.utils.data
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from self_critique.minimal.config import Config
from self_critique.minimal.util import (
    log_metrics,
    save_model,
    set_seed,
    setup_logging,
    suppress_transformers_warnings,
)


@dataclass
class EntailmentEntry:
    sentence1: str
    sentence2: str
    label: str
    id: str


@dataclass
class Labeller:
    num_labels: int
    id2label: dict[int, str]
    label2id: dict[str, int]

    def __init__(self, *datasets: Iterable[EntailmentEntry] | None) -> None:
        data = [d for d in datasets if d][0]
        labels = sorted({d.label for d in data})
        self.num_labels = len(labels)
        self.id2label = dict(enumerate(labels))
        self.label2id = {label: i for i, label in self.id2label.items()}

    def encode(self, txt_labels: Iterable[str]) -> list[int]:
        return [self.label2id[label] for label in txt_labels]

    def decode(self, int_labels: Iterable[int]) -> list[str]:
        return [self.id2label[label] for label in int_labels]


def preprocess_data(
    tokeniser: PreTrainedTokenizer,
    data: list[EntailmentEntry],
    config: Config,
    labeller: Labeller,
    shuffle: bool,
    batch_size: int,
    desc: str | None = None,
) -> DataLoader:
    desc = desc or ""
    logging.info(f"Preprocessing {desc} data")
    model_inputs = tokeniser(
        [d.sentence1 for d in data],
        [d.sentence2 for d in data],
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        max_length=config.max_seq_length,
    )
    labels = labeller.encode(d.label for d in data)

    dataset = EntailmentDataset(
        input_tokens=model_inputs,
        labels=torch.tensor(labels),
        data=data,
        device=config.device,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class EntailmentDatasetEntry(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    txt_labels: str
    sentence1: str
    sentence2: str
    id: str


class EntailmentDatasetSeries(TypedDict):
    input_tokens: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    txt_labels: list[str]
    sentence1: list[str]
    sentence2: list[str]
    id: list[str]


@dataclass
class EntailmentDataset(Dataset):
    input_tokens: Mapping[str, torch.Tensor]
    labels: torch.Tensor
    data: list[EntailmentEntry]
    device: str

    def __len__(self) -> int:
        return self.input_tokens["input_ids"].size(0)

    def __getitem__(self, idx: int) -> EntailmentDatasetEntry:
        return {
            "input_ids": self.input_tokens["input_ids"][idx].to(self.device),
            "attention_mask": self.input_tokens["attention_mask"][idx].to(self.device),
            "labels": self.labels[idx].to(self.device),
            "txt_labels": self.data[idx].label,
            "sentence1": self.data[idx].sentence1,
            "sentence2": self.data[idx].sentence2,
            "id": self.data[idx].id,
        }


def collect_model_data(
    all_data: list[EntailmentDatasetSeries],
) -> list[EntailmentDatasetEntry]:
    model_data: list[EntailmentDatasetEntry] = []
    for d in all_data:
        for i in range(len(d["id"])):
            entry = {k: d[k][i] for k in d}
            model_data.append(EntailmentDatasetEntry(**entry))
    return model_data


@dataclass
class EvalResult:
    loss: float
    metrics: dict[str, float]
    predictions: list[str]


def eval(
    model: PreTrainedModel,
    loader: DataLoader,
    labeller: Labeller,
    desc: str | None = None,
) -> EvalResult:
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    num_batches = 0

    all_predictions: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for inputs in tqdm(loader, desc=desc):
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
            loss = criterion(outputs.logits, inputs["labels"])
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)

            all_predictions.extend(int(x.item()) for x in preds)
            all_labels.extend(int(x.item()) for x in inputs["labels"])

            num_batches += 1

    metrics = calculate_metrics(all_labels, all_predictions)
    log_metrics(metrics, desc)

    avg_loss = total_loss / num_batches
    txt_predictions = labeller.decode(all_predictions)

    return EvalResult(
        loss=avg_loss,
        metrics=metrics,
        predictions=txt_predictions,
    )


def train(
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizer,
    train_data: list[EntailmentEntry],
    eval_data: list[EntailmentEntry] | None,
    config: Config,
    labeller: Labeller,
) -> PreTrainedModel:
    train_data = train_data[: config.max_train_samples]
    train_loader = preprocess_data(
        tokeniser,
        train_data,
        config,
        labeller,
        shuffle=True,
        batch_size=config.per_device_train_batch_size,
        desc="training",
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    num_optimisation_steps = config.num_train_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_optimisation_steps,
    )

    eval_loader = None
    if eval_data:
        eval_data = eval_data[: config.max_eval_samples]
        eval_loader = preprocess_data(
            tokeniser,
            eval_data,
            config,
            labeller,
            shuffle=True,
            batch_size=config.per_device_eval_batch_size,
            desc="evaluation",
        )

    best_f1 = -1.0
    early_stopping_counter = 0
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info("Starting training...")
    logging.info(f"Num examples: {len(train_data)}")
    logging.info(f"Num epochs: {config.num_train_epochs}")
    logging.info(f"Number of trainable parameters = {num_params}")
    logging.info(f"Total optimisation steps: {num_optimisation_steps}")

    for epoch in range(config.num_train_epochs):
        model.train()

        logging.info(f"Epoch {epoch+1} learning rate: {scheduler.get_last_lr()[0]}")

        total_loss = 0
        num_batches = 0

        for inputs in tqdm(train_loader, desc=f"Epoch {epoch+1} training"):
            optimizer.zero_grad()

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
            loss = criterion(outputs.logits, inputs["labels"])
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
                eval_loader,
                labeller,
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

    # Either we're not saving based on eval F1, or we're at the end of training
    # and haven't saved yet
    if best_f1 == -1:
        save_model(model, tokeniser, config.output_dir)

    return model


def calculate_metrics(gold: list[int], preds: list[int]) -> dict[str, float]:
    accuracy = accuracy_score(gold, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, preds, average="macro", zero_division=0  # type: ignore
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


@dataclass
class InferenceResult:
    predictions: list[dict[str, str]]
    metrics: dict[str, float]


def infer(
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizer,
    data: list[EntailmentEntry],
    config: Config,
    labeller: Labeller,
    desc: str,
) -> InferenceResult:
    """Perform prediction on data.

    `model` must be a Seq2Seq model and compatible with `tokeniser`.
    """
    desc = desc.capitalize()
    logging.info("*** %s ***", desc)
    model.eval()

    data = data[: config.max_predict_samples]
    logging.info("%d samples", len(data))
    loader = preprocess_data(
        tokeniser,
        data,
        config,
        labeller,
        shuffle=False,
        batch_size=config.per_device_train_batch_size,
        desc=desc,
    )

    predictions: list[int] = []
    labels: list[int] = []
    all_data: list[EntailmentDatasetSeries] = []
    for inputs in tqdm(loader, desc=desc):
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        preds = torch.argmax(logits.logits, -1)
        predictions.extend(int(x.item()) for x in preds)
        labels.extend(int(x.item()) for x in inputs["labels"])
        all_data.append(inputs)

    metrics = calculate_metrics(labels, predictions)
    log_metrics(metrics, desc)

    txt_predictions = labeller.decode(predictions)
    model_data = collect_model_data(all_data)
    output = [
        {
            "id": d["id"],
            "sentence1": d["sentence1"],
            "sentence2": d["sentence2"],
            "gold": d["txt_labels"],
            "prediction": out,
        }
        for out, d in zip(txt_predictions, model_data)
    ]
    return InferenceResult(predictions=output, metrics=metrics)


def load_model(
    model_name_or_path: str | Path, device: str, labeller: Labeller
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    logging.info("Loading model from %s", model_name_or_path)

    config = AutoConfig.from_pretrained(
        model_name_or_path, num_labels=labeller.num_labels, revision="main"
    )
    tokeniser = AutoTokenizer.from_pretrained(model_name_or_path, revision="main")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, config=config, revision="main"
    )
    model.config.label2id = labeller.label2id
    model.config.id2label = labeller.id2label

    return model.to(device), tokeniser


def load_data(file_path: Path) -> list[EntailmentEntry]:
    return [EntailmentEntry(**d) for d in json.loads(file_path.read_text())]


def save_results(desc: str, output_dir: Path, result: InferenceResult) -> None:
    desc = desc.lower()
    logging.info("Saving %s results to: %s", desc, output_dir.resolve())
    (output_dir / f"{desc}_output.json").write_text(json.dumps(result.predictions))
    (output_dir / f"{desc}_metrics.json").write_text(json.dumps(result.metrics))


def main() -> None:
    config = simple_parsing.parse(Config, add_config_path_arg=True)
    set_seed(config.seed)

    setup_logging(config.log_level)
    logging.info("%s", config)

    suppress_transformers_warnings()

    # TODO: I think this is erasing the contents of the directory, if there are
    # any. That shouldn't happen. Needs investigation.
    config.output_dir.mkdir(exist_ok=True, parents=True)

    train_data, eval_data, predict_data = None, None, None
    if config.train_file is not None:
        train_data = load_data(config.train_file)
    if config.validation_file is not None:
        eval_data = load_data(config.validation_file)
    if config.test_file is not None:
        predict_data = load_data(config.test_file)

    if all(x is None for x in [train_data, eval_data, predict_data]):
        raise ValueError(
            "At least one file (train, validation, test) must be specified"
        )

    labeller = Labeller(train_data, eval_data, predict_data)
    model, tokeniser = load_model(config.model_name_or_path, config.device, labeller)

    if config.do_train:
        if train_data is None:
            raise ValueError("train_file must be specified when training")

        model = train(model, tokeniser, train_data, eval_data, config, labeller)
        if config.load_best_model_at_end:
            logging.info("Loading best model from %s", config.output_dir.resolve())
            model, tokeniser = load_model(config.output_dir, config.device, labeller)

    if config.do_eval:
        if eval_data is None:
            raise ValueError(
                "validation_file must be specified when evaluating training"
            )
        result = infer(model, tokeniser, eval_data, config, labeller, "evaluation")
        save_results("eval", config.output_dir, result)

    if config.do_predict:
        if predict_data is None:
            raise ValueError("test_file must be specified when training")
        model, tokeniser = load_model(config.output_dir, config.device, labeller)
        result = infer(model, tokeniser, predict_data, config, labeller, "prediction")
        save_results("predict", config.output_dir, result)


if __name__ == "__main__":
    main()
