import collections
import dataclasses
import json
import logging
import os
import random
import shutil
import sys
import warnings
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, cast

import numpy as np
import simple_parsing
import torch
import torch.backends.mps
import transformers
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evaluation import log
from evaluation.metrics import (
    EvaluationResult,
    calc_metrics,
    report_gpu_memory,
    report_metrics,
)

# Suppress TensorFlow warnings. This must be done before importing transformers.
# Yes, it's an ugly hack, but it's necessary.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from transformers import (  # noqa: E402
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
)

logger = logging.getLogger("classifier")


class Prompt(Enum):
    PASSAGE = "passage"
    GOLD = "gold"
    COMBINED = "combined"
    INSTRUCTIONS = "instructions"


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
    # Batch size
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
    # Which prompt to use
    prompt: Prompt = Prompt.COMBINED
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
    # LR scheduler: 'constant', 'linear', 'cosine', etc. See transformers.get_scheduler.
    lr_scheduler: str = "constant"
    # Type of model: entailment or valid
    model_type: str = "valid"
    # Metric to compare best models and decide early stopping
    metric_for_best: str = "f1"
    # Number of prompt examples to print
    print_prompt_examples: int = 0

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


@dataclasses.dataclass
class LabelConfig:
    label2id: dict[str, int]
    id2label: dict[int, str]
    true_class: str


def get_labelling(reward_type: str) -> LabelConfig:
    "Get configuration necessary for labelling entailment or valid models."
    reward_type = reward_type.casefold()
    if reward_type.casefold() == "entailment":
        label2id = {
            "CONTRADICTION": 0,
            "ENTAILMENT": 1,
            "NEUTRAL": 2,
        }
        true_class = "ENTAILMENT"
    elif reward_type.casefold() == "valid":
        label2id = {
            "INVALID": 0,
            "VALID": 1,
        }
        true_class = "VALID"
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
    id2label = {id: label for label, id in label2id.items()}

    return LabelConfig(label2id, id2label, true_class)


def init_model(
    model_name: str, dropout: float | None, label_config: LabelConfig
) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    config = AutoConfig.from_pretrained(
        model_name, num_labels=len(label_config.label2id)
    )
    config.label2id = label_config.label2id
    config.id2label = label_config.id2label

    if dropout is not None:
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config
    )
    return tokenizer, model


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: AdamW,
    lr_scheduler: LRScheduler,
    device: torch.device,
    desc: str,
) -> float:
    total_loss = 0

    model.train()
    for batch in tqdm(train_loader, desc=desc):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(train_loader)


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
    tags: list[str] = []
    loss: float = 0

    model.eval()
    for batch in tqdm(val_loader, desc=desc):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            model_outputs = model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            preds.extend(torch.argmax(model_outputs.logits, dim=1).tolist())
            golds.extend(labels.tolist())
            loss += model_outputs.loss.item()

            passages.extend(batch["input"])
            outputs.extend(batch["output"])
            annotations.extend(batch["gold"])

            if "tag" in batch:
                tags.extend(batch["tag"])

    return EvaluationResult(
        golds=golds,
        preds=preds,
        passages=passages,
        outputs=outputs,
        annotations=annotations,
        loss=loss / len(val_loader),
        tags=tags or None,
    )


@dataclasses.dataclass
class InferenceResult:
    preds: list[int]
    passages: list[str]
    outputs: list[str]
    annotations: list[str]
    tags: list[str] | None = None


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
    tags: list[str] = []

    model.eval()
    for batch in tqdm(val_loader, desc=desc):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        with torch.no_grad():
            model_outputs = model(
                input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )

            preds.extend(torch.argmax(model_outputs.logits, dim=1).tolist())

            passages.extend(batch["input"])
            outputs.extend(batch["output"])
            annotations.extend(batch["gold"])

            if "tag" in batch:
                tags.extend(batch["tag"])

    return InferenceResult(
        preds=preds,
        passages=passages,
        outputs=outputs,
        annotations=annotations,
        tags=tags or None,
    )


def train(
    model: PreTrainedModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    early_stopping_patience: int,
    scheduler_type: str,
    metric_for_best: str,
    metrics_file: Path,
) -> PreTrainedModel:
    model = model.to(device)  # type: ignore
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_loader) * num_epochs),
    )

    # Saving the temporary best model in same place as the metrics file in case
    # training is interrupted and the final model is not saved. This will be deleted
    # after training is done and properly saved afterwards.
    best_model_path = metrics_file.parent / "best_model"
    model.save_pretrained(best_model_path)

    best_metric = 0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        avg_train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            device,
            desc=f"Training ({epoch + 1}/{num_epochs})",
        )
        logger.info(f"Epoch {epoch + 1}/{num_epochs} -> Loss: {avg_train_loss}")

        results = evaluate(
            model, val_loader, device, desc=f"Evaluation ({epoch + 1}/{num_epochs})"
        )

        metrics = calc_metrics(results)
        report_metrics(logger, metrics, "Train evaluation")
        save_metrics(metrics_file, **metrics, train_loss=avg_train_loss)

        if metrics[metric_for_best] > best_metric:
            best_metric = metrics[metric_for_best]
            model.save_pretrained(best_model_path)
            logger.info(f"New best: {best_metric:.4f} {metric_for_best}")
        else:
            epochs_without_improvement += 1
            logger.info(f"{epochs_without_improvement} epochs without improvement")
            if epochs_without_improvement >= early_stopping_patience:
                logger.info("Early stopping")
                break

    best_model = AutoModelForSequenceClassification.from_pretrained(
        best_model_path, config=model.config
    ).to(device)
    shutil.rmtree(best_model_path, ignore_errors=True)
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
    tag: str | None = None


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
            "token_type_ids": self.input_tokens["token_type_ids"][idx],
            "input": self.data[idx].input,
            "output": self.data[idx].output,
            "gold": self.data[idx].gold,
        }
        if self.labels is not None:
            d["labels"] = self.labels[idx]
        if tag := self.data[idx].tag:
            d["tag"] = tag
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
            tag=d.get("tag"),
        )
        for d in data[:n]
    ]


def get_prompt(prompt: Prompt, data: list[DataEntry]) -> list[str]:
    match prompt:
        case Prompt.PASSAGE:
            return [d.input for d in data]
        case Prompt.GOLD:
            return [d.gold for d in data]
        case Prompt.COMBINED:
            return [f"{d.input}\n{d.gold}" for d in data]
        case Prompt.INSTRUCTIONS:
            return [
                f"""\
Given the following input and reference answer, decide whether the candidate answer is valid:

Input:
{d.input}

Reference answer:
{d.gold}"""
                for d in data
            ]


def get_answer(prompt: Prompt, data: list[DataEntry]) -> list[str]:
    match prompt:
        case Prompt.PASSAGE | Prompt.GOLD | Prompt.COMBINED:
            return [d.output for d in data]
        case Prompt.INSTRUCTIONS:
            return [f"Candidate answer:\n{d.output}" for d in data]


def print_prompt_examples(prompt: Prompt, data: list[DataEntry], *, n: int) -> None:
    if n == 0:
        return
    print_prompt_example(prompt, [d for d in data if d.valid], "Valid examples", n)
    print_prompt_example(
        prompt, [d for d in data if not d.valid], "Invalid examples", n
    )


def print_prompt_example(
    prompt: Prompt, data: list[DataEntry], title: str, n: int
) -> None:
    logger.warning(f"\n\n>>> {title}\n")
    for _ in range(n):
        idx = random.randint(0, len(data) - 1)
        prompt_ = get_prompt(prompt, [data[idx]])[0]
        answer = get_answer(prompt, [data[idx]])[0]
        sep = "-" * 40
        example = f"Prompt\n{sep}\n{prompt_}\n\nAnswer\n{sep}\n{answer}\n\n"
        logger.warning(f"\n{example}")


def get_max_seq_length(
    model_inputs: BatchEncoding, tokenizer: PreTrainedTokenizer
) -> int:
    input_ids = model_inputs["input_ids"]
    non_padding_tokens = cast(torch.Tensor, input_ids != tokenizer.pad_token_id)
    actual_lengths = non_padding_tokens.sum(dim=1)
    return int(actual_lengths.max().item())


def preprocess_data(
    data: list[DataEntry],
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    batch_size: int,
    prompt: Prompt,
    prompt_examples: int,
    has_labels: bool,
) -> DataLoader:
    prompts = get_prompt(prompt, data)
    answers = get_answer(prompt, data)

    print_prompt_examples(prompt, data, n=prompt_examples)

    model_inputs = tokenizer(
        text=prompts,
        text_pair=answers,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=max_seq_length,
        return_token_type_ids=True,
    )
    logger.info(f"Max sequence length: {get_max_seq_length(model_inputs, tokenizer)}")

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
    logger.setLevel(logging.INFO)

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(output_dir / f"train_{ts}.log")
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log.ColourFormatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(console_handler)


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
            "tag": results.tags[i] if results.tags else None,
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
            "tag": results.tags[i] if results.tags else None,
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
        config.prompt,
        config.print_prompt_examples,
        has_labels=True,
    )

    eval_data = load_json(config.eval_data_path, config.max_samples)
    eval_loader = preprocess_data(
        eval_data,
        tokenizer,
        config.max_seq_length,
        config.batch_size,
        config.prompt,
        config.print_prompt_examples,
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
        config.lr_scheduler,
        config.metric_for_best,
        metrics_file,
    )
    save_model(trained_model, tokenizer, output_dir)

    results = evaluate(trained_model, eval_loader, device, desc="Final evaluation")
    metrics = calc_metrics(results)
    report_metrics(logger, metrics, "Final train evaluation")
    save_eval_results(results, metrics, output_dir, desc="eval")

    return trained_model


def report_inference(results: InferenceResult) -> None:
    c = collections.Counter(results.preds)
    logger.info("Inference results:")
    logger.info(f"  Valid: {c[True]} ({c[True] / len(results.preds):.2%})")
    logger.info(f"  Invalid: {c[False]} ({c[False] / len(results.preds):.2%})")


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
        config.prompt,
        config.print_prompt_examples,
        has_labels=True,
    )

    results = evaluate(model, loader, device, desc=f"{desc.capitalize()} evaluation")
    metrics = calc_metrics(results)
    report_metrics(logger, metrics, desc)
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
        config.prompt,
        config.print_prompt_examples,
        has_labels=False,
    )

    results = infer(model, loader, device, desc="Inference")
    report_inference(results)
    save_inference_results(results, output_dir)


def main() -> None:
    config = simple_parsing.parse(Config, add_config_path_arg=True)

    set_seed(config.seed)
    suppress_transformers_warnings()

    output_name = config.output_name or datetime.now().isoformat()
    output_dir = config.output_path / output_name
    output_dir.mkdir(exist_ok=True, parents=True)

    setup_logger(output_dir)
    git_commit = log.get_current_commit_shorthash()
    logger.info(f"\n{config}")
    logger.info(f"Output directory: {output_dir.resolve()}")
    logger.info(f"Git commit: {git_commit}\n")

    (output_dir / "args.json").write_text(
        json.dumps(
            dataclasses.asdict(config) | {"git_commit": git_commit},
            default=str,
            indent=2,
        )
    )

    label_config = get_labelling(config.model_type)
    tokenizer, model = init_model(config.model_name, config.dropout, label_config)
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
    report_gpu_memory(main, logger)
