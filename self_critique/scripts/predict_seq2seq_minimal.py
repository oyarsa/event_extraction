import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import simple_parsing
import torch
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


def filter_kwargs(cls: type) -> type:
    if not is_dataclass(cls):
        raise TypeError("filter_kwargs should only be used with dataclasses")

    original_init = cls.__init__  # type: ignore

    def new_init(cls: type, **kwargs: dict[str, Any]) -> None:
        filtered_kwargs = {
            f.name: kwargs[f.name] for f in fields(cls) if f.name in kwargs
        }
        return original_init(cls, **filtered_kwargs)

    cls.__init__ = new_init  # type: ignore
    return cls


@filter_kwargs
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


def log_metrics(metrics: dict[str, float], desc: str | None) -> None:
    desc = desc or "metrics"
    logging.info(f">>>> {desc.upper()}")

    padding = max(len(k) for k in metrics)
    for k, v in metrics.items():
        logging.info(f"  {k:>{padding}}: {v}")


def preprocess_data(
    tokeniser: PreTrainedTokenizer,
    data: list[dict[str, str]],
    config: Config,
    shuffle: bool,
    batch_size: int,
) -> DataLoader:
    source_texts = [f"{d['question'].lstrip()}\n{d['context'].lstrip()}" for d in data]
    target_texts = [d["answers"] for d in data]

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
        input_tokens=model_inputs, target_tokens=labels, device=config.device
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


@dataclass
class Seq2SeqDataset(Dataset):
    input_tokens: dict[str, torch.Tensor]
    target_tokens: dict[str, torch.Tensor]
    device: str

    def __len__(self) -> int:
        return self.input_tokens["input_ids"].size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_tokens["input_ids"][idx].to(self.device),
            "attention_mask": self.input_tokens["attention_mask"][idx].to(self.device),
            "labels": self.target_tokens["input_ids"][idx].to(self.device),
        }


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
    config: Config,
    desc: str | None = None,
) -> EvalResult:
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokeniser.pad_token_id)
    total_loss = 0
    num_batches = 0

    all_predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for inputs in tqdm(loader, desc=desc):
            outputs = model(**inputs)

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

            num_batches += 1

    logging.info("Decoding output")
    predicted_texts = tokeniser.batch_decode(all_predictions, skip_special_tokens=True)

    logging.info("Calculating metrics")
    metrics = calculate_metrics(data, predicted_texts)
    log_metrics(metrics, desc)
    avg_loss = total_loss / num_batches
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
        shuffle=False,
        batch_size=config.per_device_train_batch_size,
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokeniser.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
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
            shuffle=False,
            batch_size=config.per_device_eval_batch_size,
        )

    best_f1 = 0
    early_stopping_counter = 0

    for epoch in range(config.num_train_epochs):
        model.train()

        total_loss = 0
        num_batches = 0

        for inputs in tqdm(train_loader, desc=f"Epoch {epoch+1} training"):
            optimizer.zero_grad()

            outputs = model(**inputs)
            logits = outputs.logits

            # TODO: Understand why this doesn't work: loss = criterion(logits, inputs["labels"])
            loss = criterion(
                logits.view(-1, logits.shape[-1]), inputs["labels"].view(-1)
            )
            loss.backward()

            optimizer.step()
            scheduler.step()

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
                config,
                desc=f"Epoch {epoch+1} evaluation",
            )
            logging.info(f"Epoch {epoch+1}, evaluation loss: {eval_result.loss}")

        if eval_result.metrics["f1"] > best_f1:
            best_f1 = eval_result.metrics["f1"]
            early_stopping_counter = 0

            logging.info("New best model! Saving to: %s", config.output_dir.resolve())
            model.save_pretrained(config.output_dir)
            tokeniser.save_pretrained(config.output_dir)
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= config.early_stopping_patience:
            logging.info(
                f"Early stopping: {early_stopping_counter} epochs without improvement"
            )
            break

    return model


def calculate_metrics(
    data: list[dict[str, str]], output: list[str]
) -> dict[str, float]:
    references: list[metric.MetricReference] = [
        {
            "id": inp["id"],
            "answers": inp["answers"],
            "question_type": inp["question_type"],
        }
        for inp in data
    ]
    predictions: list[metric.MetricPrediction] = [
        {
            "id": inp["id"],
            "prediction_text": out,
        }
        for inp, out in zip(data, output)
    ]

    return metric.FGCRCls()._compute(predictions=predictions, references=references)


@dataclass
class InferenceResult:
    predictions: list[dict[str, str]]
    metrics: dict[str, float]


def do_inference(
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizer,
    data: list[dict[str, str]],
    config: Config,
    desc: str,
) -> InferenceResult:
    """Perform prediction on data.

    `model` must be a Seq2Seq model and compatible with `tokeniser`.
    """
    desc = desc.capitalize()
    logging.info("*** %s ***", desc)
    model.eval()

    logging.info("Tokenising input")

    data = data[: config.max_predict_samples]
    logging.info("%d samples", len(data))
    loader = preprocess_data(
        tokeniser,
        data,
        config,
        shuffle=False,
        batch_size=config.per_device_train_batch_size,
    )

    logging.info("Generating output")

    predicted_ids: list[torch.Tensor] = []
    for input_batch in tqdm(loader, desc=desc):
        batch_predicted_ids = model.generate(
            input_batch["input_ids"],
            num_beams=config.generation_num_beams or model.config.num_beams,
            max_length=config.max_seq_length,
        )
        predicted_ids.extend(batch_predicted_ids)

    logging.info("Decoding output")
    predicted_texts = tokeniser.batch_decode(predicted_ids, skip_special_tokens=True)

    logging.info("Calculating metrics")
    metrics = calculate_metrics(data, predicted_texts)
    log_metrics(metrics, desc)
    output = [
        {"input": d["context"], "output": out, "gold": d["answers"]}
        for out, d in zip(predicted_texts, data)
    ]
    return InferenceResult(predictions=output, metrics=metrics)


def log_config(config: Config) -> None:
    logging.info(">>>> CONFIGURATON")
    for key, value in asdict(config).items():
        logging.info(f"{key}: {value}")


def load_model(
    model_name_or_path: str | Path, max_seq_length: int, device: str
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path, config=model_config
    ).to(device)
    tokeniser = AutoTokenizer.from_pretrained(
        model_name_or_path, model_max_length=max_seq_length
    )
    model.resize_token_embeddings(len(tokeniser))
    return model, tokeniser


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = simple_parsing.parse(Config, add_config_path_arg=True)

    logging.basicConfig(
        level=logging.getLevelName(config.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    config.output_dir.mkdir(exist_ok=True, parents=True)
    log_config(config)

    model, tokeniser = load_model(
        config.model_name_or_path, config.max_seq_length, config.device
    )
    if config.do_train:
        if config.train_file is None:
            raise ValueError("train_file must be specified when training")

        train_data = json.loads(config.train_file.read_text())["data"]
        eval_data = (
            json.loads(config.validation_file.read_text())["data"]
            if config.validation_file is not None
            else None
        )

        model = do_train(model, tokeniser, train_data, eval_data, config)
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
        result = do_inference(model, tokeniser, eval_data, config, "evaluation")

        logging.info("Saving evaluation results to: %s", config.output_dir.resolve())
        (config.output_dir / "eval_output.json").write_text(
            json.dumps(result.predictions)
        )
        (config.output_dir / "eval_metrics.json").write_text(json.dumps(result.metrics))

    if config.do_predict:
        if config.test_file is None:
            raise ValueError("test_file must be specified when training")
        predict_data = json.loads(config.test_file.read_text())["data"]
        result = do_inference(model, tokeniser, predict_data, config, "prediction")

        logging.info("Saving prediction results to: %s", config.output_dir.resolve())
        (config.output_dir / "predict_output.json").write_text(
            json.dumps(result.predictions)
        )
        (config.output_dir / "predict_metrics.json").write_text(
            json.dumps(result.metrics)
        )


if __name__ == "__main__":
    main()
