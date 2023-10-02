# pyright: basic
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import dataclasses
import json
import logging
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import simple_parsing
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl import (
    AutoModelForSeq2SeqLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
)

import self_critique.util
from self_critique.minimal.util import (
    save_model,
    set_seed,
    supress_transformers_warnings,
)

logger = logging.getLogger("extract_train")


@dataclass
class Config:
    "The name of the Casual LM model we wish to fine with PPO."

    # Path to training data
    train_file: Path
    # Extraction model name or path
    extraction_model: str
    # Valid classifier model name or path
    valid_model: str
    # Learning rate
    learning_rate: float = 1e-5
    # PPO minibatch size
    mini_batch_size: int = 16
    # Reward model batch size
    batch_size: int = 256
    # Epochs
    num_epochs: int = 1
    # Gradient accumulation steps
    gradient_accumulation_steps: int = 1
    # Maximum sequence length for Seq2Seq model
    max_seq_length: int = 128
    # Fixed random seed
    seed: int = 0
    # Maximum number of samples used for training
    max_train_samples: int | None = None
    # Maximum number of samples used for evaluation
    max_eval_samples: int | None = None
    # Maximum number of samples used for prediction
    max_predict_samples: int | None = None
    # Max length for generated sequences from the Seq2Seq model
    max_generation_length: int = 128
    # Path to output directory where metrics, checkpoints and predictions will be saved
    output_dir: Path = Path("output")
    # Path to evaluation data
    eval_file: Path | None = None
    # Contrastive top-k used for reranking
    contrastive_top_k: int = 5
    # Contrastive degeneration penalty (alphe)
    degeneration_penalty: float = 0.5
    # KL penalty options:
    #    'kl': model_logp - ref_logp
    #    'abs': abs(kl)
    #    'mse': mean squared error mse(kl)
    #    'full': the actual kl for all tokens in the distribution"
    kl_penalty: str = "kl"
    # Use adaptive KL control, otherwise linear
    adaptive_kl_ctrl: bool = True
    # Initial KL penalty coefficient (used for adaptive and linear control)
    init_kl_coef: float = 0.2
    # Log with either 'wandb' or 'tensorboard'
    log_with: str | None = None
    # Every N batches to evaluate the model
    eval_batches: int = 10

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


def data_collator(data: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    return {key: [d[key] for d in data] for key in data[0]}


@dataclass
class Module:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer


def load_valid_classifier(model_name_or_path: str) -> Module:
    config = AutoConfig.from_pretrained(
        model_name_or_path, num_labels=len(LABEL2ID), revision="main"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision="main")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, config=config, revision="main"
    ).train(False)
    model.config.label2id = LABEL2ID
    model.config.id2label = ID2LABEL

    return Module(model, tokenizer)


def load_seq2seq_valuehead_model(model_name: str, *, train: bool) -> Module:
    model_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        model_name, config=model_config
    ).train(train)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return Module(model, tokenizer)


def load_seq2seq_model(model_name: str, *, train: bool) -> Module:
    model_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, config=model_config
    ).train(train)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return Module(model, tokenizer)


def text_decode(tokenizer: PreTrainedTokenizer, tensor: torch.Tensor) -> list[str]:
    output = tokenizer.batch_decode([r[1:] for r in tensor])
    return [clean_response(o) for o in output]


def text_encode(
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    text: list[str],
    text_pair: list[str] | None = None,
) -> BatchEncoding:
    return tokenizer(
        text=text,
        text_pair=text_pair,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    )


LABEL2ID = {"INVALID": 0, "VALID": 1}
ID2LABEL = {id: label for label, id in LABEL2ID.items()}


def run_valid_classifier(
    valid: Module,
    max_seq_length: int,
    batch_size: int,
    sentence1: list[str],
    sentence2: list[str],
    device: torch.device,
) -> tuple[list[torch.FloatTensor], list[str]]:
    inputs = text_encode(valid.tokenizer, max_seq_length, sentence1, sentence2)
    dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"])
    loader = DataLoader(dataset, batch_size=batch_size)

    scores: list[torch.FloatTensor] = []
    predictions: list[str] = []

    valid.model.eval()
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            outputs = valid.model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            # Get logit for the valid class and use it as a score
            scores.extend(outputs.logits.select(dim=-1, index=LABEL2ID["VALID"]))

            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(ID2LABEL[int(x.item())] for x in preds)

    return scores, predictions


def log_tensorboard(
    writer: SummaryWriter,
    eval_output: list[dict[str, Any]],
    n_iter: int,
) -> None:
    ratio = sum(x["valid_label"] == "VALID" for x in eval_output) / len(eval_output)
    writer.add_scalar("eval/valid_ratio", ratio, n_iter)


def train_extract(
    extract: Module,
    extract_ref: PreTrainedModel,
    valid: Module,
    train_dataset: Dataset,
    args: Config,
    eval_dataset: Dataset | None,
    output_dir: Path,
    eval_batches: int,
) -> tuple[Module, torch.device]:
    ppo_config = PPOConfig(
        model_name=args.extraction_model,
        learning_rate=args.learning_rate,
        mini_batch_size=args.mini_batch_size,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        adap_kl_ctrl=args.adaptive_kl_ctrl,
        kl_penalty=args.kl_penalty,
        init_kl_coef=args.init_kl_coef,
        log_with=args.log_with,
    )
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=extract.model,
        ref_model=extract_ref,
        tokenizer=extract.tokenizer,
        dataset=train_dataset,
        data_collator=data_collator,
    )

    best_model = copy.deepcopy(ppo_trainer.model)
    best_ratio = 0.0

    device = ppo_trainer.accelerator.device
    valid.model = valid.model.to(device)
    tb_writer = SummaryWriter(log_dir=output_dir / "tb")

    # Evaluate before training to facilitate comparison with batches
    if eval_dataset is not None:
        eval_result = evaluate(
            dataset=eval_dataset,
            extract=extract,
            extract_ref=extract_ref,
            valid=valid,
            args=args,
            device=device,
            desc="Eval (-1)",
        )
        save_results(
            result=eval_result,
            dir=output_dir,
            file_name="mini_eval_result_0.0.json",
        )
        log_tensorboard(tb_writer, eval_result, -1)

    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(
            tqdm(ppo_trainer.dataloader, desc=f"Train ({epoch})")
        ):
            query_tensors = batch["input_ids"]

            # Contrastive generation
            response_tensors = ppo_trainer.generate(
                query_tensors,
                max_length=args.max_generation_length,
                penalty_alpha=args.degeneration_penalty,
                top_k=args.contrastive_top_k,
            )
            extract_response = text_decode(extract.tokenizer, response_tensors)

            scores, labels = run_valid_classifier(
                valid=valid,
                max_seq_length=args.max_seq_length,
                batch_size=args.batch_size,
                sentence1=batch["context"],
                sentence2=extract_response,
                device=device,
            )
            rewards = scores

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            stats["metrics/valid_ratio"] = sum(
                label == "VALID" for label in labels
            ) / len(labels)
            log_batch = {
                "query": query_tensors,
                "response": response_tensors,
            }
            ppo_trainer.log_stats(stats, log_batch, rewards)

            if eval_dataset is not None and batch_idx % eval_batches == 0:
                eval_result = evaluate(
                    dataset=eval_dataset,
                    extract=extract,
                    extract_ref=extract_ref,
                    valid=valid,
                    args=args,
                    device=device,
                    desc=f"Eval  ({epoch}.{batch_idx+1})",
                )
                save_results(
                    result=eval_result,
                    dir=output_dir,
                    file_name=f"mini_eval_result_{epoch}.{batch_idx+1}.json",
                )
                log_tensorboard(
                    tb_writer,
                    eval_result,
                    epoch * len(ppo_trainer.dataloader) + batch_idx,
                )

                eval_ratio = sum(
                    d["valid_label"] == "VALID" for d in eval_result
                ) / len(eval_result)
                if eval_ratio > best_ratio:
                    best_model = copy.deepcopy(ppo_trainer.model)
                    best_ratio = eval_ratio
                    path = output_dir / "best"
                    save_model(
                        model=best_model, tokeniser=extract.tokenizer, output_dir=path
                    )
                    (path / "stats.json").write_text(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "batch": batch_idx,
                                "ratio": eval_ratio,
                            }
                        )
                    )
                    logger.info(
                        "New best model at epoch %d. Saving to %s.", epoch, path
                    )

        if eval_dataset is not None:
            eval_result = evaluate(
                dataset=eval_dataset,
                extract=extract,
                valid=valid,
                extract_ref=extract_ref,
                args=args,
                device=device,
                desc=f"Eval  ({epoch})",
            )
            save_results(
                result=eval_result,
                dir=output_dir,
                file_name=f"eval_result_{epoch}.json",
            )
    logger.info(
        "Finished training. Best model at %s with ratio %f.",
        output_dir / "best",
        best_ratio,
    )
    return Module(best_model, extract.tokenizer), device


@dataclass
class Seq2SeqEntry:
    id: str
    context: str
    question: str
    answers: str
    question_type: str


def load_data(file_path: Path, max_samples: int | None = None) -> list[Seq2SeqEntry]:
    data = json.loads(file_path.read_text())
    if "data" in data:
        data = data["data"]
    return [
        Seq2SeqEntry(
            id=d["id"],
            context=d["context"],
            question=d["question"],
            answers=d["answers"],
            question_type=d["question_type"],
        )
        for d in data
    ][:max_samples]


def preprocess_data(
    tokeniser: PreTrainedTokenizer,
    data: list[Seq2SeqEntry],
    max_seq_length: int,
    device: str | torch.device,
    desc: str | None = None,
) -> Dataset:
    desc = desc or ""
    source_texts = [f"{d.question.lstrip()}\n{d.context.lstrip()}" for d in data]
    target_texts = [d.answers for d in data]

    model_inputs = tokeniser(
        source_texts,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    )
    labels = tokeniser(
        text_target=target_texts,
        padding="max_length",
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
    )

    return Seq2SeqDataset(
        input_tokens=model_inputs,
        target_tokens=labels,
        data=data,
        device=device,
    )


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
    device: str | torch.device

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


def clean_response(s: str, eos_tag: str = "</s>") -> str:
    try:
        return s[: s.index(eos_tag)]
    except ValueError:
        return s


def evaluate(
    dataset: Dataset,
    extract: Module,
    extract_ref: PreTrainedModel,
    valid: Module,
    device: torch.device,
    args: Config,
    desc: str | None = None,
) -> list[dict[str, Any]]:
    @dataclass
    class BlockOutput:
        extract_txt: list[str]
        valid_labels: list[str]
        scores: list[torch.FloatTensor]

    def run_block(
        extract_model: PreTrainedModel,
        inputs: torch.Tensor,
        original_sentence: list[str],
    ) -> BlockOutput:
        # Contrastive generation
        extract_response_tensor = extract_model.generate(
            inputs,
            max_length=args.max_generation_length,
            penalty_alpha=args.degeneration_penalty,
            top_k=args.contrastive_top_k,
        )
        extract_response_txt = text_decode(extract.tokenizer, extract_response_tensor)

        scores, valid_labels = run_valid_classifier(
            valid=valid,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            sentence1=original_sentence,
            sentence2=extract_response_txt,
            device=device,
        )

        return BlockOutput(extract_response_txt, valid_labels, scores)

    desc = desc or "Evaluate"
    loader = DataLoader(dataset, batch_size=args.batch_size)

    output: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc=desc):
        inputs = batch["input_ids"].to(device)
        original_sentence = batch["context"]

        rl_output = run_block(extract.model, inputs, original_sentence)
        ref_output = run_block(extract_ref, inputs, original_sentence)

        assert len(rl_output.valid_labels) == len(inputs)
        for i in range(len(inputs)):
            output.append(
                {
                    "id": batch["id"][i],
                    "answers": batch["answers"][i],
                    "question_type": batch["question_type"][i],
                    "context": batch["context"][i],
                    "rl_extract_txt": rl_output.extract_txt[i],
                    "ref_extract_txt": ref_output.extract_txt[i],
                    "valid_label": rl_output.valid_labels[i],
                    "ref_valid_label": ref_output.valid_labels[i],
                    "scores": rl_output.scores[i].tolist(),
                }
            )

    log_label_distribution([d["valid_label"] for d in output], desc=f"{desc}: RL model")
    log_label_distribution(
        [d["ref_valid_label"] for d in output], desc=f"{desc}: Ref model"
    )

    return output


def log_label_distribution(labels: list[str], desc: str = "Model") -> None:
    label_dist = Counter(labels)
    msg = "\n".join(
        [
            f"\n{desc} label distribution:",
            *(
                f"  {label}: {count} ({count / len(labels)})"
                for label, count in label_dist.items()
            ),
        ]
    )
    logger.info(f"{msg}\n")


def save_results(
    result: list[dict[str, Any]], dir: Path, file_name: str = "eval_result.json"
) -> None:
    (dir / file_name).write_text(json.dumps(result))


def resolve(path_or_name: str | Path) -> str:
    """Resolve the path to from the project root. If it exists, return it,
    otherwise return the original path.
    """
    resolved = self_critique.util.resolve_path(path_or_name)
    if Path(resolved).exists():
        return str(resolved)
    return str(path_or_name)


def resolve_arg_paths(args: Config) -> Config:
    return dataclasses.replace(
        args,
        extraction_model=resolve(args.extraction_model),
        valid_model=resolve(args.valid_model),
        train_file=Path(resolve(args.train_file)),
        eval_file=Path(resolve(args.eval_file)) if args.eval_file else None,
        output_dir=Path(resolve(args.output_dir)),
    )


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
    args = simple_parsing.parse(Config, add_config_path_arg=True)
    args = resolve_arg_paths(args)

    output_dir = args.output_dir / datetime.now().isoformat()
    output_dir.mkdir(exist_ok=True, parents=True)
    setup_logger(output_dir)

    logger.info(f"\n{args}")
    logger.info(f"output files: {output_dir}")

    (output_dir / "args.json").write_text(
        json.dumps(dataclasses.asdict(args), default=str, indent=2)
    )

    set_seed(args.seed)
    supress_transformers_warnings()

    if args.train_file is None:
        raise ValueError("Must provide a training file")

    extract = load_seq2seq_valuehead_model(args.extraction_model, train=True)
    extract_ref = create_reference_model(extract.model)
    valid = load_valid_classifier(args.valid_model)

    train_data = load_data(args.train_file, args.max_train_samples)
    train_dataset = preprocess_data(
        tokeniser=extract.tokenizer,
        data=train_data,
        max_seq_length=args.max_seq_length,
        device="cpu",
        desc="training",
    )

    eval_dataset = None
    if args.eval_file is not None:
        eval_dataset = load_data(args.eval_file, args.max_eval_samples)
        eval_dataset = preprocess_data(
            tokeniser=extract.tokenizer,
            data=eval_dataset,
            max_seq_length=args.max_seq_length,
            device="cpu",
            desc="evaluation",
        )

    extract, device = train_extract(
        extract=extract,
        extract_ref=extract_ref,
        valid=valid,
        train_dataset=train_dataset,
        args=args,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        eval_batches=args.eval_batches,
    )
    save_model(extract.model, extract.tokenizer, output_dir)

    if eval_dataset is not None:
        eval_result = evaluate(
            dataset=eval_dataset,
            extract=extract,
            valid=valid,
            extract_ref=extract_ref,
            args=args,
            device=device,
        )
        save_results(result=eval_result, dir=output_dir, file_name="eval_result.json")

    logger.info(f"Reminder: output files are in {output_dir}")


if __name__ == "__main__":
    main()
