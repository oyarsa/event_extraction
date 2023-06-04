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

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import simple_parsing
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl import (
    AutoModelForSeq2SeqLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
)

from self_critique.minimal.util import (
    save_model,
    set_seed,
    supress_transformers_warnings,
)


@dataclass
class Config:
    "The name of the Casual LM model we wish to fine with PPO."

    # Path to training data
    train_file: Path
    # Seq2Seq model name or path
    seq2seq_model: str
    # Entailment model name or path
    entailment_model: str
    # Learning rate
    learning_rate: float = 5e-5
    # PPO minibatch size
    mini_batch_size: int = 16
    # Reward model batch size
    batch_size: int = 256
    # Epochs
    num_epochs: int = 1
    # Gradient accumulation steps
    gradient_accumulation_steps: int = 1
    # Maximum sequence length for Seq2Seq model
    max_seq_length: int = 512
    # Fixed random seed
    seed: int = 0
    # Maximum number of samples used for training
    max_train_samples: int | None = None
    # Maximum number of samples used for evaluation
    max_eval_samples: int | None = None
    # Maximum number of samples used for prediction
    max_predict_samples: int | None = None
    # Max length for generated sequences from the Seq2Seq model
    max_generation_length: int = 512
    # Path to output directory where metrics, checkpoints and predictions will be saved
    output_dir: Path = Path("output")
    # Path to evaluation data
    eval_file: Path | None = None

    def __init__(self, **kwargs: Any) -> None:
        "Ignore unknown arguments"
        for f in fields(self):
            if f.name in kwargs:
                setattr(self, f.name, kwargs[f.name])

    def __str__(self) -> str:
        config_lines = [">>>> CONFIGURATION"]
        for key, val in asdict(self).items():
            value = val.resolve() if isinstance(val, Path) else val
            config_lines.append(f"  {key}: {value}")
        return "\n".join(config_lines)


def data_collator(data: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    return {key: [d[key] for d in data] for key in data[0]}


@dataclass
class Labeller:
    label2id: dict[str, int]
    id2label: dict[int, str] = field(init=False)
    num_classes: int = field(init=False)

    def __post_init__(self) -> None:
        self.id2label = {id: label for label, id in self.label2id.items()}
        self.num_classes = len(self.label2id)

    def decode(self, labels: Iterable[int | float]) -> list[str]:
        return [self.id2label[int(label)] for label in labels]


def load_entailment_model(
    model_name_or_path: str,
    labeller: Labeller,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    config = AutoConfig.from_pretrained(
        model_name_or_path, num_labels=labeller.num_classes, revision="main"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision="main")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, config=config, revision="main"
    )
    model.config.label2id = labeller.label2id
    model.config.id2label = labeller.id2label

    return model, tokenizer


def load_seq2seq_model(
    model_name: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        model_name, config=model_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def label_to_reward(label: str) -> float:
    return {
        "CONTRADICTION": -1.0,
        "ENTAILMENT": 1.0,
        "NEUTRAL": 0.0,
    }[label]


def run_entailment(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    labeller: Labeller,
    max_seq_length: int,
    batch_size: int,
    sentence1: list[str],
    sentence2: list[str],
    device: torch.device,
) -> list[str]:
    inputs = tokenizer(
        sentence1,
        sentence2,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    )
    dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"])
    loader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    predictions: list[str] = []
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(labeller.decode(x.item() for x in preds))

    return predictions


def train(
    seq2seq_model: PreTrainedModel,
    seq2seq_ref_model: PreTrainedModel,
    seq2seq_tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    entailment_model: PreTrainedModel,
    entailment_tokenizer: PreTrainedTokenizer,
    labeller: Labeller,
    args: Config,
    eval_dataset: Dataset | None,
    output_dir: Path,
) -> tuple[PreTrainedModel, torch.device]:
    ppo_config = PPOConfig(
        model_name=args.seq2seq_model,
        learning_rate=args.learning_rate,
        mini_batch_size=args.mini_batch_size,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    ppo_trainer = PPOTrainer(
        ppo_config,
        seq2seq_model,
        seq2seq_ref_model,
        seq2seq_tokenizer,
        dataset=train_dataset,
        data_collator=data_collator,
    )

    device = ppo_trainer.accelerator.device
    entailment_model = entailment_model.to(device)

    for epoch in range(args.num_epochs):
        for batch in tqdm(
            ppo_trainer.dataloader,
            total=len(ppo_trainer.dataloader),
            desc=f"Train ({epoch})",
        ):
            query_tensors = batch["input_ids"]

            response_tensors = ppo_trainer.generate(
                query_tensors,
                num_beams=seq2seq_model.config.num_beams,
                max_length=args.max_generation_length,
            )
            response = text_decode(seq2seq_tokenizer, response_tensors)

            entailment_labels = run_entailment(
                model=entailment_model,
                tokenizer=entailment_tokenizer,
                max_seq_length=args.max_seq_length,
                batch_size=args.batch_size,
                labeller=labeller,
                sentence1=batch["original"],
                sentence2=response,
                device=device,
            )

            rewards = [
                torch.tensor(label_to_reward(label)).to(device)
                for label in entailment_labels
            ]

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            log_batch = {
                "query": query_tensors,
                "response": response_tensors,
            }
            ppo_trainer.log_stats(stats, log_batch, rewards)

        if eval_dataset is not None:
            eval_result = evaluate(
                dataset=eval_dataset,
                seq2seq_ref_model=seq2seq_ref_model,
                seq2seq_model=seq2seq_model,
                seq2seq_tokenizer=seq2seq_tokenizer,
                entailment_model=entailment_model,
                entailment_tokenizer=entailment_tokenizer,
                labeller=labeller,
                args=args,
                device=device,
                desc=f"Eval  ({epoch})",
            )
            save_results(
                result=eval_result,
                dir=output_dir,
                file_name=f"eval_result_{epoch}.json",
            )
    return ppo_trainer.model, device


@dataclass
class Seq2SeqEntry:
    id: str
    original: str
    context: str
    question: str
    answers: str
    question_type: str


def load_data(file_path: Path, max_samples: int | None = None) -> list[Seq2SeqEntry]:
    data = json.loads(file_path.read_text())
    if "data" in data:
        data = data["data"]
    return [Seq2SeqEntry(**d) for d in data][:max_samples]


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
    original: str
    answers: str
    question_type: str
    context: str


class Seq2SeqDatasetSeries(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    id: list[str]
    original: list[str]
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
            "original": self.data[idx].original,
            "answers": self.data[idx].answers,
            "question_type": self.data[idx].question_type,
            "context": self.data[idx].context,
        }


def clean_response(s: str, eos_tag: str = "</s>") -> str:
    try:
        return s[: s.index(eos_tag)]
    except ValueError:
        return s


def text_decode(tokenizer: PreTrainedTokenizer, tensor: torch.Tensor) -> list[str]:
    output = tokenizer.batch_decode([r[1:] for r in tensor])
    return [clean_response(o) for o in output]


def evaluate(
    dataset: Dataset,
    seq2seq_model: PreTrainedModel,
    seq2seq_ref_model: PreTrainedModel,
    seq2seq_tokenizer: PreTrainedTokenizer,
    entailment_model: PreTrainedModel,
    entailment_tokenizer: PreTrainedTokenizer,
    labeller: Labeller,
    device: torch.device,
    args: Config,
    desc: str | None = None,
) -> list[dict[str, Any]]:
    desc = desc or "Evaluate"
    loader = DataLoader(dataset, batch_size=args.batch_size)

    output: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc=desc, total=len(loader)):
        query_tensors = batch["input_ids"].to(device)

        rl_response_tensor = seq2seq_model.generate(
            query_tensors,
            num_beams=seq2seq_model.config.num_beams,
            max_length=args.max_generation_length,
        )
        rl_response = text_decode(seq2seq_tokenizer, rl_response_tensor)

        ref_response_tensor = seq2seq_ref_model.generate(
            query_tensors,
            num_beams=seq2seq_model.config.num_beams,
            max_length=args.max_generation_length,
        )
        ref_response = text_decode(seq2seq_tokenizer, ref_response_tensor)

        entailment_labels = run_entailment(
            model=entailment_model,
            tokenizer=entailment_tokenizer,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            labeller=labeller,
            sentence1=batch["original"],
            sentence2=rl_response,
            device=device,
        )

        rewards = [
            torch.tensor(label_to_reward(label)).to(device)
            for label in entailment_labels
        ]

        ref_entailment_labels = run_entailment(
            model=entailment_model,
            tokenizer=entailment_tokenizer,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            labeller=labeller,
            sentence1=batch["original"],
            sentence2=ref_response,
            device=device,
        )

        assert len(rewards) == len(entailment_labels) == len(batch["input_ids"])
        output.extend(
            {
                "id": batch["id"][i],
                "original": batch["original"][i],
                "answers": batch["answers"][i],
                "question_type": batch["question_type"][i],
                "context": batch["context"][i],
                "rl_response": rl_response[i],
                "ref_response": ref_response[i],
                "entailment_label": entailment_labels[i],
                "ref_entailment_label": ref_entailment_labels[i],
                "reward": rewards[i].tolist(),
            }
            for i in range(len(rewards))
        )
    log_label_distribution([d["entailment_label"] for d in output], desc="RL model")
    log_label_distribution(
        [d["ref_entailment_label"] for d in output], desc="Ref model"
    )

    return output


def log_label_distribution(labels: list[str], desc: str = "Model") -> None:
    label_dist = Counter(labels)
    print(f"\n{desc} label distribution:")
    for label, count in label_dist.items():
        print(f"  {label}: {count} ({count / len(labels)})")
    print()


def save_results(
    result: list[dict[str, Any]], dir: Path, file_name: str = "eval_result.json"
) -> None:
    (dir / file_name).write_text(json.dumps(result))


def main() -> None:
    args = simple_parsing.parse(Config, add_config_path_arg=True)
    print(f"{args}\n")

    set_seed(args.seed)
    supress_transformers_warnings()

    if args.train_file is None:
        raise ValueError("Must provide a training file")

    seq2seq_model, seq2seq_tokenizer = load_seq2seq_model(args.seq2seq_model)
    seq2seq_ref_model = create_reference_model(seq2seq_model)
    train_data = load_data(args.train_file, args.max_train_samples)
    train_dataset = preprocess_data(
        tokeniser=seq2seq_tokenizer,
        data=train_data,
        max_seq_length=args.max_seq_length,
        device="cpu",
        desc="training",
    )
    labeller = Labeller(
        {
            "CONTRADICTION": 0,
            "ENTAILMENT": 1,
            "NEUTRAL": 2,
        }
    )

    entailment_model, entailment_tokenizer = load_entailment_model(
        model_name_or_path=args.entailment_model, labeller=labeller
    )

    output_dir = args.output_dir / datetime.now().isoformat()
    output_dir.mkdir(exist_ok=True, parents=True)

    eval_dataset = None
    if args.eval_file is not None:
        eval_dataset = load_data(args.eval_file, args.max_eval_samples)
        eval_dataset = preprocess_data(
            tokeniser=seq2seq_tokenizer,
            data=eval_dataset,
            max_seq_length=args.max_seq_length,
            device="cpu",
            desc="evaluation",
        )

    seq2seq_model, device = train(
        seq2seq_model=seq2seq_model,
        seq2seq_ref_model=seq2seq_ref_model,
        seq2seq_tokenizer=seq2seq_tokenizer,
        train_dataset=train_dataset,
        entailment_model=entailment_model,
        entailment_tokenizer=entailment_tokenizer,
        labeller=labeller,
        args=args,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
    )
    save_model(seq2seq_model, seq2seq_tokenizer, output_dir)

    if eval_dataset is not None:
        eval_result = evaluate(
            dataset=eval_dataset,
            seq2seq_ref_model=seq2seq_ref_model,
            seq2seq_model=seq2seq_model,
            seq2seq_tokenizer=seq2seq_tokenizer,
            entailment_model=entailment_model,
            entailment_tokenizer=entailment_tokenizer,
            labeller=labeller,
            args=args,
            device=device,
        )
        save_results(result=eval_result, dir=output_dir, file_name="eval_result.json")


if __name__ == "__main__":
    main()
