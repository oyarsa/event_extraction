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

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from typing import Any, TypedDict

import simple_parsing
import torch
from datasets import load_dataset
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

from self_critique.minimal.util import set_seed, supress_transformers_warnings


@dataclass
class Config:
    "The name of the Casual LM model we wish to fine with PPO."

    # Seq2Seq model name or path
    seq2seq_model: str = "t5-small"
    # Entailment model name or path
    entailment_model: str = "microsoft/deberta-v3-xsmall"
    # Learning rate
    learning_rate: float = 5e-5
    # PPO minibatch size
    mini_batch_size: int = 16
    # Reward model batch size
    batch_size: int = 256
    # Gradient accumulation steps
    gradient_accumulation_steps: int = 1
    # Maximum sequence length for Seq2Seq model
    max_seq_length: int = 512
    # Fixed random seed
    seed: int = 0
    # Max data samples
    max_samples: int | None = None
    # Max length for generated sequences from the Seq2Seq model
    max_generation_length: int = 512

    def __init__(self, **kwargs: Any) -> None:
        "Ignore unknown arguments"
        for f in fields(self):
            if f.name in kwargs:
                setattr(self, f.name, kwargs[f.name])

    def __str__(self) -> str:
        config_lines = [">>>> CONFIGURATION"]
        for key, value in asdict(self).items():
            config_lines.append(f"  {key}: {value}")
        return "\n".join(config_lines)


class ImdbDatasetEntry(TypedDict):
    input_ids: list[int]
    query: str


def build_imdb_dataset(
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    max_samples: int | None,
) -> Dataset:
    class ImdbSample(TypedDict):
        review: str

    def tokenize(sample: ImdbSample) -> ImdbDatasetEntry:
        input_ids: list[int] = tokenizer.encode(
            sample["review"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
        )
        query: str = tokenizer.decode(input_ids)
        return {
            "input_ids": input_ids,
            "query": query,
        }

    ds = (
        load_dataset("imdb", split="train")
        .rename_columns({"text": "review"})
        .filter(lambda x: len(x["review"]) > 200)
    )
    if max_samples is not None:
        max_samples = min(max_samples, len(ds))
        ds = ds.select(range(max_samples))
    return ds.map(tokenize).with_format("torch")


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
        for input_ids, attention_mask in tqdm(loader, desc="> Running entailment"):
            outputs = model(
                input_ids=input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
            )
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(labeller.decode(x.item() for x in preds))

    return predictions


def train(
    seq2seq_model: PreTrainedModel,
    seq2seq_tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    entailment_model: PreTrainedModel,
    entailment_tokenizer: PreTrainedTokenizer,
    labeller: Labeller,
    args: Config,
) -> PreTrainedModel:
    ppo_config = PPOConfig(
        model_name=args.seq2seq_model,
        learning_rate=args.learning_rate,
        mini_batch_size=args.mini_batch_size,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    ref_model = create_reference_model(seq2seq_model)
    ppo_trainer = PPOTrainer(
        ppo_config,
        seq2seq_model,
        ref_model,
        seq2seq_tokenizer,
        dataset=dataset,
        data_collator=data_collator,
    )

    device = ppo_trainer.accelerator.device
    entailment_model = entailment_model.to(device)

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        print(f"Epoch {epoch}")
        query_tensors = batch["input_ids"]

        print("\n# Get response from t5")
        response_tensors = ppo_trainer.generate(
            query_tensors,
            num_beams=seq2seq_model.config.num_beams,
            max_length=args.max_generation_length,
        )
        print("# Decode response")
        batch["response"] = seq2seq_tokenizer.batch_decode(
            [r[1:] for r in response_tensors]
        )

        print("# Compute entailment labels")
        entailment_labels = run_entailment(
            model=entailment_model,
            tokenizer=entailment_tokenizer,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            labeller=labeller,
            sentence1=batch["query"],
            sentence2=batch["response"],
        )

        print("# Calculate rewards")
        rewards = [
            torch.tensor(label_to_reward(label)).to(device)
            for label in entailment_labels
        ]

        print("# Run PPO step")
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
    return ppo_trainer.model


def main() -> None:
    args = simple_parsing.parse(Config, add_config_path_arg=True)
    print(f"{args}\n")

    set_seed(args.seed)
    supress_transformers_warnings()

    seq2seq_model, seq2seq_tokenizer = load_seq2seq_model(args.seq2seq_model)
    dataset = build_imdb_dataset(
        tokenizer=seq2seq_tokenizer,
        max_seq_length=args.max_seq_length,
        max_samples=args.max_samples,
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
    seq2seq_model = train(
        seq2seq_model=seq2seq_model,
        seq2seq_tokenizer=seq2seq_tokenizer,
        dataset=dataset,
        entailment_model=entailment_model,
        entailment_tokenizer=entailment_tokenizer,
        labeller=labeller,
        args=args,
    )


if __name__ == "__main__":
    main()
