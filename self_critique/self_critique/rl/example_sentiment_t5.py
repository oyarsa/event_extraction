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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizer,
    pipeline,
)
from trl import (
    AutoModelForSeq2SeqLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
)

from self_critique.minimal.util import set_seed


@dataclass
class ScriptArguments:
    "The name of the Casual LM model we wish to fine with PPO."

    model_name: str = field(default="t5-small", metadata={"help": "the model name"})
    classification_model: str = field(
        default="t5-small", metadata={"help": "the model name"}
    )
    learning_rate: Optional[float] = field(
        default=5e-5, metadata={"help": "the learning rate"}
    )
    mini_batch_size: Optional[int] = field(
        default=16, metadata={"help": "the PPO minibatch size"}
    )
    batch_size: Optional[int] = field(default=256, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "the maximum sequence length"}
    )
    seed: int = field(default=0, metadata={"help": "Fixed random seed"})
    max_samples: Optional[int] = field(
        default=None, metadata={"help": "Max data samples"}
    )


def build_imdb_dataset(
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    max_samples: int | None,
) -> Dataset:
    def tokenize(sample: Mapping[str, Sequence[Any]]) -> Mapping[str, Sequence[Any]]:
        sample["input_ids"] = tokenizer.encode(
            sample["review"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
        )
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = (
        load_dataset("imdb", split="train")
        .rename_columns({"text": "review"})
        .filter(lambda x: len(x["review"]) > 200, batched=False)
    )
    if max_samples is not None:
        max_samples = min(max_samples, len(ds))
        ds = ds.select(range(max_samples))
    return ds.map(tokenize).with_format("torch")


def data_collator(data: Sequence[Mapping[str, Any]]) -> Mapping[str, list[Any]]:
    return {key: [d[key] for d in data] for key in data[0]}


def load_entailment_model(
    model_name_or_path: str, device: str
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    config = AutoConfig.from_pretrained(
        model_name_or_path, num_labels=3, revision="main"
    )
    tokeniser = AutoTokenizer.from_pretrained(model_name_or_path, revision="main")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, config=config, revision="main"
    )
    model.config.label2id = {
        "CONTRADICTION": 0,
        "ENTAILMENT": 1,
        "NEUTRAL": 2,
    }
    model.config.id2label = {id: label for label, id in model.config.label2id.items()}

    return model.to(device), tokeniser


def load_seq2seq_model(
    model_name: str,
) -> tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizer]:
    model_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        model_name, config=model_config
    )
    ref_model = create_reference_model(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, ref_model, tokenizer


def parse_arguments() -> ScriptArguments:
    # Define the command-line arguments using the dataclass
    parser = HfArgumentParser(ScriptArguments)
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a JSON configuration file",
    )
    args = parser.parse_args()

    # Load the configuration from a JSON file if provided
    if args.config is not None:
        config_dict = json.loads(args.config.read_text())
        config = ScriptArguments(**config_dict)
    else:
        config = ScriptArguments()

    # Update the configuration with the command-line arguments
    for key, value in vars(args).items():
        if value is not None and key != "config":
            setattr(config, key, value)

    print(config)
    exit()
    return config


def main() -> None:
    # TODO: Support overrides from the CLI
    # parser = HfArgumentParser(ScriptArguments)
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     args = parser.parse_json_file(sys.argv[1])[0]
    # else:
    #     args = parser.parse_args_into_dataclasses()[0]
    args = parse_arguments()

    set_seed(args.seed)

    model, ref_model, tokenizer = load_seq2seq_model(args.model_name)
    dataset = build_imdb_dataset(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        max_samples=args.max_samples,
    )

    ppo_config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        mini_batch_size=args.mini_batch_size,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    ppo_trainer = PPOTrainer(
        ppo_config,
        model,
        ref_model,
        tokenizer,
        dataset=dataset,
        data_collator=data_collator,
    )

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"

    entailment_model, entailment_tokenizer = load_entailment_model(
        args.classification_model, device
    )
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=entailment_model,
        tokenizer=entailment_tokenizer,
        device=device,
    )

    label_to_reward = {
        "CONTRADICTION": -1.0,
        "ENTAILMENT": 1.0,
        "NEUTRAL": 0.0,
    }

    for _, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        print("# Get response from t5")
        response_tensors = ppo_trainer.generate(
            query_tensors,
            num_beams=model.config.num_beams,
            max_length=10,
        )
        print("# Decode response")
        batch["response"] = tokenizer.batch_decode([r[1:] for r in response_tensors])

        print("# Compute entailment labels")
        texts = [f"{q}\n{r}" for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(
            texts, top_k=1, function_to_apply="none", batch_size=16
        )

        print("# Calculate rewards")
        rewards = [
            torch.tensor(label_to_reward[output[0]["label"]]).to(device)
            for output in pipe_outputs
        ]

        print("# Run PPO step")
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)


if __name__ == "__main__":
    main()
