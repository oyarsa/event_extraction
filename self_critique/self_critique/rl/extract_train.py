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
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict, cast

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
from trl.models.modeling_base import PreTrainedModelWrapper

import self_critique.util
from self_critique.metric.fgcr_metric_cls import parse_instance
from self_critique.minimal.util import (
    save_model,
    set_seed,
    suppress_transformers_warnings,
)

logger = logging.getLogger("extract_train")


class EvalPrompt(Enum):
    PASSAGE = "passage"
    GOLD = "gold"
    COMBINED = "combined"

    def get_eval_input(self, entry: "Seq2SeqEntry") -> str:
        match self:
            case EvalPrompt.PASSAGE:
                return entry.context
            case EvalPrompt.GOLD:
                return entry.answer
            case EvalPrompt.COMBINED:
                return f"{entry.context}\n{entry.answer}"


@dataclass
class Config:
    "The name of the Casual LM model we wish to fine with PPO."

    # Path to training data
    train_file: Path
    # Extraction model name or path
    extraction_model: str
    # Reward model name or path
    reward_model: str
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
    # Max length for generated sequences from the Seq2Seq model
    max_generation_length: int = 128
    # Path to output directory where metrics, checkpoints and predictions will be saved
    output_dir: Path = Path("output")
    # Name of the dir for this run inside `output_dir`
    run_name: str | None = None
    # Path to evaluation data
    eval_file: Path | None = None
    # Contrastive top-k used for reranking
    contrastive_top_k: int = 5
    # Contrastive degeneration penalty (alphe)
    degeneration_penalty: float = 0.5
    # KL penalty options:
    #    `kl`: model_logp - ref_logp
    #    `abs`: abs(kl)
    #    `mse`: mean squared error mse(kl)
    #    `full`: the actual kl for all tokens in the distribution"
    kl_penalty: str = "kl"
    # Use adaptive KL control, otherwise linear
    adaptive_kl_ctrl: bool = True
    # Initial KL penalty coefficient (used for adaptive and linear control)
    init_kl_coef: float = 0.2
    # Log with either 'wandb' or 'tensorboard'
    log_with: str | None = None
    # Every N batches to evaluate the model
    eval_batches: int = 10
    # Reward type: 'entailment' or 'valid'
    reward_type: str = "entailment"
    # Whether to perform training
    do_train: bool = True
    # Whether to perform evaluation
    do_eval: bool = True
    # Whether to rewrite the extraction output to natural language using a template
    rewrite: bool = False
    # Which prompt to use
    eval_prompt: EvalPrompt = EvalPrompt.GOLD

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
    model: PreTrainedModelWrapper
    tokenizer: PreTrainedTokenizer

    def to(self, device: torch.device) -> "Module":
        return Module(
            self.model.to(device),
            self.tokenizer,
        )


def load_reward_model(
    model_name_or_path: str, label2id: dict[str, int], id2label: dict[int, str]
) -> Module:
    config = AutoConfig.from_pretrained(
        model_name_or_path, num_labels=len(label2id), revision="main"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision="main")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, config=config, revision="main"
    ).train(False)
    model.config.label2id = label2id
    model.config.id2label = id2label

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
    output = tokenizer.batch_decode(torch.tensor([r[1:] for r in tensor]))
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


def run_reward(
    reward: Module,
    max_seq_length: int,
    batch_size: int,
    sentence1: list[str],
    sentence2: list[str],
    device: torch.device,
    label2id: dict[str, int],
    id2label: dict[int, str],
    true_class: str,
) -> tuple[list[torch.FloatTensor], list[str]]:
    inputs = text_encode(reward.tokenizer, max_seq_length, sentence1, sentence2)
    dataset = TensorDataset(
        torch.tensor(inputs["input_ids"]),
        torch.tensor(inputs["attention_mask"]),
    )
    loader = DataLoader(dataset, batch_size=batch_size)

    scores: list[torch.FloatTensor] = []
    predictions: list[str] = []

    reward.model.eval()
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            outputs = reward.model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            # Get logit for the reward class and use it as a score
            scores.extend(outputs.logits.select(dim=-1, index=label2id[true_class]))

            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(id2label[int(x.item())] for x in preds)

    return scores, predictions


def log_tensorboard(
    writer: SummaryWriter,
    eval_output: list[dict[str, Any]],
    n_iter: int,
    true_class: str,
) -> None:
    ratio = sum(x["reward_label"] == true_class for x in eval_output) / len(eval_output)
    writer.add_scalar("eval/reward_ratio", ratio, n_iter)


def train_extract(
    extract: Module,
    extract_ref: PreTrainedModel,
    reward: Module,
    train_dataset: Dataset,
    args: Config,
    eval_dataset: Dataset | None,
    output_dir: Path,
    eval_batches: int,
    true_class: str,
    label2id: dict[str, int],
    id2label: dict[int, str],
    rewrite: bool,
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
        ref_model=cast(PreTrainedModelWrapper, extract_ref),
        tokenizer=extract.tokenizer,
        dataset=train_dataset,
        data_collator=data_collator,
    )
    assert ppo_trainer.dataloader, "Error initialising PPOTrainer dataloader."

    best_model = copy.deepcopy(ppo_trainer.model)
    best_ratio = 0.0

    device = ppo_trainer.accelerator.device
    reward = reward.to(device)
    tb_writer = SummaryWriter(log_dir=output_dir / "tb")

    # Evaluate before training to facilitate comparison with batches
    if eval_dataset is not None:
        eval_result = evaluate(
            dataset=eval_dataset,
            extract=extract,
            extract_ref=extract_ref,
            reward=reward,
            args=args,
            device=device,
            true_class=true_class,
            label2id=label2id,
            id2label=id2label,
            rewrite=rewrite,
            desc="Eval (-1)",
        )
        save_results(
            result=eval_result,
            dir=output_dir,
            file_name="mini_eval_result_0.0.json",
        )
        log_tensorboard(tb_writer, eval_result, -1, true_class)

    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(
            tqdm(ppo_trainer.dataloader, desc=f"Train ({epoch})")
        ):
            ppo_trainer.model.train()
            query_tensors = batch["input_ids"]

            # Contrastive generation
            response_tensors = ppo_trainer.generate(
                query_tensors,
                max_length=args.max_generation_length,
                penalty_alpha=args.degeneration_penalty,
                top_k=args.contrastive_top_k,
            )
            extract_response = text_decode(
                extract.tokenizer, torch.tensor(response_tensors)
            )
            if rewrite:
                extract_response = [rewrite_extraction(x) for x in extract_response]

            scores, labels = run_reward(
                reward=reward,
                max_seq_length=args.max_seq_length,
                batch_size=args.batch_size,
                sentence1=batch["eval_inputs"],
                sentence2=extract_response,
                device=device,
                true_class=true_class,
                label2id=label2id,
                id2label=id2label,
            )
            rewards = scores

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            stats["metrics/reward_ratio"] = sum(
                label == true_class for label in labels
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
                    reward=reward,
                    args=args,
                    device=device,
                    true_class=true_class,
                    label2id=label2id,
                    id2label=id2label,
                    rewrite=rewrite,
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
                    true_class,
                )

                eval_ratio = sum(
                    d["reward_label"] == true_class for d in eval_result
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
                reward=reward,
                extract_ref=extract_ref,
                args=args,
                device=device,
                true_class=true_class,
                label2id=label2id,
                id2label=id2label,
                rewrite=rewrite,
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
    """Entry in a Seq2Seq dataset.

    Fields:
        id: The unique identifier for this entry.
        context: The input for this entry. E.g. original text for extraction or
            question + passage for QA).
        question: The question for this entry. E.g. for extraction, it's
            "What are the events?"
        answer: The gold answer for the entry.
    """

    id: str
    context: str
    question: str
    answer: str


def load_data(file_path: Path, max_samples: int | None = None) -> list[Seq2SeqEntry]:
    """Load data with the expected format.

    JSON list of objects with the following fields:
    - id: str
    - context: str
    - question: str
    - answers: str

    The JSON list may be behind a "data" key.
    """
    data = json.loads(file_path.read_text())
    if "data" in data:
        data = data["data"]
    return [
        Seq2SeqEntry(
            id=d["id"],
            context=d["context"],
            question=d["question"],
            answer=d["answers"],
        )
        for d in data
    ][:max_samples]


def preprocess_data(
    tokeniser: PreTrainedTokenizer,
    data: list[Seq2SeqEntry],
    max_seq_length: int,
    device: str | torch.device,
    eval_prompt: EvalPrompt,
    desc: str | None = None,
) -> Dataset:
    desc = desc or ""
    source_texts = [f"{d.question.lstrip()}\n{d.context.lstrip()}" for d in data]
    target_texts = [d.answer for d in data]
    eval_inputs = [eval_prompt.get_eval_input(d) for d in data]

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
        eval_inputs=eval_inputs,
        device=device,
    )


class Seq2SeqDatasetEntry(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    id: str
    answer: str
    context: str
    eval_inputs: str


class Seq2SeqDatasetSeries(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    id: list[str]
    answer: list[str]
    context: list[str]
    eval_inputs: list[str]


@dataclass
class Seq2SeqDataset(Dataset):
    input_tokens: Mapping[str, torch.Tensor]
    target_tokens: Mapping[str, torch.Tensor]
    data: list[Seq2SeqEntry]
    eval_inputs: list[str]
    device: str | torch.device

    def __len__(self) -> int:
        return self.input_tokens["input_ids"].size(0)

    def __getitem__(self, idx: int) -> Seq2SeqDatasetEntry:
        return {
            "input_ids": self.input_tokens["input_ids"][idx].to(self.device),
            "attention_mask": self.input_tokens["attention_mask"][idx].to(self.device),
            "labels": self.target_tokens["input_ids"][idx].to(self.device),
            "id": self.data[idx].id,
            "answer": self.data[idx].answer,
            "context": self.data[idx].context,
            "eval_inputs": self.eval_inputs[idx],
        }


def clean_response(s: str, eos_tag: str = "</s>") -> str:
    try:
        return s[: s.index(eos_tag)]
    except ValueError:
        return s


@dataclass
class BlockOutput:
    extract_txt: list[str]
    reward_labels: list[str]
    scores: list[torch.FloatTensor]


def generate_and_reward(
    extract_model: PreTrainedModel,
    inputs: torch.Tensor,
    original_sentence: list[str],
    args: Config,
    tokenizer: PreTrainedTokenizer,
    reward: Module,
    device: torch.device,
    true_class: str,
    label2id: dict[str, int],
    id2label: dict[int, str],
    rewrite: bool,
) -> BlockOutput:
    # Contrastive generation
    extract_response_tensor = extract_model.generate(
        inputs,
        max_length=args.max_generation_length,
        penalty_alpha=args.degeneration_penalty,
        top_k=args.contrastive_top_k,
    )
    extract_response_txt = text_decode(tokenizer, torch.tensor(extract_response_tensor))
    if rewrite:
        extract_response_txt_rw = [rewrite_extraction(s) for s in extract_response_txt]
    else:
        extract_response_txt_rw = extract_response_txt

    scores, reward_labels = run_reward(
        reward=reward,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        sentence1=original_sentence,
        sentence2=extract_response_txt_rw,
        device=device,
        true_class=true_class,
        label2id=label2id,
        id2label=id2label,
    )

    return BlockOutput(extract_response_txt, reward_labels, scores)


def evaluate(
    dataset: Dataset,
    extract: Module,
    extract_ref: PreTrainedModel,
    reward: Module,
    device: torch.device,
    args: Config,
    true_class: str,
    label2id: dict[str, int],
    id2label: dict[int, str],
    rewrite: bool,
    desc: str | None = None,
) -> list[dict[str, Any]]:
    desc = desc or "Evaluate"
    loader = DataLoader(dataset, batch_size=args.batch_size)
    extract.model.eval()

    output: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            inputs = batch["input_ids"].to(device)
            eval_inputs: list[str] = batch["eval_inputs"]

            rl_output = generate_and_reward(
                cast(PreTrainedModel, extract.model),
                inputs,
                eval_inputs,
                args,
                extract.tokenizer,
                reward,
                device,
                true_class,
                label2id,
                id2label,
                rewrite,
            )
            ref_output = generate_and_reward(
                extract_ref,
                inputs,
                eval_inputs,
                args,
                extract.tokenizer,
                reward,
                device,
                true_class,
                label2id,
                id2label,
                rewrite,
            )

            assert len(rl_output.reward_labels) == len(inputs)
            output.extend(
                {
                    "id": batch["id"][i],
                    "answer": batch["answer"][i],
                    "context": batch["context"][i],
                    "eval_inputs": batch["eval_inputs"][i],
                    "rl_extract_txt": rl_output.extract_txt[i],
                    "ref_extract_txt": ref_output.extract_txt[i],
                    "reward_label": rl_output.reward_labels[i],
                    "ref_reward_label": ref_output.reward_labels[i],
                    "scores": rl_output.scores[i].tolist(),
                }
                for i in range(len(inputs))
            )

    log_label_distribution(
        [d["reward_label"] for d in output], desc=f"{desc}: RL model"
    )
    log_label_distribution(
        [d["ref_reward_label"] for d in output], desc=f"{desc}: Ref model"
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
        reward_model=resolve(args.reward_model),
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


def decapitalise(s: str) -> str:
    if not s:
        return ""
    return s[0].lower() + s[1:]


def adjust_casing(ss: list[str]) -> list[str]:
    if not ss:
        return []
    return [ss[0].capitalize()] + [decapitalise(s) for s in ss[1:]]


def rewrite_extraction(extraction: str) -> str:
    """Rewrite structured extraction into natural language using a template.

    Simple case:
    > [Cause] This is a cause [Effect] This is an effect
    This is a cause, causes this is an effect

    Complex case:
    > [Cause] This cause 1 | This cause 2 [Effect] This effect 1 | This effect 2
    This cause 1, and this cause 2 causes this effect 1 and this effect 2
    """
    entities, _ = parse_instance(extraction)
    new_causes = ", and ".join(adjust_casing(entities["Cause"]))
    new_effects = ", and ".join(adjust_casing(entities["Effect"]))
    return f"{new_causes}, causes {new_effects}"


def get_labelling(reward_type: str) -> tuple[dict[str, int], dict[int, str], str]:
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

    return label2id, id2label, true_class


def main() -> None:
    args = simple_parsing.parse(Config, add_config_path_arg=True)
    args = resolve_arg_paths(args)

    run_name = args.run_name or datetime.now().isoformat()
    output_dir = args.output_dir / run_name
    output_dir.mkdir(exist_ok=True, parents=True)
    setup_logger(output_dir)

    git_commit = self_critique.util.get_current_commit_shorthash()
    logger.info(f"\n{args}")
    logger.info(f"output files: {output_dir}")
    logger.info(f"git commit: {git_commit}")

    (output_dir / "args.json").write_text(
        json.dumps(
            dataclasses.asdict(args) | {"git_commit": git_commit}, default=str, indent=2
        )
    )

    set_seed(args.seed)
    suppress_transformers_warnings()

    label2id, id2label, true_class = get_labelling(args.reward_type)

    if args.train_file is None:
        raise ValueError("Must provide a training file")

    extract = load_seq2seq_valuehead_model(args.extraction_model, train=True)
    extract_ref = create_reference_model(cast(PreTrainedModelWrapper, extract.model))
    reward = load_reward_model(args.reward_model, label2id=label2id, id2label=id2label)

    train_data = load_data(args.train_file, args.max_train_samples)
    train_dataset = preprocess_data(
        tokeniser=extract.tokenizer,
        data=train_data,
        max_seq_length=args.max_seq_length,
        device="cpu",
        eval_prompt=args.eval_prompt,
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
            eval_prompt=args.eval_prompt,
            desc="evaluation",
        )

    device: torch.device | None = None
    if args.do_train:
        extract, device = train_extract(
            extract=extract,
            extract_ref=cast(PreTrainedModel, extract_ref),
            reward=reward,
            train_dataset=train_dataset,
            args=args,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            eval_batches=args.eval_batches,
            true_class=true_class,
            label2id=label2id,
            id2label=id2label,
            rewrite=args.rewrite,
        )
        save_model(extract.model, extract.tokenizer, output_dir)

    if eval_dataset is not None and args.do_eval:
        device = device or self_critique.util.get_device()
        eval_result = evaluate(
            dataset=eval_dataset,
            extract=extract.to(device),
            reward=reward.to(device),
            extract_ref=cast(PreTrainedModel, extract_ref.to(device)),
            args=args,
            label2id=label2id,
            id2label=id2label,
            true_class=true_class,
            device=device,
            rewrite=args.rewrite,
        )
        save_results(result=eval_result, dir=output_dir, file_name="eval_result.json")

    logger.info(f"Reminder: output files are in {output_dir}")


if __name__ == "__main__":
    main()
