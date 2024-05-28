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
import os
import warnings
from abc import ABC
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict, cast

import simple_parsing
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Suppress TensorFlow warnings. This must be done before importing transformers.
# Yes, it's an ugly hack, but it's necessary.
# ruff: noqa: E402
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
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

from self_critique.metric import fgcr_metric_cls
from self_critique.util import (
    get_current_commit,
    get_device,
    report_gpu_memory,
    resolve_path,
    save_model,
    set_seed,
    setup_logger,
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

    def get_answer(self, item: str) -> str:
        match self:
            case EvalPrompt.PASSAGE:
                raise ValueError("Cannot get answer for PASSAGE prompt.")
            case EvalPrompt.GOLD:
                return item
            case EvalPrompt.COMBINED:
                return item.splitlines()[1]


class EvaluatorType(Enum):
    REWARD = "reward"
    F1 = "f1"
    SENTENCE_TRANSFORMER = "sentence_transformer"


@dataclass
class Config:
    "The name of the Casual LM model we wish to fine with PPO."

    # Path to training data
    train_file: Path
    # Extraction model name or path
    extraction_model: str
    # Evaluator type
    evaluator: EvaluatorType = EvaluatorType.REWARD
    # Reward model name or path. Used by both model-based and SentenceTransformer
    # evaluators
    reward_model: str | None = None
    # Threshold used by SentenceTransformer and F1 evaluator to determine the class.
    # See the respective classes documentation for more information: `F1Evaluator` and
    # `SentenceTransformerEvaluator`.
    # Note that the the thresholds are optimum at the time of writing this code and
    # may need to be adjusted for different datasets.
    evaluator_threshold: float = 0.9656984508037567
    # Learning rate
    learning_rate: float = 1e-5
    # PPO minibatch size (inner PPO loop)
    ppo_minibatch_size: int = 16
    # PPO batch size (outer training loop)
    ppo_batch_size: int = 256
    # Reward model batch size
    reward_batch_size: int = 256
    # Epochs
    num_epochs: int = 1
    # Gradient accumulation steps
    gradient_accumulation_steps: int = 1
    # Maximum sequence length for Seq2Seq model
    max_input_seq_length: int = 128
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
    # Generation top-k used for reranking
    generation_top_k: int | None = None
    # Contrastive degeneration penalty (alphe)
    degeneration_penalty: float | None = None
    # Generation top-p used for selecting tokens
    generation_top_p: float | None = None
    # Whether to sample during generation
    generation_do_sample: bool = False
    # Number of beams for Beam Search
    generation_num_beams: int = 1
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
    # Evaluate the model every N batches (waiting for the whole epoch takes too long)
    eval_batches: int = 100
    # Reward type: 'entailment' or 'valid'
    reward_type: str = "entailment"
    # Whether to perform training
    do_train: bool = True
    # Whether to perform evaluation
    do_eval: bool = True
    # Which prompt to use
    eval_prompt: EvalPrompt = EvalPrompt.COMBINED
    # Max sequence length for the reward model
    max_reward_seq_length: int = 400
    # Use reward scaling for PPO
    use_reward_scaling: bool = False
    # Use reward normalization for PPO
    use_reward_norm: bool = False

    def __init__(self, **kwargs: Any) -> None:
        "Ignore unknown arguments"
        for f in dataclasses.fields(self):
            if f.name in kwargs:
                setattr(self, f.name, kwargs[f.name])

    def __str__(self) -> str:
        config_lines = [">>>> CONFIGURATION"]
        for key, val in dataclasses.asdict(self).items():
            match val:
                case Path(path=path):
                    value = path.resolve()
                case EvalPrompt(name=name):
                    value = name
                case EvaluatorType(name=name):
                    value = name
                case _:
                    value = val
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


def load_seq2seq_model(model_name: str, max_seq_length: str, *, train: bool) -> Module:
    model_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, config=model_config
    ).train(train)
    tokeniser = AutoTokenizer.from_pretrained(
        model_name, model_max_length=max_seq_length
    )
    model.resize_token_embeddings(len(tokeniser))
    return Module(model, tokeniser)


def text_decode(tokenizer: PreTrainedTokenizer, tensor: torch.Tensor) -> list[str]:
    return tokenizer.batch_decode(tensor, skip_special_tokens=True)


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
        truncation=True,
        return_tensors="pt",
        max_length=max_seq_length,
        return_token_type_ids=True,
    )


def log_tensorboard(
    writer: SummaryWriter,
    eval_output: list[dict[str, Any]],
    n_iter: int,
    true_class: str,
) -> None:
    ratio = sum(x["reward_label"] == true_class for x in eval_output) / len(eval_output)
    writer.add_scalar("eval/reward_ratio", ratio, n_iter)


class Evaluator(ABC):
    def run_reward(
        self,
        sentence1: list[str],
        sentence2: list[str],
        label2id: dict[str, int],
        id2label: dict[int, str],
        true_class: str,
    ) -> tuple[list[torch.Tensor], list[str]]:
        raise NotImplementedError


def train_extract(
    extract: Module,
    extract_ref: PreTrainedModel,
    evaluator: Evaluator,
    train_dataset: Dataset,
    args: Config,
    eval_dataset: Dataset | None,
    output_dir: Path,
    eval_batches: int,
    true_class: str,
    label2id: dict[str, int],
    id2label: dict[int, str],
    generation_kwargs: dict[str, Any],
) -> tuple[Module, torch.device]:  # sourcery skip: low-code-quality
    ppo_config = PPOConfig(
        model_name=args.extraction_model,
        learning_rate=args.learning_rate,
        mini_batch_size=args.ppo_minibatch_size,
        batch_size=args.ppo_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        adap_kl_ctrl=args.adaptive_kl_ctrl,
        kl_penalty=args.kl_penalty,
        init_kl_coef=args.init_kl_coef,
        log_with=args.log_with,
        use_score_scaling=args.use_reward_scaling,
        use_score_norm=args.use_reward_norm,
    )
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=extract.model,
        ref_model=cast(PreTrainedModelWrapper, extract_ref),
        tokenizer=extract.tokenizer,
        dataset=train_dataset,
        data_collator=data_collator,
    )
    if not ppo_trainer.dataloader:
        raise ValueError("Error initialising PPOTrainer dataloader.")

    best_model = copy.deepcopy(ppo_trainer.model)
    best_ratio = 0.0

    device = ppo_trainer.accelerator.device
    logger.info(f"Training on {device}")
    tb_writer = SummaryWriter(log_dir=output_dir / "tb")

    if isinstance(evaluator, RewardEvaluator):
        evaluator.set_device(device)

    if eval_dataset is not None:
        ref_desc = "Ref eval"
        ref_result = evaluate(
            dataset=eval_dataset,
            extract=Module(extract_ref, extract.tokenizer),
            evaluator=evaluator,
            args=args,
            device=device,
            true_class=true_class,
            label2id=label2id,
            id2label=id2label,
            generation_kwargs=generation_kwargs,
            desc=ref_desc,
        )
        save_results(
            result=ref_result,
            dir=output_dir,
            file_name="ref_result.json",
        )
        log_tensorboard(tb_writer, ref_result, -1, true_class)
        log_label_distribution(ref_result, desc=ref_desc)

    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(
            tqdm(ppo_trainer.dataloader, desc=f"Train ({epoch})")
        ):
            ppo_trainer.model.train()
            query_tensors = batch["input_ids"]

            response_tensors = ppo_trainer.generate(
                query_tensors,
                **generation_kwargs,
            )
            extract_response = text_decode(extract.tokenizer, response_tensors)

            rewards, labels = evaluator.run_reward(
                sentence1=batch["eval_inputs"],
                sentence2=extract_response,
                true_class=true_class,
                label2id=label2id,
                id2label=id2label,
            )

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            stats["metrics/reward_ratio"] = sum(
                label == true_class for label in labels
            ) / len(labels)
            log_batch = {
                "query": query_tensors,
                "response": response_tensors,
            }
            ppo_trainer.log_stats(stats, log_batch, rewards)

            # Mid-epoch evaluation
            if eval_dataset is not None and (batch_idx + 1) % eval_batches == 0:
                desc = f"Eval  ({epoch}.{batch_idx+1})"
                eval_result = evaluate(
                    dataset=eval_dataset,
                    extract=extract,
                    evaluator=evaluator,
                    args=args,
                    device=device,
                    true_class=true_class,
                    label2id=label2id,
                    id2label=id2label,
                    generation_kwargs=generation_kwargs,
                    desc=desc,
                )
                save_results(
                    result=eval_result,
                    dir=output_dir,
                    file_name=f"mini_eval_result_{epoch}.{batch_idx+1}.json",
                )

                log_label_distribution(ref_result, desc="Reference")
                log_label_distribution(eval_result, desc=desc)
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
            desc = f"Eval  ({epoch})"
            eval_result = evaluate(
                dataset=eval_dataset,
                extract=extract,
                evaluator=evaluator,
                args=args,
                device=device,
                true_class=true_class,
                label2id=label2id,
                id2label=id2label,
                generation_kwargs=generation_kwargs,
                desc=desc,
            )
            save_results(
                result=eval_result,
                dir=output_dir,
                file_name=f"eval_result_{epoch}.json",
            )
            log_label_distribution(ref_result, desc="Reference")
            log_label_distribution(eval_result, desc=desc)

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
    source_texts = [f"{d.question}\n{d.context.lstrip()}" for d in data]
    eval_inputs = [eval_prompt.get_eval_input(d) for d in data]

    model_inputs = tokeniser(
        source_texts,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    )

    return Seq2SeqDataset(
        input_tokens=model_inputs,
        data=data,
        eval_inputs=eval_inputs,
        device=device,
    )


class Seq2SeqDatasetEntry(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    id: str
    answer: str
    context: str
    eval_inputs: str


class Seq2SeqDatasetSeries(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    id: list[str]
    answer: list[str]
    context: list[str]
    eval_inputs: list[str]


@dataclass
class Seq2SeqDataset(Dataset):
    input_tokens: Mapping[str, torch.Tensor]
    data: list[Seq2SeqEntry]
    eval_inputs: list[str]
    device: str | torch.device

    def __len__(self) -> int:
        return self.input_tokens["input_ids"].size(0)

    def __getitem__(self, idx: int) -> Seq2SeqDatasetEntry:
        return {
            "input_ids": self.input_tokens["input_ids"][idx].to(self.device),
            "attention_mask": self.input_tokens["attention_mask"][idx].to(self.device),
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
    scores: list[torch.Tensor]


def generate_and_reward(
    extract: Module,
    evaluator: Evaluator,
    inputs: Mapping[str, torch.Tensor],
    original_sentence: list[str],
    device: torch.device,
    true_class: str,
    label2id: dict[str, int],
    id2label: dict[int, str],
    generation_kwargs: dict[str, Any],
) -> BlockOutput:
    extract_response_tensor = extract.model.generate(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        **generation_kwargs,
    )
    extract_response_txt = text_decode(extract.tokenizer, extract_response_tensor)

    scores, reward_labels = evaluator.run_reward(
        sentence1=original_sentence,
        sentence2=extract_response_txt,
        true_class=true_class,
        label2id=label2id,
        id2label=id2label,
    )

    return BlockOutput(extract_response_txt, reward_labels, scores)


def evaluate(
    dataset: Dataset,
    extract: Module,
    evaluator: Evaluator,
    device: torch.device,
    args: Config,
    true_class: str,
    label2id: dict[str, int],
    id2label: dict[int, str],
    generation_kwargs: dict[str, Any],
    desc: str = "Evaluate",
) -> list[dict[str, Any]]:
    loader = DataLoader(dataset, batch_size=args.reward_batch_size)
    extract.model.eval()

    output_data: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            eval_inputs: list[str] = batch["eval_inputs"]

            output = generate_and_reward(
                extract,
                evaluator,
                batch,
                eval_inputs,
                device,
                true_class,
                label2id,
                id2label,
                generation_kwargs,
            )

            assert len(output.reward_labels) == len(batch["input_ids"])
            output_data.extend(
                {
                    "id": batch["id"][i],
                    "answer": batch["answer"][i],
                    "context": batch["context"][i],
                    "eval_inputs": batch["eval_inputs"][i],
                    "output": output.extract_txt[i],
                    "reward_label": output.reward_labels[i],
                    "scores": output.scores[i].tolist(),
                }
                for i in range(len(output.reward_labels))
            )

    return output_data


def log_label_distribution(result: list[dict[str, Any]], desc: str) -> None:
    labels = [d["reward_label"] for d in result]
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
    resolved = resolve_path(path_or_name)
    if Path(resolved).exists():
        return str(resolved)
    return str(path_or_name)


def resolve_arg_paths(args: Config) -> Config:
    return dataclasses.replace(
        args,
        extraction_model=resolve(args.extraction_model),
        reward_model=resolve(args.reward_model) if args.reward_model else None,
        train_file=Path(resolve(args.train_file)),
        eval_file=Path(resolve(args.eval_file)) if args.eval_file else None,
        output_dir=Path(resolve(args.output_dir)),
    )


def decapitalise(s: str) -> str:
    if not s:
        return ""
    return s[0].lower() + s[1:]


def adjust_casing(ss: list[str]) -> list[str]:
    if not ss:
        return []
    return [ss[0].capitalize()] + [decapitalise(s) for s in ss[1:]]


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


# TODO: Add thresholds for F1 and SentenceTransformer
class F1Evaluator(Evaluator):
    """Evaluator using macro-average token-F1 score.

    The F1 score is used the score. The class is obtained by thresholding the F1 score.

    The best known thresholds depend on the dataset:
    - FCR: 0.8133333333333334
    - FinCausal: 0.8210526315789475
    """

    def __init__(self, threshold: float, prompt: EvalPrompt) -> None:
        self.threshold = threshold
        self.prompt = prompt

    def run_reward(
        self,
        sentence1: list[str],
        sentence2: list[str],
        label2id: dict[str, int],
        id2label: dict[int, str],
        true_class: str,
    ) -> tuple[list[torch.Tensor], list[str]]:
        sentence1 = [self.prompt.get_answer(s) for s in sentence1]

        f1_scores = [calc_f1(s1, s2) for s1, s2 in zip(sentence1, sentence2)]
        scores = [torch.tensor(x) for x in f1_scores]
        predictions = [id2label[int(x >= self.threshold)] for x in f1_scores]

        return scores, predictions


def calc_f1(sentence1: str, sentence2: str) -> float:
    entities1, _ = fgcr_metric_cls.parse_instance(sentence1)
    entities2, _ = fgcr_metric_cls.parse_instance(sentence2)

    f1_cause = calc_f1_sentence(entities1["Cause"], entities2["Cause"])
    f1_effect = calc_f1_sentence(entities1["Effect"], entities2["Effect"])

    return (f1_cause + f1_effect) / 2


def calc_f1_sentence(gold: list[str], pred: list[str]) -> float:
    gold_toks = fgcr_metric_cls.get_tokens(" ".join(gold))
    pred_toks = fgcr_metric_cls.get_tokens(" ".join(pred))

    if not gold_toks or not pred_toks:
        return 0

    common = Counter(gold_toks) & Counter(pred_toks)
    precision = sum(common.values()) / len(pred_toks)
    recall = sum(common.values()) / len(gold_toks)

    if precision + recall != 0:
        return (2 * precision * recall) / (precision + recall)
    else:
        return 0


class SentenceTransformerEvaluator(Evaluator):
    """Evaluator using SentenceTransformer embeddings and cosine similarity.

    The normalised cosine similarity is used as the score. The class is obtained by
    thresholding the similarity score.

    The best known thresholds are for the model all-MiniLM-L6-v2 and depend on dataset:
    - FCR: 0.9656984508037567
    - FinCausal: 1.4834709167480469
    """

    def __init__(
        self, model: SentenceTransformer, threshold: float, prompt: EvalPrompt
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.prompt = prompt

    def run_reward(
        self,
        sentence1: list[str],
        sentence2: list[str],
        label2id: dict[str, int],
        id2label: dict[int, str],
        true_class: str,
    ) -> tuple[list[torch.Tensor], list[str]]:
        sentence1 = [self.prompt.get_answer(s) for s in sentence1]

        sent1_emb = self.model.encode(sentence1)
        sent2_emb = self.model.encode(sentence2)

        cosine_sims = cosine_similarity(sent1_emb, sent2_emb)
        cosine_sim_norm = [float(sim) + 1 / 2 for sim in cosine_sims.diagonal()]

        scores = [torch.tensor(x) for x in cosine_sim_norm]
        predictions = [id2label[int(x >= self.threshold)] for x in cosine_sim_norm]

        return scores, predictions


class RewardEvaluator(Evaluator):
    def __init__(
        self,
        model: Module,
        max_seq_length: int,
        batch_size: int,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = device

    def set_device(self, device: torch.device) -> None:
        self.device = device

    def run_reward(
        self,
        sentence1: list[str],
        sentence2: list[str],
        label2id: dict[str, int],
        id2label: dict[int, str],
        true_class: str,
    ) -> tuple[list[torch.Tensor], list[str]]:
        if self.device is None:
            raise ValueError("Device must be set before running the evaluator.")

        inputs = text_encode(
            self.model.tokenizer, self.max_seq_length, sentence1, sentence2
        )
        dataset = TensorDataset(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"],
        )
        loader = DataLoader(dataset, batch_size=self.batch_size)

        scores: list[torch.Tensor] = []
        predictions: list[str] = []

        self.model.model.eval()
        with torch.no_grad():
            for input_ids, attention_mask, token_type_ids in loader:
                outputs = self.model.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    token_type_ids=token_type_ids.to(self.device),
                )
                # Get logit for the reward class and use it as a score
                scores.extend(outputs.logits.select(dim=-1, index=label2id[true_class]))

                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(id2label[int(x.item())] for x in preds)

        return scores, predictions


def suppress_warnings() -> None:
    suppress_transformers_warnings()
    # trl PPO warning on high KL divergence
    warnings.filterwarnings(
        "ignore",
        message="The average ratio of batch .* exceeds threshold .*. Skipping batch.",
        category=UserWarning,
    )
    # accelerate warning about how the linux kernel 5.4 is too old, but it works fine
    logging.getLogger("accelerate").setLevel(logging.ERROR)


def main() -> None:
    args = simple_parsing.parse(Config, add_config_path_arg=True)
    args = resolve_arg_paths(args)

    run_name = args.run_name or datetime.now().isoformat()
    output_dir = args.output_dir / run_name
    output_dir.mkdir(exist_ok=True, parents=True)
    setup_logger(logger, output_dir)

    git_commit = get_current_commit()
    logger.info(f"\n{args}")
    logger.info(f"output files: {output_dir}")
    logger.info(f"git commit: {git_commit}")

    (output_dir / "args.json").write_text(
        json.dumps(
            dataclasses.asdict(args) | {"git_commit": git_commit},
            default=str,
            indent=2,
        )
    )

    set_seed(args.seed)
    suppress_warnings()

    label2id, id2label, true_class = get_labelling(args.reward_type)
    generation_kwargs = {
        "max_length": args.max_generation_length,
        "penalty_alpha": args.degeneration_penalty,
        "top_k": args.generation_top_k,
        "top_p": args.generation_top_p,
        "do_sample": args.generation_do_sample,
        "num_beams": args.generation_num_beams,
    }
    logger.info(f"Generation type: {log_generation_type(generation_kwargs)}")

    if args.train_file is None:
        raise ValueError("Must provide a training file")

    extract = load_seq2seq_valuehead_model(args.extraction_model, train=True)
    extract_ref = create_reference_model(cast(PreTrainedModelWrapper, extract.model))

    if args.reward_model is None:
        raise ValueError("Must provide a reward model when using a learned evaluator.")
    reward = load_reward_model(args.reward_model, label2id=label2id, id2label=id2label)

    match args.evaluator:
        case EvaluatorType.REWARD:
            evaluator = RewardEvaluator(
                model=reward,
                max_seq_length=args.max_reward_seq_length,
                batch_size=args.reward_batch_size,
                device=None,
            )
        case EvaluatorType.SENTENCE_TRANSFORMER:
            model = SentenceTransformer(args.reward_model)
            threshold = args.evaluator_threshold
            evaluator = SentenceTransformerEvaluator(model, threshold, args.eval_prompt)
        case EvaluatorType.F1:
            threshold = args.evaluator_threshold
            evaluator = F1Evaluator(threshold, args.eval_prompt)

    train_data = load_data(args.train_file, args.max_train_samples)
    train_dataset = preprocess_data(
        tokeniser=extract.tokenizer,
        data=train_data,
        max_seq_length=args.max_input_seq_length,
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
            max_seq_length=args.max_input_seq_length,
            device="cpu",
            eval_prompt=args.eval_prompt,
            desc="evaluation",
        )

    device: torch.device | None = None
    if args.do_train:
        extract, device = train_extract(
            extract=extract,
            extract_ref=cast(PreTrainedModel, extract_ref),
            evaluator=evaluator,
            train_dataset=train_dataset,
            args=args,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            eval_batches=args.eval_batches,
            true_class=true_class,
            label2id=label2id,
            id2label=id2label,
            generation_kwargs=generation_kwargs,
        )
        save_model(extract.model, extract.tokenizer, output_dir)

    if eval_dataset is not None and args.do_eval:
        device = device or get_device()
        if isinstance(evaluator, RewardEvaluator):
            evaluator.set_device(device)

        desc = "Final evaluation"
        eval_result = evaluate(
            dataset=eval_dataset,
            extract=extract.to(device),
            evaluator=evaluator,
            args=args,
            label2id=label2id,
            id2label=id2label,
            true_class=true_class,
            device=device,
            generation_kwargs=generation_kwargs,
            desc=desc,
        )
        log_label_distribution(eval_result, desc=desc)
        save_results(result=eval_result, dir=output_dir, file_name="eval_result.json")

    logger.info(f"Reminder: output files are in {output_dir}")


def log_generation_type(args: dict[str, Any]) -> str:
    """Figure out which form of generation will be used based on the arguments.

    From the transformers documentation[1]:

    - greedy decoding if num_beams=1 and do_sample=False
    - contrastive search if penalty_alpha>0. and top_k>1
    - multinomial sampling if num_beams=1 and do_sample=True
    - beam-search decoding if num_beams>1 and do_sample=False
    - beam-search multinomial sampling if num_beams>1 and do_sample=True

    [1] https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig
    """
    # sourcery skip: hoist-repeated-if-condition, remove-redundant-if
    if (
        args["penalty_alpha"]
        and args["top_k"]
        and args["penalty_alpha"] > 0.0
        and args["top_k"] > 1
    ):
        return "contrastive search"

    if args["num_beams"] == 1:
        if args["do_sample"]:
            return "multinomial sampling"
        else:
            return "greedy decoding"
    if args["num_beams"] > 1:
        if args["do_sample"]:
            return "beam-search multinomial sampling"
        else:
            return "beam-search decoding"
    return "unknown generation type"


if __name__ == "__main__":
    report_gpu_memory(main, logger)
