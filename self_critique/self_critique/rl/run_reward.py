# pyright: basic
import dataclasses
import itertools
import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import simple_parsing
import torch
import torch.backends.mps
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Suppress TensorFlow warnings. This must be done before importing transformers.
# Yes, it's an ugly hack, but it's necessary.
# ruff: noqa: E402
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from self_critique import metric
from self_critique.metric.fgcr_metric_cls import parse_instance
from self_critique.rl.extract_train import (
    Module,
    save_results,
    setup_logger,
    text_encode,
)
from self_critique.util import (
    get_current_commit,
    load_json,
    set_seed,
    suppress_transformers_warnings,
)

logger = logging.getLogger("self.reward")


# TODO: Use Reward ABC instead of this
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
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["token_type_ids"],
    )
    loader = DataLoader(dataset, batch_size=batch_size)

    scores: list[torch.FloatTensor] = []
    predictions: list[str] = []

    reward.model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids in loader:
            outputs = reward.model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
            )
            # Get logit for the reward class and use it as a score
            scores.extend(outputs.logits.select(dim=-1, index=label2id[true_class]))

            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(id2label[int(x.item())] for x in preds)

    return scores, predictions


class EvalPrompt(Enum):
    PASSAGE = "passage"
    GOLD = "gold"
    COMBINED = "combined"

    def get_eval_input(self, entry: "EvalEntry") -> str:
        match self:
            case EvalPrompt.PASSAGE:
                return entry.input
            case EvalPrompt.GOLD:
                return entry.gold
            case EvalPrompt.COMBINED:
                return f"{entry.input}\n{entry.gold}"


@dataclass
class Config:
    """Evaluate model output with a reward model.

    Expected input data format is JSON file with the following structure:
    [
        {
            "input": "Context or passage",
            "output": "[Cause] ... [Relation] cause [Effect] ..."
        },
        ...
    ]
    """

    # Path to the reward model
    model_path: str
    # Path to the data to be evaluated
    data_file: Path
    eval_prompt: EvalPrompt = EvalPrompt.COMBINED
    # Metrics mode: 'fcr', 'extract', 'maven', or 'maven_s'.
    # 'extract' is a synonym for 'fcr'.
    mode: str = "fcr"
    # Reward model type ("valid" or "entailment")
    reward_type: str = "valid"
    # Maximum sequence length
    max_seq_length: int = 400
    # Batch size
    batch_size: int = 32
    # Seed
    seed: int = 0
    # Max samples
    max_samples: int | None = None
    # Path to the output directory
    output_dir: Path = Path("output/reward")
    # Name of the directory inside the output dir for this run
    run_name: str | None = None
    # Device to use (default: automatic)
    device: str | None = None

    def __str__(self) -> str:
        config_lines = [">>>> CONFIGURATION"]
        for key, val in dataclasses.asdict(self).items():
            value = val.resolve() if isinstance(val, Path) else val
            config_lines.append(f"  {key}: {value}")
        return "\n".join(config_lines)


@dataclass
class EvalEntry:
    input: str
    output: str
    gold: str


def hash_entry(entry: EvalEntry) -> str:
    return str(hash((entry.input, entry.gold)))


def calculate_extract_metrics(data: list[EvalEntry], mode: str) -> dict[str, float]:
    return metric.get_metrics(
        mode,
        predictions=[
            {
                "id": hash_entry(entry),
                "prediction_text": entry.output,
            }
            for entry in data
        ],
        references=[
            {
                "id": hash_entry(entry),
                "answers": entry.gold,
                "question_type": parse_instance(entry.gold)[1] or "none",
            }
            for entry in data
        ],
    )


def load_data(file_path: Path, max_samples: int | None = None) -> list[EvalEntry]:
    data = load_json(file_path)
    return [
        EvalEntry(input=d["input"], output=d["output"], gold=d["gold"]) for d in data
    ][:max_samples]


T = TypeVar("T")


def batched(iterable: Iterable[T], n: int) -> Iterable[list[T]]:
    """batched('ABCDEFG', 3) --> ABC DEF G

    From https://docs.python.org/3/library/itertools.html#itertools.batched
    """
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield list(batch)


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


def evaluate(
    dataset: list[EvalEntry],
    reward: Module,
    device: torch.device,
    args: Config,
    label_config: LabelConfig,
    batch_size: int,
    desc: str | None = None,
) -> list[dict[str, Any]]:
    desc = desc or "Evaluate"
    batches = list(batched(dataset, batch_size))

    output: list[dict[str, Any]] = []
    for batch in tqdm(batches, desc=desc):
        inputs = [x.input for x in batch]
        eval_inputs = [args.eval_prompt.get_eval_input(x) for x in batch]
        extractions = [x.output for x in batch]
        golds = [x.gold for x in batch]

        scores, reward_labels = run_reward(
            reward=reward,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            sentence1=eval_inputs,
            sentence2=extractions,
            device=device,
            true_class=label_config.true_class,
            label2id=label_config.label2id,
            id2label=label_config.id2label,
        )

        assert len(reward_labels) == len(inputs)
        output.extend(
            {
                "input": inputs[i],
                "gold": golds[i],
                "rl_extract_txt": extractions[i],
                "reward_label": reward_labels[i],
                "scores": scores[i].tolist(),
            }
            for i in range(len(inputs))
        )

    return output


def get_metrics(
    results: list[dict[str, Any]], true_class: str, mode: str
) -> dict[str, Any]:
    extract_metrics = calculate_extract_metrics(
        data=[
            EvalEntry(input=r["input"], output=r["rl_extract_txt"], gold=r["gold"])
            for r in results
        ],
        mode=mode,
    )
    return {
        "reward": sum(r["reward_label"] == true_class for r in results) / len(results),
        "classes": {
            class_: sum(r["reward_label"] == class_ for r in results)
            for class_ in {r["reward_label"] for r in results}
        },
    } | extract_metrics


def get_device() -> torch.device:
    "Returns MPS if available, CUDA if available, otherwise CPU device."
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return torch.device(device)


def load_reward_model(
    model_name_or_path: str, label_config: LabelConfig
) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    config = AutoConfig.from_pretrained(
        model_name_or_path, num_labels=len(label_config.label2id)
    )
    config.label2id = label_config.label2id
    config.id2label = label_config.id2label

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, config=config
    )
    return tokenizer, model


def main() -> None:
    args = simple_parsing.parse(Config, add_config_path_arg=True)
    if args.mode not in ["extract", "fcr", "maven", "maven_s"]:
        raise SystemExit(f"Invalid mode: {args.mode}")

    if args.run_name is not None:
        output_dir = args.output_dir / args.run_name
    else:
        output_dir = args.output_dir / datetime.now().isoformat()
    output_dir.mkdir(exist_ok=True, parents=True)

    setup_logger(logger, output_dir)

    git_commit = get_current_commit()
    logger.info(f"\n{args}")
    logger.info(f"Git commit: {git_commit}")
    logger.info(f"Output files: {output_dir}")

    (output_dir / "args.json").write_text(
        json.dumps(
            dataclasses.asdict(args) | {"git_commit": git_commit},
            default=str,
            indent=2,
        )
    )

    set_seed(args.seed)
    suppress_transformers_warnings()

    label_config = get_labelling(args.reward_type)

    device = torch.device(args.device or get_device())
    tokeniser, model = load_reward_model(args.model_path, label_config)
    reward = Module(model.to(device), tokeniser)

    dataset = load_data(args.data_file, args.max_samples)
    results = evaluate(
        dataset=dataset,
        reward=reward,
        device=device,
        args=args,
        label_config=label_config,
        batch_size=args.batch_size,
    )
    save_results(results, output_dir)

    metrics = get_metrics(results, label_config.true_class, args.mode)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info(f"\n>>>> METRICS\n{json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
