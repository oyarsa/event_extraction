# pyright: basic
import dataclasses
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import simple_parsing
import torch
import torch.backends.mps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput

import self_critique.rl.extract_train as ex
from self_critique.minimal.util import set_seed, suppress_transformers_warnings
from self_critique.util import get_device

logger = logging.getLogger("self.ensemble")


@dataclasses.dataclass
class ModelConfig:
    # Path to model, or name if taken from HuggingFace model hub
    path_or_name: str
    # Model type (supervised or rl)
    model_type: str


@dataclasses.dataclass
class Config:
    "Ensemble and inner models configuration"
    # Path to data to be evaluated in the ensemble
    data_file: Path
    model_1: ModelConfig
    model_2: ModelConfig
    batch_size: int = 32
    max_seq_length: int = 128
    seed: int = 0
    max_samples: int | None = None
    output_dir: Path = Path("output/ensemble")
    # Contrastive top-k used for reranking
    contrastive_top_k: int = 5
    # Contrastive degeneration penalty (alphe)
    degeneration_penalty: float = 0.5

    def __init__(self, **kwargs: Any) -> None:
        "Ignore unknown arguments"
        for f in dataclasses.fields(self):
            if f.name in kwargs:
                setattr(self, f.name, kwargs[f.name])

    def __str__(self) -> str:
        config_lines = [">>>> CONFIGURATION"]
        for key, val in dataclasses.asdict(self).items():
            value = val.resolve() if isinstance(val, Path) else val
            if isinstance(value, dict):
                config_lines.append(f"  {key}:")
                config_lines.extend(f"    {k}: {v}" for k, v in value.items())
            else:
                config_lines.append(f"  {key}: {value}")
        return "\n".join(config_lines)


def resolve_arg_paths(config: Config) -> Config:
    return dataclasses.replace(
        config,
        data_file=Path(ex.resolve(config.data_file)),
        model_1=dataclasses.replace(
            config.model_1, path_or_name=ex.resolve(config.model_1.path_or_name)
        ),
        model_2=dataclasses.replace(
            config.model_2, path_or_name=ex.resolve(config.model_2.path_or_name)
        ),
        output_dir=Path(ex.resolve(config.output_dir)),
    )


def load_model(config: ModelConfig, device: torch.device) -> ex.Module:
    if config.model_type == "rl":
        module = ex.load_seq2seq_valuehead_model(config.path_or_name, train=False)
    elif config.model_type == "supervised":
        module = ex.load_seq2seq_model(config.path_or_name, train=False)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    module.model = module.model.to(device)
    return module


def get_logit(output: Any) -> torch.Tensor:
    if isinstance(output, Seq2SeqLMOutput):
        return output.logits
    elif isinstance(output, tuple) and len(output) == 3:
        return output[0]
    else:
        raise ValueError(f"Unexpected outputput type: {type(output)}")


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch: Any,
    max_length: int,
    penalty_alpha: float,
    top_k: int,
) -> tuple[list[str], list[float]]:
    with torch.no_grad():
        inputs = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        responses = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_length=max_length,
            penalty_alpha=penalty_alpha,
            top_k=top_k,
        )
        output = model(
            input_ids=inputs, attention_mask=attention_mask, labels=batch["labels"]
        )
        logits = get_logit(output)

    txt_responses = ex.text_decode(tokenizer, responses)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(
        log_probs, 2, batch["input_ids"].unsqueeze(-1)
    ).squeeze(-1)
    mean_log_probs = token_log_probs.mean(dim=-1).tolist()

    return txt_responses, mean_log_probs


def evaluate_ensemble(
    model1: PreTrainedModel,
    model2: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int,
    max_length: int,
    contrastive_top_k: int,
    degeneration_penalty: float,
    desc: str | None = None,
) -> list[dict[str, Any]]:
    model1.eval()
    model2.eval()

    desc = desc or "Evaluating ensemble"
    loader = DataLoader(dataset, batch_size=batch_size)

    outputs: list[tuple[str, str]] = []
    mean_logprobs: list[tuple[float, float]] = []
    data: list[dict[str, str]] = []

    for batch in tqdm(loader, desc=desc):
        output1, mean_logprob1 = generate(
            model1,
            tokenizer,
            batch,
            max_length,
            degeneration_penalty,
            contrastive_top_k,
        )
        output2, mean_logprob2 = generate(
            model2,
            tokenizer,
            batch,
            max_length,
            degeneration_penalty,
            contrastive_top_k,
        )

        outputs.extend(zip(output1, output2))
        mean_logprobs.extend(zip(mean_logprob1, mean_logprob2))
        data.extend(
            {key: batch[key][i] for key in ["id", "answers", "context"]}
            for i in range(len(batch["id"]))
        )

    return [
        {
            "output1": output1,
            "output2": output2,
            "mean_logprob1": mean_logprob1,
            "mean_logprob2": mean_logprob2,
            "text_chosen": output1 if mean_logprob1 >= mean_logprob2 else output2,
            "model_chosen": 1 if mean_logprob1 >= mean_logprob2 else 2,
            **d,
        }
        for (output1, output2), (mean_logprob1, mean_logprob2), d in zip(
            outputs, mean_logprobs, data
        )
    ]


def main() -> None:
    args = simple_parsing.parse(Config, add_config_path_arg=True)
    args = resolve_arg_paths(args)

    output_dir = args.output_dir / datetime.now().isoformat()
    output_dir.mkdir(exist_ok=True, parents=True)
    ex.setup_logger(output_dir)

    logger.info(f"\n{args}")
    logger.info(f"output files: {output_dir}")

    (output_dir / "args.json").write_text(
        json.dumps(dataclasses.asdict(args), default=str, indent=2)
    )

    set_seed(args.seed)
    suppress_transformers_warnings()

    device = get_device()
    model1 = load_model(args.model_1, device)
    model2 = load_model(args.model_2, device)

    assert type(model1.tokenizer) == type(  # noqa: E721
        model2.tokenizer
    ), "Models must be of the same type"

    data = ex.load_data(args.data_file, args.max_samples)
    dataset = ex.preprocess_data(
        model1.tokenizer, data, args.max_seq_length, device=device
    )
    result = evaluate_ensemble(
        model1.model,
        model2.model,
        model1.tokenizer,
        dataset,
        args.batch_size,
        args.max_seq_length,
        args.contrastive_top_k,
        args.degeneration_penalty,
    )
    (output_dir / "result.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
