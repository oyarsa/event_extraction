# pyright: basic
import dataclasses
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import simple_parsing
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from self_critique.minimal.util import set_seed, supress_transformers_warnings
from self_critique.rl.extract_train import (
    LABELLER,
    Module,
    load_entailment_model,
    load_seq2seq_model,
    resolve,
    run_entailment,
    text_decode,
    text_encode,
)


@dataclass
class Config:
    "The name of the Casual LM model we wish to fine with PPO."

    # Extraction model name or path
    extraction_model: str
    # Reconstruct model name or path
    reconstruction_model: str
    # Entailment model name or path
    entailment_model: str
    # Path to evaluation data
    eval_file: Path
    # Reward model batch size
    batch_size: int = 256
    # Maximum sequence length for Seq2Seq model
    max_seq_length: int = 128
    # Fixed random seed
    seed: int = 0
    # Maximum number of samples used for evaluation
    max_samples: int | None = None
    # Max length for generated sequences from the Seq2Seq model
    max_generation_length: int = 128
    # Path to output directory where metrics, checkpoints and predictions will be saved
    output_dir: Path = Path("output")
    # Contrastive top-k used for reranking
    contrastive_top_k: int = 5
    # Contrastive degeneration penalty (alphe)
    degeneration_penalty: float = 0.5
    # Device for inference
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def resolve_arg_paths(args: Config) -> Config:
    return dataclasses.replace(
        args,
        extraction_model=resolve(args.extraction_model),
        reconstruction_model=resolve(args.reconstruction_model),
        entailment_model=resolve(args.entailment_model),
        eval_file=Path(resolve(args.eval_file)) if args.eval_file else None,
        output_dir=Path(resolve(args.output_dir)),
    )


def evaluate(
    dataset: Dataset,
    reconstruct: Module,
    entailment: Module,
    args: Config,
    device: torch.device,
) -> list[dict[str, Any]]:
    loader = DataLoader(dataset, batch_size=args.batch_size)

    output: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="Evaluate"):
        original_sentence = batch["original"]

        extract_response_tokens = text_encode(
            reconstruct.tokenizer, args.max_seq_length, batch["extracted"]
        )
        reconstruct_response_tensor = reconstruct.model.generate(
            extract_response_tokens["input_ids"].to(device),
            num_beams=reconstruct.model.config.num_beams,
            max_length=args.max_generation_length,
        )
        reconstruct_response_txt = text_decode(
            reconstruct.tokenizer, reconstruct_response_tensor
        )

        entailment_labels = run_entailment(
            entailment=entailment,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            labeller=LABELLER,
            sentence1=original_sentence,
            sentence2=reconstruct_response_txt,
            device=device,
        )

        output.extend(
            {
                "id": batch["id"][i],
                "original": batch["original"][i],
                "answers": batch["answers"][i],
                "question_type": batch["question_type"][i],
                "context": batch["context"][i],
                "extracted": batch["extracted"]["i"],
                "rl_reconstruct_txt": reconstruct_response_txt[i],
                "entailment_label": entailment_labels[i],
            }
            for i in range(len(entailment_labels))
        )
    return output


def get_eval_result(eval_output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "entailment_ratio": sum(
            d["entailment_label"] == "ENTAILMENT" for d in eval_output
        )
        / len(eval_output),
        "class_frequencies": Counter(d["question_type"] for d in eval_output).items(),
    }


def main() -> None:
    args = simple_parsing.parse(Config, add_config_path_arg=True)
    args = resolve_arg_paths(args)

    output_dir = args.output_dir / datetime.now().isoformat()
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n{args}")
    print(f"output files: {output_dir}")

    (output_dir / "args.json").write_text(
        json.dumps(dataclasses.asdict(args), default=str, indent=2)
    )

    set_seed(args.seed)
    supress_transformers_warnings()

    reconstruction_model = load_seq2seq_model(args.reconstruction_model, train=False)
    entailment_model = load_entailment_model(args.entailment_model, LABELLER)
    # TODO: load data as just text passage and answer
    data = load_data(args.eval_file, args.max_samples)
    # There might not be a preprocess function
    dataset = preprocess_data(
        data, reconstruction_model.tokenizer, entailment_model.tokenizer, device="cpu"
    )

    output = evaluate(
        dataset=dataset,
        reconstruct=reconstruction_model,
        entailment=entailment_model,
        args=args,
        device=args.device,
    )
    eval_result = get_eval_result(output)

    result = {
        "output": output,
        "result": eval_result,
    }
    (output_dir / "eval_result.json").write_text(json.dumps(result))
