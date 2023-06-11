# pyright: basic
import dataclasses
import json
import sys
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

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
    device: str,
) -> list[dict[str, Any]]:
    device_pt = torch.device(device)
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
            device=device_pt,
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
    freqs = Counter(d["entailment_label"] for d in eval_output)
    return {
        "entailment_ratio": freqs["ENTAILMENT"] / len(eval_output),
        "class_frequencies": freqs,
    }


@dataclass
class ExtractDataset(Dataset):
    original: list[str] = dataclasses.field(init=False)
    extracted: list[str] = dataclasses.field(init=False)
    gold: list[str] = dataclasses.field(init=False)

    def __init__(self, data: list[dict[str, Any]], max_samples: int | None) -> None:
        data = data[:max_samples]
        self.original = [d["input"] for d in data]
        self.extracted = [d["output"] for d in data]
        self.gold = [d["gold"] for d in data]

    def __len__(self) -> int:
        return len(self.original)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "original": self.original[idx],
            "extracted": self.extracted[idx],
            "gold": self.gold[idx],
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

    data = json.loads(args.eval_file.read_text())
    dataset = ExtractDataset(data, args.max_samples)

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


if __name__ == "__main__" and not hasattr(sys, "ps1"):
    main()
