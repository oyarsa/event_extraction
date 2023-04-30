# pyright: basic
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class Config:
    """Configuration for the model training, evaluation and prediction."""

    # Model name from the HuggingFace model hub, or path to a local model saved
    # with `model.save_pretrained`.
    model_name_or_path: str
    # Directory to save the model, tokeniser, metrics and predictions
    output_dir: Path
    # Maximum length of the input sequence
    max_seq_length: int = 256
    # Number of beams to use for beam search
    generation_num_beams: int | None = None
    # Batch size for training
    per_device_train_batch_size: int = 32
    # Batch size for evaluation
    per_device_eval_batch_size: int = 32
    # Batch size for prediction
    per_device_test_batch_size: int = 32
    # Device to use
    device: str = "cuda:0"
    # Number of training epochs
    num_train_epochs: int = 20
    # Learning rate
    learning_rate: float = 5e-4
    # Path to the training data
    train_file: Path | None = None
    # Path to the validation data
    validation_file: Path | None = None
    # Path to the test data
    test_file: Path | None = None
    # Whether to run prediction at the end
    do_predict: bool = True
    # Whether to run evaluation
    do_eval: bool = True
    # Whether to run training
    do_train: bool = True
    # Maximum number of samples used for training
    max_train_samples: int | None = None
    # Maximum number of samples used for evaluation
    max_eval_samples: int | None = None
    # Maximum number of samples used for prediction
    max_predict_samples: int | None = None
    # Level for logging. Most messages are INFO.
    log_level: str = "info"
    # Load the best model by token F1 at the end of training
    load_best_model_at_end: bool = True
    # Early stopping patience
    early_stopping_patience: int = 5
    # Random seed for reproducibility
    seed: int = 0

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
