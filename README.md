# event_extraction

Causal event extraction from text

## Setup

1. Clone the repository
2. Install the requirements
```fish
> uv venv
> source .venv/bin/activate.fish # or .venv/bin/activate on bash/zsh
> uv pip compile requirements.in -o requirements.txt
> uv pip sync requirements.txt
```

I recommend using [uv](https://github.com/astral-sh/uv) for managing dependencies
because it's a lot faster, but it should work with python's built-in `venv`, `pip` and
`pip-tools` as well.

## Projects

Each project is a separate directory with its own README.md file. They all use PDM to
manage dependencies, but we use pip-tools for a repository-wide environment.

- `agreement`: Calculate agreement between LLM judges and human evaluation
- `chatgpt`: Use GPT OpenAI API to extract causal events
- `data`: Datasets for the project, including processed data
- `error_analysis`: Analyze errors in the extraction LLM model
- `extractive_qa`: Span-based model for causal event extraction
- `gen_qa`: QA-based model for causal event extraction
- `preprocess`: Scripts to preprocess data for the different models
- `presentations`: Slides for presentations on the project
- `self_critique`: LLM-based extraction, supervised and RL training
- `sequence_labelling`: BIO labelling-based model for causal event extraction
