# event_extraction

Causal event extraction from text

## Setup

1. Setup a virtual environment
2. Install dependencies with `./setup`
Requires `uv` or `pip-tools`.

I recommend using [uv](https://github.com/astral-sh/uv) for managing dependencies
because it's a lot faster, but it should work with python's built-in `venv` and
`pip-tools` as well.

## Projects

Each project is a separate directory with its own README.md file. They all use PDM to
manage dependencies, but we use pip-tools for a repository-wide environment.

- `agreement`: Calculate agreement between LLM judges and human evaluation
- `chatgpt`: Use GPT OpenAI API to extract causal events
- `data`: Datasets for the project, including processed data
- `preprocess`: Scripts to preprocess data for the different models
- `self_critique`: LLM-based extraction, supervised and RL training
- `sequence_labelling`: BIO labelling-based model for causal event extraction
