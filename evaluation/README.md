# Evaluation

This directory contains scripts to evaluate the performance of the models, especially
LLM-based evaluators.

## Getting Started

This project uses [PDM](https://pdm-project.org) for dependency management. To install
the dependencies, run the following command:

```bash
pdm install
```

This command will take some time to run, as it will resolve all dependencies on the
user's machine.

This is because the lockfile isn't commited to the repository because it's specific to
the platform, as we want it to be generated on the user's machine so PyTorch can
dynamically support CUDA, MPS, etc.

Before running the scripts, make sure to activate the virtual environment in `.venv`.
