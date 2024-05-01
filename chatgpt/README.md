# ChatGPT scripts

Scripts:
- `extraction.py`: extract cause, effect and relations from text.
	- Uses the same data format as the `genqa_joint` model
	- Uses some instances as demonstration examples
- `contradiction_sentence.py`: generates a simple contradiction to the input.
  No examples.
- `contradiction_structured.py`: creates a contradiction from structured input.
	- The contradiction is obtained by swapping cause and effect
	- Uses the same data format as the reconstruct model
	- Uses some instances as demonstration examples

Run `python <script>.py -h` for options.

For convenience, there's a `run_extraction.fish` script that sets up common
options. There's also `experiment_extraction.fish` that runs the extraction
script through various combinations.
Requires the [Fish shell](https://fishshell.com/).

Tools used:
- pyright (basic)
- ruff (lint and format)
- sourcery

The scripts assume a specific location for the python binary. Set up the
virtualenv with:

```sh
uv venv
source .venv/bin/activate.fish # remove .fish if $SHELL is bash/zsh
uv pip install -r requirements.txt
```

Use [uv](https://github.com/astral-sh/uv) because it's a lot faster. Standard Python
venv/pip works too.

The data is already preprocessed in the `data` folder.

## Main GPT extraction results

The are in the `output/main-results` folder.

The real output file is `output.tags.json` and the real metrics one is
`output.tags.metrics.json`.

The other files are the original ones, but they used the LINES format, which is better
for GPT but worse for the rest of the project, which uses TAGS. I preserved them, but
gzipped them to save space.

They were converted from LINES to TAGS with the following command:

```fish
chatgpt/scripts/convert_lines_to_tags.py output.json > output.tags.json
```

The metrics were calculated using the following command:

```fish
self_critique/scripts/eval_std.py output.tags.json
```

Both of these commands were run in the root of the project and require the venv from
`self_critique` to be active.

Git commit from these runs: ce8f008
