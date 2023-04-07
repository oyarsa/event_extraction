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
- mypy (strict)
- ruff 
- black

The scripts assume a specific location for the python binary. Set up the
virtualenv with:

```sh
python -m venv .venv
pip install -r requirements.txt
```

The data is already preprocessed in the `data` folder.

