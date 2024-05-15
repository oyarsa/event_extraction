# Evaluation errors from GPT-3.5

Files:
- `full_output.json`: full output from the GPT-3.5 script
- `results.json`: reshaped output from GPT-3.5, ran on 100 examples from
  `evaluation/data/evaluation/all_labelled.json`
- `metrics.json`: evaluation metrics (mainly agreement)
- `reproduction.json`: parameters used to run the script, including the command invoked,
  the hash of the input data and the git commit for the code
- `errors.json`: `results.json` filtered for when `.pred != .gold`
