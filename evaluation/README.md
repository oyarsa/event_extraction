# Evaluation

This directory contains scripts to evaluate the performance of the models, especially
LLM-based evaluators.

## Train and Evaluate

The `train_and_eval.py` script is a wrapper to train and evaluate a classifier. It
trains the classifier using the provided classification script and configuration file,
and then evaluates the classifier using the provided evaluation script.

The default evaluation script is `agreement/calc.py`, which calculates the agreement
between the human and model results along with other metrics.

## Classifier Specification

Defines how the classifier scripts should be behave to be used with the
`train_and_eval.py` script.

### Input

CLI usage:

```bash
python classifier.py \
    --config $config \
    --output_path $output_path \
    --output_name $output_name \
```

Where:
- `$config`: Path to the configuration file.
- `$output_path`: Path to the output directory.
- `$output_name`: Name of the directory for the run, to be created inside
  `$output_path`.

The configuration file format doesn't actually matter, as long as the classifier
script can read it.

### Output

The script must generate a file with the results in the path
`$output_path/$output_name/$output_file`. This will be read by the evaluation script (by
default, `agreement/calc.py`)

The file must have the following format:

JSON file with a list of objects, each with the following fields:
- `'gold'`: integer, 0 or 1. *Human* evaluation result for the item.
- `'pred'`: integer, 0 or 1. *Model* evaluation result for the item.

### Result

The `train_and_eval.py` script will print the following metrics:
- Agreement: Percentage of items where the human and model results agree.
- Krippendorff's alpha: A measure of agreement between multiple raters.
- Cohen's kappa: A measure of agreement between two raters.
- Spearmann's correlation: A measure of the strength and direction of the
    relationship between the human and model results.

### Observations

This only defines the input/output for a binary classifier scripts. For other types, a
new specification should be written.
