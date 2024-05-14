# annotation

Streamlit app to annotate data for the extraction evaluation evaluation model.

## How to run

1. Create a virtual environment and install the dependencies:

```bash
> python3 -m venv venv
> source venv/bin/activate
> pip install .
```

2. Get/create the necessary files. You'll need
- `config/config.yaml`: configuration file controlling everything
- A valid `data` directory
- `data/inputs/*.json`: the JSON files to be annotated
- `data/inputs/split_to_user.json`: mapping between the JSON files and the users. It
  should start with an empty mapping (i.e. file name -> null)
- A `data/answers` directory will be created to store the annotations

Note that the config file can go anywhere you want, but you need to set the
`ANNOTATION_CONFIG_PATH` environment variable to point to it. If unset, the default
path is `config/config.yaml`. The data paths can be set in the config file.

You can find an example configuration file in `config.yaml.example`.

3. Run the app:

```bash
> streamlit run --global.developmentMode False src/annotation/Start_Page.py
```

## Results

The annotation results are in the `data/answers` directory. There should be one file
per user in the study. It will be initiated as a copy of their input file, and store
the answers in the `answers` field. The answers are initially all `null`.

## Data Files

To generate the data for the study, you have to use two scripts:

1. `scripts/filter_data.py`: takes files that are model outputs and filters out the
   cases that can be automatically annotated using rules (exact matches or clauses that
   have no words overlapping).
  - It combines the input files into a single one, adding the file name to each example.
  - It creates two files: a version with all the entries plus the rule-based tags, and a
    version with only the entries that need to be annotated.
2. `scripts/split_files.py`: takes a file (the output of `filter_data.py`) and splits
   it into multiple files, one per user. It also creates the `split_to_user.json` file
   that maps files to users, but it starts with an empty mapping.
   - It also takes the percentage of examples that are shared between all users. These
     overlapping examples are used to calculate the inter-annotator agreement.

Example combining multiple model test files:

```bash
> python scripts/filter_data.py flan-t5-large_test.json gpt4_test.json gpt35_test.json \
    --output data/source/to_annotate.json
> python scripts/split_files.py data/source/to_annotate.json 5 \
    --data-output-dir data/inputs
```
