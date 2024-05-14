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
