# BERT-Sequence-Labeling

This repostiory integrates [HuggingFaces](https://github.com/huggingface)'s models in an end-to-end pipeline for sequence labeling. [Here](https://huggingface.co/transformers/pretrained_models.html)
is a complete list of the available models.

## Install

```sh
$ git clone https://github.com/avramandrei/BERT-Sequence-Labeling.git
$ cd BERT-Sequence-Labeling
$ conda create -n bert-sl python=3.10
$ conda activate bert-sl
$ pip install -r requirements.txt
```

## Input Format

The files used for training, validation and testing must be in the following format:
- Each line contains the token and the label separated by space
- Each document or sentence is separated by a blank line

The labels can be whatever you want.

```
This O
is O
the O
first O
sentence B-Label1
. I-Label1

This B-Label2
is I-Label2
the O
second O
```

There can be other columns in the file, and the token-label order can be switched. All
that matters is that you use the correct column indices (starting from 0) when calling
the scripts, and that you keep the sentences or documents separated by a blank line.

## Training

To train a model, use the `train.py` script. This will start training a model that will predict the labels of the column specified by the `[predict_column]` argument.

```
python3 train.py [path_train_file] [path_dev_file] [tokens_column] [predict_column] [lang_model_name]
```

## Inference

To predict new values, use the `predict.py` script. This will create a new file by replacing the predicted column of the test file with the predicted values.

```
python3 predict.py [path_test_file] [model_path] [tokens_column] [predict_column] [lang_model_name]
```

## Results

#### FGCR

See `data/fgcr` for the data and attribution.

| model | macro_f1 |
| --- | --- |
| bert-base-cased | 73.23 |
| roberta-base | 74.1 |
