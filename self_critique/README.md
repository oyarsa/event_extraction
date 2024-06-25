<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Note:
The code is based on Huggingface transformer examples: [`question-answering`](https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering)

# Instruction:

## Data preprocess:

```sh
# From the repository root:
python preprocess/genqa_joint.py
```

## Train:

This subproject uses PDM to manage dependencies and virtual environments. To
install PDM, run: check [their
website](https://pdm.fming.dev/latest/#installation).

Python 3.10+ is required. I recommend using
[pyenv](https://github.com/pyenv/pyenv#installation) to manage Python versions.

After that, run the following commands to install dependencies and create a
virtual environment:

```sh
pdm install
source .venv/bin/activate
```

Train joint extraction and classification model:
```sh
python self_critique/minimal/seq2seq.py --config config/extraction.json
```

Train Reinforcement Learning model:
```sh
cd self_critique/rl
python extract_train.py --config config.json
```
