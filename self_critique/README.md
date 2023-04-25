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
Data preprocess:

```sh
# From the repository root:
$ python preprocess/reconstruct.py
$ python preprocess/genqa_joint.py
$ python preprocess/entailment.py
```

For training, use the `genqa` environment (see the `requirements` folder).

Train joint extraction and classification model:
```sh
python run_seq2seq_qa.py config/genqa_joint.json
```

Train reconstruction model:
```
python run_seq2seq_qa.py config/reconstruct.json
```

Train text entailment model:
```
python run_classification.py config/entailment.json
```

