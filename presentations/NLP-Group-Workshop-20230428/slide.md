---
title: Extracting cause and effect from sentences
author: Italo Luis da Silva
date: 2023-03-28
theme: Boadilla
header-includes:
    - \usepackage{svg}
---

# Problem statement

<!-- TODO -->
- Extracting cause and effect from sentences
- Determine the type of relation between the cause and the effect
    - Causes: example
    - Enables: example
    - Prevents: example

# Examples: one cause, one effect

<!-- TODO -->
Sentence: "The car crashed into a tree because the driver was texting."

Cause: "the driver was texting"

Effect: "the car crashed into a tree"

# Examples: one cause, many effects

<!-- TODO -->
- One cause, multiple effects

# Examples: many causes, one effect

<!-- TODO -->
- Multiple causes, one effect

# Examples: many causes, many effects

<!-- TODO -->
- Multiple causes, multiple effects

# Dataset statistics

## Extraction task

| Split | # Examples | # Relations | # Causes | # Effects |
|-------|------------|-------------|----------|-----------|
| Dev   | 2482       | 3226        | 3224     | 3238      |
| Train | 19892      | 25938       | 26174    | 26121     |
| Test  | 2433       | 3045        | 3065     | 3062      |

## Classification task

| Split | # Examples | % Causes | % Prevents | % Enables |
|-------|------------|----------|------------|-----------|
| Dev   | 2482       | 63.78%   | 5.40%      | 30.82%    |
| Train | 19892      | 63.05%   | 5.90%      | 31.05%    |
| Test  | 2433       | 64.00%   | 5.38%      | 30.62%    |

# Some results

| Model name                | Token F1 |  EM        | Class Acc | Class F1 |
|---------------------------|----------|------------|-----------|----------|
| GenQA (extraction)        | 81.09%   | 48.14%     |    -      |    -     |
| GenQA (joint)             | 79.47%   | 52.16%     | 71.19%    | 54.08%   |
| Sequence Labelling        | 73.23%   | 22.95%     |    -      |    -     |
| BERT (extraction)[^1]     | 84.37%   | 51.48%     |    -      |    -     |
| BERT (classification)^1^  |    -     |   -        | 70.43%    | 71.74%   |
| BERT (joint)^1^           |    -     | 21.21%     |    -      |    -     |

[^1]: From the original paper

# Problem with EM evaluation

<!-- TODO -->
- Example of data annotation vs model output

# Proposed solution: RL framework

![RL framework](rl.pdf){height=70%}

# Next steps

- Execution of the RL framework
- Experiments to find out the best algorithm, setup, rewards, etc.

# Current issues

- Model size in memory
- How to best train this?
