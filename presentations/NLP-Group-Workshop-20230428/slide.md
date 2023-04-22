---
title: Extracting cause and effect from sentences
author: [Your Name]
date: \today
theme: Copenhagen
header-includes:
    - \usepackage{svg}
---

# Problem statement

- Extracting cause and effect from sentences
- Quick example

## Quick example

Sentence: "The car crashed into a tree because the driver was texting."

Cause: "the driver was texting"

Effect: "the car crashed into a tree"

# More examples

- One cause, multiple effects
- Multiple causes, one effect
- Multiple causes, multiple effects

# Statistics about the dataset

| Split | # Examples | 1:1 | 1:N | N:1 | N:N |
|-------|------------|-----|-----|-----|-----|
| Dev   |            |     |     |     |     |
| Train |            |     |     |     |     |
| Test  |            |     |     |     |     |

# Primary model results

| Model name         | Token F1 |  EM | Class Acc | Class F1 |
|--------------------|----------|-----|-----------|----------|
| GenQA (extraction) |          |     |           |          |
| GenQA (joint)      |          |     |           |          |
| Sequence Labelling |          |     |           |          |
| BERT (data paper)  |          |     |           |          |

# Problem with EM evaluation

- Example of data annotation vs model output

# Solution: RL framework

## The RL framework diagram

![RL framework](rl.pdf){height=70%}

# Next steps

- Execution of the RL framework
- Experiments to find out the best algorithm, setup, rewards, etc.

# Current issues

- Model size in memory
- How to best train this?
