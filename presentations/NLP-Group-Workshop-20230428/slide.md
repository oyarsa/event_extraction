---
title: Extracting cause and effect from sentences
author: Italo Luis da Silva
date: 2023-03-28
institute: King's College London
theme: metropolis
colorlinks: true
---

# Problem statement

- Fine-grained causal reasoning (Yang et al., 2022)[^paper][^dataset]
- Extract cause and effect from the context
    - Spans of the context, not trigger words
- Classify the relation between cause and effect
    - **Cause**: sufficient and necessary condition
        - Cause is enough to make the effect happen
        - Must happen for effect to happen
    - **Enable**: sufficient but not necessary condition
        - Cause is enough to make the effect happen
        - Other conditions can also lead to the effect
    - **Prevent**: sufficient condition to stop the effect from happening
        - If the cause happens, the effect cannot happen

[^paper]:
    [Towards Fine-grained Causal Reasoning and QA](
    https://arxiv.org/abs/2204.07408)
[^dataset]:
    [github.com/YangLinyi/Fine-grained-Causal-Reasoning](
    https://github.com/YangLinyi/Fine-grained-Causal-Reasoning/)

# Example (part 1)

> \textcolor{teal}{The firm's gross margin is set to stabilize} as
\textcolor{red}{Harley refocuses its efforts on more profitable markets}, and
our base case assumes that it stabilizes around 32\% in 2029, helped by a more
measured approach to entering new markets.

- \textcolor{red}{Cause}~1~: Harley refocuses its efforts on more profitable markets
- \textcolor{teal}{Effect}~1~: The firm's gross margin is set to stabilize
- **Relation**~1~: cause

# Example (part 2)

There can be more than one relation in the context: \newline

> The firm's gross margin is set to stabilize as Harley refocuses its efforts on
more profitable markets, and our base case assumes that \textcolor{purple}{it
stabilizes around 32\% in 2029}, helped by \textcolor{olive}{a more measured
approach to entering new markets}.

- \textcolor{olive}{Cause}~2~: a more measured approach to entering new markets
- \textcolor{purple}{Effect}~2~: it stabilizes around 32\% in 2029
- **Relation**~2~: enable

# Dataset statistics

## Extraction

| Split | # Examples | # Relations | # Causes | # Effects |
|-------|------------|-------------|----------|-----------|
| Dev   | 2482       | 3224        | 3224     | 3238      |
| Train | 19892      | 25938       | 26174    | 26121     |
| Test  | 2433       | 3045        | 3065     | 3062      |

## Classification

| Split | # Relations | % Cause  | % Prevent  | % Enable  |
|-------|-------------|----------|------------|-----------|
| Dev   | 3224        | 63.78%   | 5.40%      | 30.82%    |
| Train | 25938       | 63.05%   | 5.90%      | 31.05%    |
| Test  | 3045        | 64.00%   | 5.38%      | 30.62%    |

# Preliminary results

| Model name                      | Token F1 |  EM        | Class Acc. | Class F1 |
|---------------------------------|----------|------------|------------|----------|
| GenQA (extraction)              | 81.09%   | 48.14%     |    -       |    -     |
| Sequence Labelling (extraction) | 73.23%   | 22.95%     |    -       |    -     |
| GenQA (joint)                   | 79.47%   | 52.16%     | 71.19%     | 54.08%   |
| BERT (extraction)[^base]        | 84.37%   | 51.48%     |    -       |    -     |
| BERT (classification)^3^        |    -     |   -        | 70.43%     | 71.74%   |
| BERT (joint)^3^                 |    -     | 21.21%     |    -       |    -     |

[^base]: Baselines from the original paper

# Problem with Exact Match evaluation

- Exact Match is an incomplete metric because it requires the output to be
  100\% identical to the annotation
- Different annotators can annotate the same sentence differently
- The model won't learn the "style" of all annotators simultaneously
    - It can't be exactly right all the time

# Exact Match example 1

::: {.block}
## Annotation

\textcolor{teal}{BB\&T and SunTrust have completed their merger, forming
Truist}, \textcolor{purple}{which we believe will drive the next step up in
profitability for the franchises}.
:::

::: {.block}
## Model prediction

\textcolor{teal}{BB\&T and SunTrust have completed their merger, forming
Truist}, which we believe will \textcolor{purple}{drive the next step up in
profitability for the franchises}.
:::

\textcolor{teal}{Cause} \textcolor{purple}{Effect}

# Exact Match example 2

::: {.block}
## Annotation

\textcolor{teal}{Given Tulip's lack of profitability (management has stated
the business was not profitable at the time of the October 2019 acquisition)},
\textcolor{purple}{we do not believe the business maintains a cost advantage}.
:::

::: {.block}
## Model prediction

Given \textcolor{teal}{Tulip's lack of profitability} (management has stated
the business was not profitable at the time of the October 2019 acquisition),
\textcolor{purple}{we do not believe the business maintains a cost advantage}.
:::

\textcolor{teal}{Cause} \textcolor{purple}{Effect}

# Proposed solution

- A human can see that the prediction is correct even if it's not an exact match
- **Apply Reinforcement Learning to train the model to produce correct answers
  instead**
  - Keep EM as a guardrail metric
- RL enables the detection of correct answers by reconstructing the original
  text from the extracted
- An entailment model detects whether the reconstructed text follows from the
  original
  - Entailment = correct answer

# RL framework: forward pass

![RL forward pass](rl_fwd.pdf){height=85%}

# RL framework: backward pass

![RL backward pass](rl_bwd.pdf){height=85%}

# RL framework: models

- Model 1: Extraction
    - This is the _GenQA (joint)_ above
    - T5-base generative QA
- Model 2: Reconstruction
    - Now: T5-base generative QA
    - Maybe: specialised structure-to-text model
- Model 3: Entailment
    - DeBERTa-base-MNLI
    - Easy problem: any transformer works here
- Models are finetuned for a few epochs before RL

# Next steps

- Implementation of the RL framework
- Experiments to determine the best algorithm, setup, rewards, etc.

# Current issues

- Model size in memory
    - 3 transformers means high VRAM usage
    - Small versions for development (t5-small, deberta-v3-xsmall)
    - Larger versions for final results (t5-base, deberta-base-mnli)
        - Batch size
        - Multiple GPUs
- How to best train this?
    - V1: alternate between freezing model 1 and training model 2, and vice-versa
    - V2: train all models at the same time

#

\centering

\begin{huge}
    Thanks!
\end{huge}

\scriptsize
[github.com/oyarsa/event_extraction/self_critique](
https://github.com/oyarsa/event_extraction/tree/master/self_critique)

