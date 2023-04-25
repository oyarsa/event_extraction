---
title: Extracting cause and effect from sentences
author: Italo Luis da Silva
date: 2023-03-28
institute: King's College London
theme: Boadilla
colorlinks: true
---

# Problem statement

- Fine grained causal reasoning[^dataset][^paper]
- Extract cause and effect from sentences
- Determine the type of relation between the cause and the effect

[^dataset]: [github.com/YangLinyi/Fine-grained-Causal-Reasoning](https://github.com/YangLinyi/Fine-grained-Causal-Reasoning/)
[^paper]: [Towards Fine-grained Causal Reasoning and QA](https://arxiv.org/pdf/2204.07408.pdf)

# Examples
**Causes**:
\textcolor{red}{The firm's gross margin is set to stabilize} as
\textcolor{teal}{Harley refocuses its efforts on more profitable markets}, and
our base case assumes that \textcolor{purple}{it stabilizes around 32\% in
2029}, helped by \textcolor{olive}{a more measured approach to entering new
markets}.

**Enables**:
While the automotive part sector has performed relatively well during the
pandemic thus far, \textcolor{teal}{spiking unemployment} and
\textcolor{teal}{unprecedented global lockdowns} have \textcolor{red}{slashed
miles driven} and \textcolor{red}{slowed vehicle wear and tear}.

**Prevents**:
Given the \textcolor{teal}{importance of Teleflex products to health
outcomes}, product issues (either related to quality control or manufacturing)
should be prevented to \textcolor{red}{avoid social ESG concerns}.

\begin{scriptsize}
    \textbf{Relation} \newline
    \textcolor{teal}{Cau}\textcolor{olive}{se} \newline
    \textcolor{red}{Eff}\textcolor{purple}{ect}
\end{scriptsize}

# Dataset statistics

## Extraction

| Split | # Examples | # Relations | # Causes | # Effects |
|-------|------------|-------------|----------|-----------|
| Dev   | 2482       | 3226        | 3224     | 3238      |
| Train | 19892      | 25938       | 26174    | 26121     |
| Test  | 2433       | 3045        | 3065     | 3062      |

## Classification

| Split | # Examples | % Causes | % Prevents | % Enables |
|-------|------------|----------|------------|-----------|
| Dev   | 2482       | 63.78%   | 5.40%      | 30.82%    |
| Train | 19892      | 63.05%   | 5.90%      | 31.05%    |
| Test  | 2433       | 64.00%   | 5.38%      | 30.62%    |

# Preliminary results

| Model name                | Token F1 |  EM        | Class Acc | Class F1 |
|---------------------------|----------|------------|-----------|----------|
| GenQA (extraction)        | 81.09%   | 48.14%     |    -      |    -     |
| GenQA (joint)             | 79.47%   | 52.16%     | 71.19%    | 54.08%   |
| Sequence Labelling        | 73.23%   | 22.95%     |    -      |    -     |
| BERT (extraction)[^base]  | 84.37%   | 51.48%     |    -      |    -     |
| BERT (classification)^3^  |    -     |   -        | 70.43%    | 71.74%   |
| BERT (joint)^3^           |    -     | 21.21%     |    -      |    -     |

[^base]: Baseline from the original paper

# Problem with EM evaluation

- Exact Match is a flawed metric because it punishes the model for correct answers that
  don't exactly match the ground truth
- Different annotators will annotate the same sentence differently
- The model output won't learn the "style" of all annotators simultaneously

# EM example 1

## Annotation

\textcolor{teal}{BB\&T and SunTrust have completed their merger, forming
Truist}, \textcolor{purple}{which we believe will drive the next step up in
profitability for the franchises}.

## Model prediction

\textcolor{teal}{BB\&T and SunTrust have completed their merger, forming
Truist}, which we believe will \textcolor{purple}{drive the next step up in
profitability for the franchises}.

# EM example 2

## Annotation

\textcolor{teal}{Given Tulip's lack of profitability (management has stated
the business was not profitable at the time of the October 2019 acquisition)},
\textcolor{purple}{we do not believe the business maintains a cost advantage}.

## Model prediction

Given \textcolor{teal}{Tulip's lack of profitability} (management has stated
the business was not profitable at the time of the October 2019 acquisition),
\textcolor{purple}{we do not believe the business maintains a cost advantage}.

# RL framework: forward pass

![RL forward pass](rl_fwd.pdf){height=70%}

# RL framework: backward pass

![RL backward pass](rl_bwd.pdf){height=70%}

# RL framework: models

- Model 1: Extraction
    - Same as the GenQA (joint) above
    - T5-base Seq2Seq QA
- Model 2: Reconstruction
    - Now: T5-base Seq2Seq QA
    - Maybe: specialised structued to text model
- Model 3: Entailment
    - DeBERTa-base-MNLI
    - Easy problem: any transformer works here

# Next steps

- Implementation of the RL framework
- Experiments to determine the best algorithm, setup, rewards, etc.

# Current issues

- Model size in memory
    - 3 language models
    - All finetuned a little bit before the RL training
    - Model 3 is always frozen
- How to best train this?
    - V1: alternate between freezing model 1 and training model 2, and vice-versa

#

\centering

\begin{huge}
    Thanks!
\end{huge}

\scriptsize
[github.com/oyarsa/event_extraction](https://github.com/oyarsa/event_extraction)


