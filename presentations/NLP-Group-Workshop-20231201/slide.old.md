---
title: Extracting cause and effect from sentences
author: Italo Luis da Silva
date: 2023-10-19
institute: King's College London
theme: metropolis
colorlinks: true
---

# Task

Extract cause, effect and relation from text passage.

::: {.block}
## Example

\textcolor{teal}{The firm's gross margin is set to stabilize} as
\textcolor{purple}{Harley refocuses its efforts on more profitable markets}, and
our base case assumes that it stabilizes around 32\% in 2029, helped by a more
measured approach to entering new markets.

- \textcolor{purple}{Cause}: Harley refocuses its efforts on more profitable markets
- \textcolor{teal}{Effect}: The firm's gross margin is set to stabilize
- **Relation**: cause
:::

# Structured output example

\texttt{ \textcolor{purple}{[Cause] Harley refocuses its efforts on more profitable
markets} \textbf{[Relation] cause} \textcolor{teal}{[Effect] The firm's gross margin
is set to stabilize } }

# Evaluation

The standard metric, Exact Match, is not a good metric for this task.

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

# Reinforcement learning
- We can't use supervised learing with the because that tries to learn exact wording
- Use Reinforcement Learning to train the extraction model
- Reward models:
    - Entailment: entailment/neutral/contradiction
    - Valid: valid/invalid
- Reward is the logit of the true class (entailment or valid)

# Evaluation results
| Models                       | Human  | Token F1 | EM     |
| ---------------------------- | ------ | -------- | ------ |
| ChatGPT (10-shot)            | 35.13% | 67.52%   | 31.95% |
| Supervised                   | 64.38% | 80.59%   | 54.31% |
| RL with entailment           | 59.23% | 76.58%   | 47.06% |
| RL with valid                | 60.48% | 78.65%   | 50.02% |
| RL with entailment (no FT)   | -      | 75.65%   | 44.92% |
| Ensemble (entailment)        | -      | 80.56%   | 54.21% |
| Ensemble (valid)             | -      | 80.55%   | 54.21% |
| Ensemble (entailment, no FT) | -      | 80.59%   | 54.31% |

# Next steps

- Automated evaluation is tricky, but human evaluation is time-consuming and expensive
- LLM-based evaluation: how can we use the LLM to evaluate the output?
    - Currently looking into how to prompt GPT-3.5 and GPT-4 for this
    - Fine-tuned open source models might be an alternative
- There are some works on abstract measure like readability, grammar, faithfulness
  and context relevance, but not so much for specialised evaluation like this[^1].
- Future work: similar approach to WhyQA (TellMeWhy dataset)
    - Also prone to the same problems, but even harder to evaluate, even manually

[^1]: RAGAS: https://github.com/explodinggradients/ragas

#

\centering

\begin{huge}
    Thanks!
\end{huge}

\scriptsize
github.com/oyarsa/event_extraction

