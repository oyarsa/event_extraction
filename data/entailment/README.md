# Entailment datasets
Variations:

- `contradiction`: only contradiction cases
- `entailment+neutral`: entailment and neutral cases, no contradiction
- `merged`: all cases in one package
- `shuffled`: `merged` shuffled with `shuffle.py`
- `parphrased`: `merged` with `sentence2` paraphrased by ChatGPT. Contains regular
  `merged` and `shuffled` as well.

`paraphrased` is the latest one. That's what you want.
