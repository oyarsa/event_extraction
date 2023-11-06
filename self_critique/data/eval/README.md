# Evaluation files

Shell commands to evaluate stuff.

Note: these use Fish shell syntax. If you use Bash, you'll need to change the syntax.

## Evaluate all files:
```fish
❯ python merge.py (fd -ejson '.metrics.json' -HI)
```

## Merge evaluation results into a single file
```fish
❯ python merge.py (fd -ejson '.metrics.json' -HI)
```
