# Evaluation data

## Extraction (FCR)

The main files are in the in these directories:

- `training`: training
- `evaluation`: development
- `prediction`: test

The main files are:

- `rule_labelled.json`: automated evaluation results using string matching
    - These are guaranteed to be correct, but may not be complete
- `hand_labelled.json`: results that could not be automated, manually annotated
- `all_labelled.json`: combination of the above two files.
    - For `evaluation` and `prediction`, contains the whole dataset.
    - For `training`, it's incomplete, but is still most of it: 15732/19892 (79%).
    - Also contains other stuff that was used for exploration, but is not overall
      relevant.
- `all_labelled.tagged.json`: the same as `all_labelled.json`, but with manually
  assigned error/situation `tag`s. The tags are:
  - `em`: gold and pred are Exact Matches (always valid)
  - `nonsubstr`: gold and pred are not substrings of each other (always invalid)
  - `substr_valid`: gold and pred are substrings of each other, and the resulting
    extraction is valid
  - `substr_invalid`: gold and pred are substrings of each other, but it's invalid
- `all_labelled.lite.json`: same as `all_labelled.tagged.json`, but with only the
  relevant fields (i.e. `input`, `gold`, `output`, `valid` and `tag`).
  - Should be the one used for most things.
