# fincausal-train-ann

Annotated and to-be annotated data from the FinCausal train set.

Files:
- tagged.json: The entire train dataset with tags: empty, exact_match, needs_annotation
  and no_intersection.
  - empty: either cause or effect predictions are the empty string
  - exact_match: both cause and effect predictions are exactly the same as annotation
  - no_intersection: either cause or effect prediction don't share at least a token
  - needs_annotation: everything else not in the above tags

  - Since exact_match is always valid, empty and no_intersection are always invalid,
    everything else here happens only on the needs_annotation subset

- to_annotate.json: The needs_annotation subset of tagged.json, without the tag key
- human_annotated.json: Portion of the human annotation task that covered train set
  examples
- remain_unannotated.json: The remaining examples from the train set that were not
  annotated by humans

Inside "inputs":
- to_annotate.json: copy of remain_unannotated.json

TODO:
- Annotate `remain_unannotated.json` and add to `human_annotated.json`
- Merge `human_annotated.json` with `tagged.json` to create a completely annotated
  dataset
