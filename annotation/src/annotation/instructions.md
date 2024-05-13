## Cause and Effect Extraction Annotation Task

### Introduction
In this annotation task, you will be presented with a source text, and two extractions:
an cause clause and effect clause. Each have a reference answer and a model answer.

The goal is to determine whether the model answer accurately captures the cause and
effect relationship expressed in the source text, using the reference answer as a guide.

**The model answer will be considered "valid" if it conveys the same meaning as the
reference answer, even if the wording doesn't match exactly.**

### Task Overview
1. Read the source text carefully to understand the context and the cause and effect
   relationship being described.
2. Examine the reference answer, which highlights the causes and effects extracted from
   the source text.
3. Compare the model answer to the reference answer, paying attention to the causes and
   effects identified.
4. Determine whether the model answer is "valid" or "invalid" based on the following
criteria:
  - The meaning of the causes and effects in the model answer should be the same as the
reference answer, even if the wording differs.
  - All words used in the model answer must be present in the source text, but they
  don't necessarily have to appear in the reference answer.
  - The model answer might contain fewer words than the reference answer. If the meaning
    is maintained, it should still be considered valid. It's possible that the lost
    words are not essential to the cause and effect relationship.
  - The model might contain more words than the reference. If that's the case, make sure
    that the additional words don't change the meaning of the cause and effect
    relationship.
5. It is important that the extractions match. Even if the model answers an otherwise
   valid explanation, it should be considered invalid if it doesn't refer to the same
  causes and effects as the reference answer.

### Tips
- Focus on the meaning and the cause and effect relationship, rather than the exact
  wording.
- If you're unsure about the validity of the model answer, re-read the source text and
  reference answer to gain clarity.
- If you're unsure, mark the example as "invalid".

### Example - Invalid

#### Source Text

While costs (mainly salaries) are largely fixed to maintain capacity, we believe that
there is some room for PRA to use these fixed resources more effectively as the CRO
business grows, in addition to spreading data solutions costs over a larger revenue
base.

#### Cause

**Reference**: the CRO business grows

**Model**: the CRO business grows, in addition to spreading data solutions costs over a
larger revenue base

#### Effect

**Reference**: some room for PRA to use these fixed resources more effectively

**Model**: there is some room for PRA to use these fixed resources more effectively

#### Is the model output valid relative to the reference?

**Answer**: _Invalid_

**Explanation**: The model answer includes additional information about spreading data
solutions costs over a larger revenue base, which is not present in the reference.

Note that the effect clauses are matches, as the words "there is" do not change the
meaning of the extraction. However, since both the cause and effect clauses must match
exactly, this example is considered invalid.

### Example - Valid

#### Source Text

Further, historically weak traffic levels augmented the importance of up-selling and
cross-selling, while providing food suitable for feeding a family became a necessity.

#### Cause

**Reference**: historically weak traffic levels

**Model**: historically weak traffic levels

#### Effect

**Reference**: the importance of up-selling and cross-selling

**Model**: augmented the importance of up-selling and cross-selling

#### Is the model output valid relative to the reference?

**Answer**: _Valid_

**Explanation**: The cause clauses match exactly, and the model answer maintains the
meaning of the effect clause, as the extra word "augmented" doesn't change the
relationship.
