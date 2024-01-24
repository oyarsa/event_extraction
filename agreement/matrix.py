import json
import sys
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Result:
    context: str
    gold: str
    human_valid: bool
    model_valid: bool
    model_label: str


if len(sys.argv) != 4:
    print("Usage: python calc.py <human_path> <model_path> <true_class>")
    sys.exit(1)

human_path = sys.argv[1]
model_path = sys.argv[2]
true_class = sys.argv[3].casefold().strip()

with open(human_path) as f:
    human = json.load(f)
with open(model_path) as f:
    model = json.load(f)

matches = []

for h in human:
    for m in model:
        if h["input"] == m["context"] and h["gold"] == m["answers"]:
            matches.append(
                Result(
                    context=h["input"],
                    gold=h["gold"],
                    human_valid=h["valid"],
                    model_valid=m["reward_label"].casefold().strip() == true_class,
                    model_label=m["reward_label"].casefold().strip(),
                )
            )
            break

# Decision Matrix Calculation
decision_matrix = defaultdict(lambda: {"True": 0, "False": 0})

for r in matches:
    human_decision = "True" if r.human_valid else "False"
    decision_matrix[r.model_label][human_decision] += 1

# Output Results
human_valid = sum(r.human_valid for r in matches) / len(matches)
model_valid = sum(r.model_valid for r in matches) / len(matches)
agreement = sum(r.human_valid == r.model_valid for r in matches) / len(matches)

print(f"Found {len(matches)} matches")
print(f"Human valid: {human_valid:.2%}")
print(f"Model valid: {model_valid:.2%}")
print(f"Agreement: {agreement:.2%}")

print("\nDecision Matrix:")
# Header
print(f"{'Model Label':<20} {'Human True':<15} {'Human False':<15}")

# Rows
for label, counts in decision_matrix.items():
    print(f"{label:<20} {counts['True']:<15} {counts['False']:<15}")
