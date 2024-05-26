import json
import sys
from itertools import combinations

from sklearn.metrics import cohen_kappa_score

with open(sys.argv[1]) as file:
    data = json.load(file)

# Extract the unique names
names = list({item["name"] for entry in data for item in entry["valid"]})

# Create a dictionary to store the valid values for each name
valid_dict = {name: [] for name in names}

# Populate the valid_dict with valid values for each name
for entry in data:
    for item in entry["valid"]:
        valid_dict[item["name"]].append(item["valid"])

kappas = []
# Calculate pairwise Cohen's Kappa
for name1, name2 in combinations(names, 2):
    valid1 = valid_dict[name1]
    valid2 = valid_dict[name2]
    kappa = cohen_kappa_score(valid1, valid2)
    print(f"{name1:<10} {name2:<10}: {kappa:.4f}")
    kappas.append(kappa)

avg_kappa = sum(kappas) / len(kappas)
print(f"\nAverage Cohen's Kappa: {avg_kappa:.4f}")
