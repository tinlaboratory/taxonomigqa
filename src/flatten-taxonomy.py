import csv
import json
import utils
import pathlib

HYPERNYM_JSON="/Users/kanishka/Downloads/arg_hypernyms.json"
taxonomy = utils.read_json(HYPERNYM_JSON)

concepts = set()
cat_mem = []
hypernyms = set()
for k, v in taxonomy.items():
    concepts.add(k)
    for vv in v:
        concepts.add(vv)
        hypernyms.add(vv)
        cat_mem.append((vv, k, "yes"))

concepts = list(concepts)
concepts = [[c] for c in concepts]

print(f"{len(concepts)} total concepts, {len(cat_mem)} total pairs, and {len(hypernyms)} total hypernyms")

utils.write_csv(concepts, path="data/gqa_entities/gqa-lemmas.csv", header=["lemma"])
utils.write_csv(cat_mem, path="data/gqa_entities/category-membership.csv", header=["premise", "conclusion", "hypernymy"])
