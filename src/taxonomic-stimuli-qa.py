import random
from collections import defaultdict

from ordered_set import OrderedSet

import utils
import re

data_path = "data/things-taxonomic-sensitivity"

category_pairs_path = f"{data_path}/things-inheritance-SPOSE_prototype_sim-pairs.csv"
lemma_path = f"{data_path}/things-lemmas-annotated.csv"

category_pairs = utils.read_csv_dict(category_pairs_path)
category_pairs = [entry for entry in category_pairs if entry["hypernymy"] == "yes"]

lexicon = defaultdict(dict)
lemmas = utils.read_csv_dict(lemma_path)
lemmas = [l for l in lemmas if l["remove"] not in ["?", "1"]]
for entry in lemmas:
    lexicon[entry["lemma"]] = entry

lexicon = dict(lexicon)

hypernyms = []
category_membership = defaultdict(list)

for entry in category_pairs:
    if entry["conclusion"] in lexicon.keys() and entry["premise"] in lexicon.keys():
        category_membership[entry["conclusion"]].append(entry["premise"])
        hypernyms.append(entry["premise"])

hypernym_sentences = []
random.seed(1024)

idx = 1
item = 1

for category, parents in category_membership.items():
    generic = lexicon[category]["generic"]
    if generic == "p":
        category_NP = lexicon[category]["plural"]
        category_item = category_NP
    else:
        category_NP = lexicon[category]["article"]
        if category_NP.startswith("a ") or category_NP.startswith("an "):
            category_item = re.sub(r"^(a|an)\s", "", category_NP)
        else:
            category_item = category_NP
    taxonomic_prefix = f"{category_NP} {lexicon[category]['taxonomic_phrase']}"

    negative_sample_space = list(OrderedSet(hypernyms) - OrderedSet(parents))

    category

    for parent in parents:
        parent_singular = lexicon[parent]["singular"]
        negative_samples = random.sample(negative_sample_space, 4)

        for ns in negative_samples:
            ns_singular = lexicon[ns]["singular"]
            hypernym_sentences.append(
                (
                    item,
                    idx,
                    category,
                    category_item,
                    parent,
                    parent_singular,
                    ns,
                    ns_singular,
                    f"Answer the question with either Yes or No. Question: Is it true that {taxonomic_prefix} {parent_singular}? Answer:",
                    f"Answer the question with either Yes or No. Question: Is it true that {taxonomic_prefix} {ns_singular}? Answer:",
                )
            )
            idx += 1

        item += 1

utils.write_csv(
    data=hypernym_sentences,
    path=f"{data_path}/things-hypernym-minimal-pairs-qa.csv",
    header=[
        "item",
        "idx",
        "category",
        "category_item",
        "hypernym",
        "hypernym_item",
        "negative_sample",
        "negative_item",
        "hypernym_question",
        "negative_question",
    ],
)
