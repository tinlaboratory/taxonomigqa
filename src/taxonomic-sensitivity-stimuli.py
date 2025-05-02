import random
from collections import defaultdict

from ordered_set import OrderedSet

import utils

data_path = "data/things-taxonomic-sensitivity"

category_pairs_path = f"{data_path}/things-inheritance-SPOSE_prototype_sim-pairs.csv"
lemma_path = f"{data_path}/things-lemmas-annotated.csv"

data_path = "data/gqa_entities"

category_pairs_path = f"{data_path}/category-membership.csv"
lemma_path = f"{data_path}/gqa-lemmas-annotated.csv"


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

for category, parents in category_membership.items():
    generic = lexicon[category]["generic"]
    if generic == "p":
        category_NP = lexicon[category]["plural"]
    else:
        category_NP = lexicon[category]["article"]

    taxonomic_prefix = f"{category_NP} {lexicon[category]['taxonomic_phrase']}"

    negative_sample_space = list(OrderedSet(hypernyms) - OrderedSet(parents))

    for parent in parents:
        if lexicon[category]['taxonomic_phrase'] == "is":
            parent_singular = lexicon[parent]["article"]
        else:
            parent_singular = lexicon[parent]["singular"]
        negative_samples = random.sample(negative_sample_space, 4)

        for ns in negative_samples:
            if lexicon[category]['taxonomic_phrase'] == "is":
                ns_singular = lexicon[ns]["article"]
            else:
                ns_singular = lexicon[ns]["singular"]
            hypernym_sentences.append(
                (
                    idx,
                    category,
                    parent,
                    ns,
                    f"{taxonomic_prefix} {parent_singular}",
                    f"{taxonomic_prefix} {ns_singular}",
                )
            )
            idx += 1

utils.write_csv(
    data=hypernym_sentences,
    # path=f"{data_path}/things-hypernym-minimal-pairs.csv",
    path=f"{data_path}/gqa-taxomps.csv",
    header=[
        "idx",
        "category",
        "hypernym",
        "negative_sample",
        "hypernym_sentence",
        "negative_sentence",
    ],
)
