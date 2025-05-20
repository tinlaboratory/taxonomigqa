import random
import re
import utils

from collections import defaultdict
from ordered_set import OrderedSet

data_path = "data/things-taxonomic-sensitivity"

category_pairs_path = f"{data_path}/things-inheritance-SPOSE_prototype_sim-pairs.csv"
lemma_path = f"{data_path}/things-lemmas-annotated.csv"

category_pairs = utils.read_csv_dict(category_pairs_path)
category_pairs = [entry for entry in category_pairs if entry["hypernymy"] == "yes"]
category_pairs = [
    entry for entry in category_pairs if entry["premise"] != entry["conclusion"]
]

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


random.seed(1024)

idx = 1
item = 1

# hypernym_sentences = []
ns_sentences = []
swapped_sentences = []

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

    for parent in parents:
        parent_singular = lexicon[parent]["singular"]
        negative_samples = random.sample(negative_sample_space, 4)

        # swapped case
        category_singular = lexicon[category]["singular"]
        parent_generic = lexicon[parent]["generic"]
        if generic == "p":
            parent_NP = lexicon[parent]["plural"]
            parent_item = parent_NP
        else:
            parent_NP = lexicon[parent]["article"]
            if parent_NP.startswith("a ") or parent_NP.startswith("an "):
                parent_item = re.sub(r"^(a|an)\s", "", parent_NP)
            else:
                parent_item = parent_NP
        swapped_prefix = f"{parent_NP} {lexicon[parent]['taxonomic_phrase']}"

        swapped_sentences.append(
            (
                item,
                idx,
                category,
                category_item,
                parent,
                parent_singular,
                parent_item,
                category_singular,
                f"Is it true that {taxonomic_prefix} {parent_singular}?",
                f"Is it true that {swapped_prefix} {category_singular}?",
            )
        )

        for k, ns in enumerate(negative_samples):
            ns_singular = lexicon[ns]["singular"]
            ns_sentences.append(
                (
                    item,
                    idx,
                    f"ns_{k+1}",
                    category,
                    category_item,
                    parent,
                    parent_singular,
                    ns,
                    ns_singular,
                    f"Is it true that {taxonomic_prefix} {parent_singular}?",
                    f"Is it true that {taxonomic_prefix} {ns_singular}?",
                    # f"Is it true that {swapped_prefix} {category_singular}?",
                )
            )
            idx += 1
        item += 1

utils.write_csv(
    # data=hypernym_sentences,
    data=ns_sentences,
    path=f"{data_path}/taxomps-ns-qa.csv",
    header=[
        "item",
        "idx",
        "ns_id",
        "category",
        "category_item",
        "hypernym",
        "hypernym_item",
        "negative_sample",
        "negative_item",
        # "swapped-hyponym",
        # "swapped-hypernym",
        "hypernym_question",
        "negative_question",
        # "swapped_question",
    ],
)

utils.write_csv(
    data=swapped_sentences,
    path=f"{data_path}/taxomps-swapped-qa.csv",
    header=[
        "item",
        "idx",
        "category",
        "category_item",
        "hypernym",
        "hypernym_item",
        "swapped-hyponym",
        "swapped-hypernym",
        "hypernym_question",
        "swapped_question",
    ],
)
