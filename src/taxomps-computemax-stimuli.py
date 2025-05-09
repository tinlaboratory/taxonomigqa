import random
import re
import utils
import csv

from collections import defaultdict
from ordered_set import OrderedSet

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

phrasings = ["are", "is", "are a type of", "is a type of"]


def taxonomic_sentence_generator(hyponym, hypernym):
    """
    Generates all phrasings of X is a Y, with ids

    Returns id, sentence, surface form of hypo, surface form of hyper
    """
    taxonomic_sentences = []
    for i, phrasing in enumerate(phrasings):
        if phrasing == "are" or phrasing == "are a type of":
            hyponym_NP = lexicon[hyponym]["plural"]
            hyponym_item = hyponym_NP

            if phrasing == "are":
                hypernym_NP = lexicon[hypernym]["plural"]
            else:
                hypernym_NP = lexicon[hypernym]["singular"]

            hypernym_item = hypernym_NP

        elif phrasing == "is" or phrasing == "is a type of":
            hyponym_NP = lexicon[hyponym]["article"]
            hyponym_item = re.sub(r"^(a|an)\s", "", hyponym_NP)

            if phrasing == "is a type of":
                hypernym_NP = lexicon[hypernym]["singular"]
                hypernym_item = hypernym_NP
            else:
                hypernym_NP = lexicon[hypernym]["article"]
                hypernym_item = re.sub(r"^(a|an)\s", "", hypernym_NP)

        taxonomic_sentence = f"{hyponym_NP} {phrasing} {hypernym_NP}"

        taxonomic_sentences.append(
            (i + 1, taxonomic_sentence, hyponym_item, hypernym_item)
        )

    return taxonomic_sentences


def save_sentences_csv(sentences, path, header):
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, entry in enumerate(sentences):
            instance = (i+1,) + entry
            writer.writerow(instance)


random.seed(1024)


hypernym_sentencs = []
ns_sentences = []
swapped_sentences = []

for category_id, (category, parents) in enumerate(category_membership.items()):
    # negative_sample_space = list(OrderedSet(hypernyms) - OrderedSet(parents))
    negative_sample_space = OrderedSet(hypernyms) - OrderedSet(parents)
    leaf_cats = OrderedSet(category_membership.keys()) - OrderedSet([category])
    negative_sample_space_extended = list(negative_sample_space.union(leaf_cats))

    # print(negative_sample_space_extended)

    for parent_id, parent in enumerate(parents):
        negative_samples = random.sample(negative_sample_space, 4)

        sents = taxonomic_sentence_generator(category, parent)
        swapped_sents = sents = taxonomic_sentence_generator(parent, category)

        for s in sents:
            phrasing_id, sentence, category_item, parent_item = s
            hypernym_sentencs.append(
                (
                    category_id + 1,
                    parent_id + 1,
                    category,
                    parent,
                    phrasing_id,
                    f"Is it true that {sentence}?",
                    category_item,
                    parent_item,
                )
            )

        for s in swapped_sents:
            phrasing_id, sentence, category_item, parent_item = s
            swapped_sentences.append(
                (
                    category_id + 1,
                    parent_id + 1,
                    category,
                    parent,
                    phrasing_id,
                    f"Is it true that {sentence}?",
                    parent_item,
                    category_item,
                )
            )

        for k, ns in enumerate(negative_samples):
            ns_sents = taxonomic_sentence_generator(category, ns)

            for nss in ns_sents:
                phrasing_id, sentence, category_item, parent_item = nss
                ns_sentences.append(
                    (
                        category_id + 1,
                        parent_id + 1,
                        f"ns_{k+1}",
                        category,
                        ns,
                        phrasing_id,
                        f"Is it true that {sentence}?",
                        category_item,
                        parent_item,
                    )
                )

save_sentences_csv(
    hypernym_sentencs,
    "data/gqa_entities/taxomps-hypernym.csv",
    header=[
        "item",
        "category_id",
        "parent_id",
        "category",
        "parent",
        "phrasing_id",
        "question",
        "category_item",
        "parent_item",
    ],
)

print(swapped_sentences[:10])

save_sentences_csv(
    swapped_sentences,
    "data/gqa_entities/taxomps-swapped.csv",
    header=[
        "item",
        "category_id",
        "parent_id",
        "category",
        "parent",
        "phrasing_id",
        "question",
        "category_item",
        "parent_item",
    ],
)

save_sentences_csv(
    ns_sentences,
    "data/gqa_entities/taxomps-ns-all.csv",
    header=[
        "item",
        "category_id",
        "parent_id",
        "ns_id",
        "category",
        "parent",
        "phrasing_id",
        "question",
        "category_item",
        "parent_item",
    ],
)
