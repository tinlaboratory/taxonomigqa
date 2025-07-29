import argparse
import csv
import os
import pathlib
import utils
import lexicon

from collections import defaultdict
from tqdm import tqdm

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

lemma_path="data/rodriguez-et-al/things-lemmas-annotated.csv"
triple_path="data/rodriguez-et-al/things-inheritance-SPOSE_prototype_sim-pairs.csv"

def lemma2concept(entry):
    return lexicon.Concept(
        lemma=entry["lemma"],
        singular=entry["singular"],
        plural=entry["plural"],
        article=entry["article"],
        generic=entry["generic"],
        taxonomic_phrase=entry["taxonomic_phrase"],
    )

stimuli = []

# read in concepts
concepts = defaultdict(lexicon.Concept)
with open(lemma_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # concepts.append(lemma2concept(row))
        if row["remove"] != "1":
            concepts[row["lemma"]] = lemma2concept(row)

contrasting_properties = {
    'daxable': lexicon.Property("feps", "has feps", "have feps"),
    "feps": lexicon.Property("daxable", "is daxable", "are daxable")
    }

triples = utils.read_csv_dict(triple_path)

fake_properties = [lexicon.Property("daxable", "is daxable", "are daxable")] * len(triples)

for triple, fake_property in zip(triples, fake_properties):
    try:
        hyponym = triple["hyponym"]
        anchor = triple["anchor"]
    except:
        anchor = triple["premise"]
        hyponym = triple["conclusion"]
    if hyponym in concepts.keys() and anchor in concepts.keys():
        child = concepts[hyponym]
        parent = concepts[anchor]

    premise = parent.property_sentence(fake_property)
    conclusion = child.property_sentence(fake_property)
    stimuli.append((parent.lemma, child.lemma, premise, conclusion, triple['hypernymy']))

utils.write_csv(stimuli, "data/rodriguez-et-al/stimuli.csv", header=["parent", "child", "premise", "conclusion", "label"])
