import argparse
import csv
import pathlib
import os
import torch
import utils

from collections import defaultdict
from copy import deepcopy
from minicons import cwe
from ordered_set import OrderedSet
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(args):

    model = args.model
    model_name = model.replace("/", "_")

    questions = utils.read_csv_dict(
        "data/gqa_dataset/contextualized_text_analysis_data.tsv", True
    )

    questions_filtered = [q for q in questions if q["substitution_hop"] != "0"]

    nots = []
    questions_filtered_clean = []
    for entry in questions_filtered:
        question = entry['Input'].split("Question:")[-1]
        new_entry = deepcopy(entry)
        hypernym = entry['Hypernym-form']
        if not (hypernym in question):
            if hypernym not in ['children', 'people', 'women', 'birds of prey']:
                possessive_hypernym = entry['Hypernym-form'][:-1]+"'s"
            else:
                if hypernym == "children":
                    possessive_hypernym = "child's"
                elif hypernym == "women":
                    possessive_hypernym = "woman's"
                elif hypernym == "people":
                    possessive_hypernym = "person's"
                elif hypernym == "birds of prey":
                    possessive_hypernym = "bird of prey's"
            if possessive_hypernym in question:
                new_entry['Hypernym-form'] = possessive_hypernym
            # elif entry['Hypernym-form'] == "people"
            else:
                nots.append(entry)
        else:
            if hypernym+"s" in question:
                # plurals.append(entry)
                new_entry['Hypernym-form'] = hypernym+"s"
        # if hypernym not in question:
        #     no_q.append(entry)
        questions_filtered_clean.append(new_entry)

    # check

    noq = []
    for entry in questions_filtered_clean:
        question = entry['Input'].split("Question:")[-1]
        # new_entry = deepcopy(entry)
        hypernym = entry['Hypernym-form']
        if not (hypernym in question):
            noq.append(entry)

    assert len(noq) == 0

    vlm = args.vlm

    if vlm:
        lm = cwe.VisualCWE(model, device=args.device)
    else:
        lm = cwe.CWE(model, device=args.device)

    batches = DataLoader(questions_filtered_clean, batch_size=args.batch_size)

    # layerwise_reps = defaultdict(list)
    layerwise = defaultdict(list)
    for batch in tqdm(batches):
        queries = list(
            zip(batch["Input"], batch["Hyponym-form"], batch["Hypernym-form"])
        )
        reps = lm.extract_paired_representations(
            queries, layer="all", multi_strategy="all"
        )
        # layerwise_reps[layer]
        for layer in range(lm.layers + 1):
            # layerwise_reps[layer].append((reps[0][layer], reps[1][layer]))
            cosines = [
                torch.cosine_similarity(x1, x2[-1].unsqueeze(0)).max().item()
                for x1, x2 in zip(reps[0][layer], reps[1][layer])
            ]
            layerwise[layer].extend(cosines)

    results = []

    for l, sims in layerwise.items():
        for i, sim in enumerate(sims):
            results.append((i, l, sim))

    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    utils.write_csv(results, f"{args.output_dir}/{model_name}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument(
        "--output_dir", type=str, default="data/results/gqa-cwe-sims"
    )
    parser.add_argument("--vlm", "-v", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    main(args)
