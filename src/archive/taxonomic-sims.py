import argparse
import csv
import pathlib
import os
import torch
import utils

from collections import defaultdict
from minicons import cwe
from ordered_set import OrderedSet
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

def main(args):

    # model stuff
    model = args.model
    vlm = args.vlm
    model_name = model.replace("/", "_")

    if vlm:
        lm = cwe.VisualCWE(model, device=args.device)
    else:
        lm = cwe.CWE(model, device=args.device)

    # data stuff

    path = "data/things-taxonomic-sensitivity/things-hypernym-minimal-pairs-qa.csv"

    stimuli = OrderedSet()
    items = defaultdict(lambda x: False)
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for line in reader:
            stimuli.add(
                (
                    line["hypernym_question"],
                    line["category_item"],
                    line["hypernym_item"],
                )
            )

    stimuli = list(stimuli)

    batches = DataLoader(stimuli, batch_size=args.batch_size)

    layerwise = defaultdict(list)
    for batch in tqdm(batches):
        batch = list(zip(*batch))
        emb1, emb2 = lm.extract_paired_representations(batch, layer="all")
        for i, (e1, e2) in enumerate(zip(emb1, emb2)):
            layerwise[i].extend(torch.cosine_similarity(e1, e2).tolist())

    results = []
    for layer, sims in layerwise.items():
        for i, sim in enumerate(sims):
            results.append((i + 1, layer, sim))

    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    utils.write_csv(
        results,
        path=f"{args.output_dir}/{model_name}.csv",
        header=["item", "layer", "sim"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument(
        "--output_dir", type=str, default="data/results/hypernym-qa-sims"
    )
    parser.add_argument("--vlm", "-v", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    main(args)
