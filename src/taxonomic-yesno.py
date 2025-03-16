import argparse
import pathlib
import torch
import utils

from minicons import scorer
from torch.utils.data import DataLoader
from tqdm import tqdm

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

OPTIONS = ["Yes", "No", "yes", "no"]


def main(args):
    model = args.model
    vlmscorer = args.vlmscorer
    model_name = model.replace("/", "_")

    eval_path = args.eval_path
    output_dir = args.output_dir

    if vlmscorer:
        lm = scorer.VLMScorer(model, device=args.device)
    else:
        lm = scorer.IncrementalLMScorer(
            model, device=args.device, trust_remote_code=True
        )

    eval_data = utils.read_csv_dict(eval_path)
    eval_set = DataLoader(eval_data, batch_size=args.batch_size)

    results = []
    for batch in tqdm(eval_set):
        idx = batch["idx"]
        hypernym_question = batch["hypernym_question"]
        negative_question = batch["negative_question"]

        if vlmscorer:
            hypernym_dist = lm.next_word_distribution(hypernym_question, image=None)
            negative_dist = lm.next_word_distribution(negative_question, image=None)
        else:
            hypernym_dist = lm.next_word_distribution(hypernym_question)
            negative_dist = lm.next_word_distribution(negative_question)

        hypernym_probs, hypernym_ranks = lm.query(
            hypernym_dist, queries=[OPTIONS] * len(hypernym_question)
        )
        negative_probs, negative_ranks = lm.query(
            negative_dist, queries=[OPTIONS] * len(hypernym_question)
        )

        hypernym_labels = [
            OPTIONS[i] for i in torch.tensor(hypernym_probs).argmax(1).tolist()
        ]
        negative_labels = [
            OPTIONS[i] for i in torch.tensor(negative_probs).argmax(1).tolist()
        ]

        # hypernym_scores = lm.sequence_score(hypernym_sentences)
        # negative_scores = lm.sequence_score(negative_sentences)

        for i, h, n in zip(idx, hypernym_labels, negative_labels):
            results.append((i, h, n))

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    utils.write_csv(
        results,
        path=f"{output_dir}/{model_name}.csv",
        header=["idx", "hypernym_pred", "negative_pred"],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument(
        "--eval_path",
        type=str,
        default="data/things-taxonomic-sensitivity/things-hypernym-minimal-pairs-qa.csv",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/results/hypernym-minimal-pairs-qa"
    )
    parser.add_argument("--vlmscorer", "-v", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    main(args)
