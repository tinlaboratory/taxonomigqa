"""
Evaluating LMs based on log-probabilities on sequences

IncrementalLMScorer: Molmo, Qwen, Vicuna, llama
VLMScorer: Llava, llama-vision
"""

import argparse
import os
import pathlib

from minicons import scorer
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


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
        hypernym_sentences = batch["hypernym_sentence"]
        negative_sentences = batch["negative_sentence"]

        hypernym_scores = lm.sequence_score(hypernym_sentences)
        negative_scores = lm.sequence_score(negative_sentences)

        for i, h, n in zip(idx, hypernym_scores, negative_scores):
            results.append((i, h, n))

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    utils.write_csv(
        results,
        path=f"{output_dir}/{model_name}.csv",
        header=["idx", "hypernym_score", "negative_score"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument(
        "--eval_path",
        type=str,
        default="data/things-taxonomic-sensitivity/things-hypernym-minimal-pairs.csv",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/results/hypernym-minimal-pairs"
    )
    parser.add_argument("--vlmscorer", "-v", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    main(args)
