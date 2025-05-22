import argparse
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

import pathlib

import torch
import json
from transformers import AutoTokenizer
import networkx as nx
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import hierarchical as hrc


def main(args):

    device = torch.device("cuda:0")
    model = args.model
    vlm = args.vlm

    model_name = model.replace("/", "_")
    g, _, _ = hrc.get_g(model, device, v2s=vlm)
    vocab_dict, vocab_list = hrc.get_vocab(model)

    cats, G, sorted_keys = hrc.get_categories("noun", model_name)

    torch.manual_seed(100)
    shuffled_g = g[torch.randperm(g.shape[0])]
    alpha = 0.7

    vec_reps = {
        "original": {"split": {}, "non_split": {}},
        "shuffled": {"split": {}, "non_split": {}},
        "train_lemmas": {},
        "test_lemmas": {},
        "g": g,
        "shuffled_g": shuffled_g,
        "alpha": alpha,
    }

    for node in tqdm(sorted_keys):
        lemmas = cats[node]
        original_dir = hrc.estimate_cat_dir(lemmas, g, vocab_dict)
        shuffled_dir = hrc.estimate_cat_dir(lemmas, shuffled_g, vocab_dict)

        vec_reps["original"]["non_split"].update({node: original_dir})
        vec_reps["shuffled"]["non_split"].update({node: shuffled_dir})

        # random.seed(100)
        # random.shuffle(lemmas)

        # train_lemmas = lemmas[:int(alpha * len(lemmas))]
        # test_lemmas = lemmas[int(alpha * len(lemmas)):]
        # original_dir = hrc.estimate_cat_dir(train_lemmas, g, vocab_dict)
        # shuffled_dir = hrc.estimate_cat_dir(train_lemmas, shuffled_g, vocab_dict)

        # vec_reps['original']['split'].update({node: original_dir})
        # vec_reps['shuffled']['split'].update({node: shuffled_dir})
        # vec_reps['train_lemmas'].update({node: train_lemmas})
        # vec_reps['test_lemmas'].update({node: test_lemmas})

    G_undirected = G.to_undirected()
    dist_matrix = nx.floyd_warshall_numpy(G_undirected, nodelist=sorted_keys)
    dist_matrix = 1 / (dist_matrix + 1)

    def compute_pairwise(version):
        original_dirs = torch.stack(
            [v[version] for k, v in vec_reps["original"]["non_split"].items()]
        )
        original_dirs = original_dirs / original_dirs.norm(dim=1).unsqueeze(1)
        shuffled_dirs = torch.stack(
            [v[version] for k, v in vec_reps["shuffled"]["non_split"].items()]
        )
        shuffled_dirs = shuffled_dirs / shuffled_dirs.norm(dim=1).unsqueeze(1)

        return (
            (original_dirs @ original_dirs.T).cpu().numpy(),
            (shuffled_dirs @ shuffled_dirs.T).cpu().numpy(),
        )

    lda = compute_pairwise("lda")
    mean = compute_pairwise("mean")

    mats_lda = [dist_matrix, lda[0], lda[1]]
    mats_mean = [dist_matrix, mean[0], mean[1]]

    path = f"reps/{model_name}"
    pathlib.Path(path).mkdir(exist_ok=True, parents=True)

    np.save(f"{path}/lda.npy", mats_lda)
    np.save(f"{path}/mean.npy", mats_mean)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--vlm", action="store_true")

    args = parser.parse_args()
    main(args)
    
