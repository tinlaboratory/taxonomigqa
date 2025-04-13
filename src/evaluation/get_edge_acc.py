import argparse
import os
import sys

import pandas as pd

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
)
import glob
import json

from evaluator import strict_answer_match
from tree_util import get_tree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        type=str,
        default="./data/",
        help="dataset name",
    )

    return parser.parse_args()


def get_hypers(hyper_tree, noun):
    return hyper_tree[noun].path()[1:-1]


def get_hypernyms(hypernym_path):
    hyper_tree = get_tree(hypernym_path)
    arg_dict = {}
    for arg in args:
        hyp = get_hypers(hyper_tree, arg)
        arg_dict[arg] = hyp
    print(arg_dict)
    print(len(arg_dict))
    return arg_dict


def concatenate_a_row(df, model, model_type, concept1, concept2, raw_counts):
    # Concatenate the new row to the DataFrame
    accuracy = raw_counts[0] / raw_counts[1] * 100 if raw_counts[1] != 0 else 0
    new_row = {
        "model": model,
        "model-type": model_type,
        "concept1": concept1,
        "concept2": concept2,
        "raw-counts": raw_counts,
        "accuracy": accuracy,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df


def get_model_name(filename):
    if "llava" in filename:
        model = "llava"
    elif "mllama" in filename:
        model = "mllama"
    elif "molmo" in filename:
        model = "molmo"
    elif "qwen" in filename:
        model = "qwen"
    elif "vicuna" in filename:
        model = "vicuna"
    elif "llama" in filename:
        model = "llama"

    return model


def process_csv(model, model_type, val, arg_hypernyms):
    # Load the CSV file
    df = pd.read_csv(f"{val}", sep="\t")
    df["strict_eval"] = df.apply(
        lambda row: strict_answer_match(
            str(row["ground_truth"]), str(row["model_output"])
        ),
        axis=1,
    )
    # args = set(df[df['substitution_hop'] == 0]['argument'].tolist())

    # start the algo
    # create a res df
    intermediate_results = {}
    res_df = pd.DataFrame(
        columns=[
            "model",
            "model-type",
            "concept1",
            "concept2",
            "raw-counts",
            "accuracy",
        ]
    )
    for arg, hyps in arg_hypernyms.items():
        df_args = df[(df["original_arg"] == arg)]
        words = [arg] + hyps
        for i, word in enumerate(words[:-1]):
            for j in range(i + 1, len(hyps) + 1):
                df_child_correct = df_args[
                    (df_args["substitution_hop"] == i)
                    & (df_args["strict_eval"] == True)
                ]
                uniq_ids_in_child = df_child_correct["question_id"].unique()
                df_parent_conditioned = df_args[
                    (df_args["question_id"].isin(uniq_ids_in_child))
                    & (df_args["substitution_hop"] != i)
                ]
                df_parent_correct = df_parent_conditioned[
                    (df_parent_conditioned["substitution_hop"] == j)
                    & (df_parent_conditioned["strict_eval"] == True)
                ]
                if (word, words[j]) not in intermediate_results:
                    if len(df_parent_correct) != 0 and len(df_child_correct) == 0:
                        print(word, words[j])
                    intermediate_results[(word, words[j])] = [
                        len(df_parent_correct),
                        len(df_child_correct),
                    ]
                else:
                    intermediate_results[(word, words[j])][0] += len(df_parent_correct)
                    intermediate_results[(word, words[j])][1] += len(df_child_correct)

    # concatenate intermediate results
    for (concept1, concept2), counts in intermediate_results.items():
        res_df = concatenate_a_row(
            res_df, model, model_type, concept1, concept2, (counts[0], counts[1])
        )
    return res_df


if __name__ == "__main__":

    # Define the base path and hypernym path
    BASE_PATH = "/projectnb/tin-lab/yuluq/"
    hypernym_path = (
        BASE_PATH + "multimodal-representations/data/gqa_entities/noun-hypernyms.json"
    )
    arg_hyp_path = "./data/arg_hypernyms.json"

    # Parse args
    args = parse_args()
    data_folder = args.data_folder

    # load or create arg-hyp dict
    if os.path.exists(arg_hyp_path):
        with open(arg_hyp_path, "r") as f:
            arg_hypernyms = json.load(f)
    else:
        arg_hypernyms = get_hypernyms(hypernym_path)
        with open(arg_hyp_path, "w") as f:
            json.dump(arg_hypernyms, f)

    # Obtain csv files
    vlm_full = glob.glob(f"{data_folder}/*constrained_full_*.csv")
    vlm_text = glob.glob(f"{data_folder}/*text_only_*.csv")
    lm = glob.glob(f"{data_folder}/*constrained_decoding_*.csv")

    csv_files = {"vlm": vlm_full, "vlm-text": vlm_text, "lm": lm}

    res_df = pd.DataFrame(
        columns=[
            "model",
            "model-type",
            "concept1",
            "concept2",
            "raw-counts",
            "accuracy",
        ]
    )
    for key, value in csv_files.items():
        model_type = key
        for val in value:
            model = get_model_name(val)
            append_df = process_csv(model, model_type, val, arg_hypernyms)
            res_df = pd.concat([res_df, append_df], axis=0, ignore_index=True)

    # save the results
    # if os.path.exists(f"{data_folder}/edge_accuracy.csv"):
    res_df.to_csv(f"{data_folder}/edge_accuracy.csv", sep="\t", index=False)
