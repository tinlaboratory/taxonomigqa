import argparse
import pandas as pd


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_files",
        nargs="*",
        default=["data/0302_llava_7B.csv"],
        help="Input file as csv",
    )
    parser.add_argument("--output_dataset", default="datasets_with_scores.csv")
    return parser


def compute_cumulative_score(dataset):
    dataset["cumulative"] = dataset.groupby("question_id")["score"].transform(
        lambda x: x.cumsum().where(x.cumsum() != 0, -1)
    )
    return dataset


def compute_leaf_score(dataset):
    dataset["leaf"] = dataset.groupby("question_id")["score"].transform("cumsum")
    return dataset


def compute_hca_score(dataset):
    dataset = dataset.sort_values(["question_id", "substitution_hop"])
    dataset["hca"] = dataset.groupby("question_id")["score"].transform(
        lambda x: x.cumprod()
    )
    return dataset


def add_base_score(dataset):
    dataset["model_output"] = dataset["model_output"].str.lower()
    dataset["response"] = dataset["model_output"].str.contains("yes")
    dataset["answer"] = dataset["ground_truth"].str.contains("yes")
    dataset["substitution"] = dataset["argument"]
    dataset["argument"] = dataset["original_arg"]
    dataset["score"] = (dataset["answer"] == dataset["response"]).astype(int)
    return dataset


# molmo allenai/Molmo-7B-D-0924	Qwen/Qwen2-7B
# llava llava-hf/llava-1.5-7b-hf	lmsys/vicuna-7b-v1.5


def main():

    parser = build_parser()

    args = parser.parse_args()
    datasets = {}
    for csv_file in args.csv_files:

        dataset = pd.read_csv(csv_file, sep=None)
        print(f"Computing scores for {csv_file}")
        dataset = add_base_score(dataset)

        dataset = compute_hca_score(dataset)
        # __import__("ipdb").set_trace(context=31)
        # aggregated_df = dataset.groupby(["argument", "substitution"], as_index=False)[
        #     "hca"
        # ].mean()

        datasets[csv_file[:-4]] = dataset

    plot_dfs = []
    root_means = {}
    root_stds = {}
    for name, dataset in datasets.items():
        df = dataset
        idx = dataset.groupby("argument")["substitution_hop"].idxmax()
        final_sub_map = df.loc[idx].set_index("argument")["substitution"]
        df["final_sub"] = df["argument"].map(final_sub_map)
        agg = df.groupby("final_sub")[["hca", "score"]].agg(["mean", "std"])

        agg.columns = ["_".join(c) for c in agg.columns]
        plot_df = agg.sort_values("hca_mean", ascending=False)
        plot_df = plot_df.sort_index()
        mean_cols = ["hca_mean", "score_mean"]
        std_cols = ["hca_std", "score_std"]
        root_means[name] = plot_df[mean_cols]
        root_stds[name] = plot_df[std_cols]
        plot_dfs.append(plot_df)

    # f = open("hca-scores.csv", "w")
    # for name, rm in root_means.items():
    #     rm["hca_mean"] *= 100
    #     f.write(name)
    #     f.write(rm["hca_mean"].round(2).to_string(index=False, max_rows=100))

    # f = open("acc-scores.csv", "w")
    # for name, rm in root_means.items():
    #     rm["score_mean"] *= 100
    #     f.write(name)
    #     f.write(rm["score_mean"].round(2).to_string(index=False, max_rows=100))

    datasets = pd.concat(datasets, names=["source"]).reset_index(level=0)
    datasets.to_csv(args.output_dataset)
    __import__("ipdb").set_trace(context=31)
    models = [
        {
            "name": "llava",
            "vision_model": {
                "name": "llava",
                "model": "llava-hf/llava-1.5-7b-hf",
                "path": "data/model_outputs/0426_vlm_llava",
            },
            "language_model": {
                "name": "Vicuna",
                "model": "lmsys/vicuna-7b-v1.5",
                "path": "data/model_outputs/0426_lm_vicuna_7b_v1.5",
            },
        },
        {
            "name": "llava - next",
            "vision_model": {
                "name": "llava-next",
                "model": "llava-hf/llava-v1.6-mistral-7b-hf",
                "path": "data/model_outputs/0426_vlm_llava",
            },
            "language_model": {
                "path": "data/model_outputs/0426_lm_Mistral_7B_Instruct_v0.2",
                "model": "mistralaiMistral-7B-Instruct",
                "name": "Mistral-7B",
            },
        },
        {
            "name": "llava - one vision",
            "vision_model": {
                "name": "llava-onevision",
                "model": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
                "path": "data/model_outputs/0426_vlm_llava_ov",
            },
            "language_model": {
                "name": "Qwen2-7B-Instruct",
                "model": "Qwen/Qwen2-7B-Instruct",
                "path": "data/model_outputs/0426_lm_Qwen2.5_7B_Instruct",
            },
        },
        {
            "name": "mllama-instruct",
            "vision_model": {
                "name": "Llama-3.2",
                "model": "meta-llama/Llama-3.2-11B-Vision-Instruct",
                "path": "data/model_outputs/0426_vlm_mllama_instruct",
            },
            "language_model": {
                "path": "data/model_outputs/0426_lm_Llama_3.1_8B",
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "name": "Llama-3.1",
            },
        },
        {
            "name": "Qwen2.5VL",
            "vision_model": {
                "name": "Qwen2.5-VL",
                "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "path": "data/model_outputs/0426_vlm_qwen2.5VL",
            },
            "language_model": {
                "path": "data/model_outputs/0426_lm_Qwen2.5_7B",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "name": "Qwen2.5",
            },
        },
    ]


#     ax = means.plot(
#         kind="line",  # "barh" + xerr=errors for horizontal
# name         yerr=errors,  # .T → shape (n_series, n_categories)

#         capsize=4,  # little “hats” on the error bars
#         figsize=(10, 6),
#         rot=60,
#     )

# Questions
# 1. Are substitutions always made from leaf nodes?
# 2. If no, how do we compute the conditionals?
if __name__ == "__main__":
    main()
