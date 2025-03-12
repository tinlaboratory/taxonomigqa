import argparse
import pandas as pd
import matplotlib.pyplot as plt


def compute_cumulative_score(dataset):
    cumulative_scores = []
    prev_qid = -1
    prev_arg = ""
    cumulative_score = -1
    counter = 0
    # TODO groupby question_id
    for row in dataset.iterrows():
        row = row[1]
        q_id = row["question_id"]
        counter += 1
        if q_id != prev_qid:
            if prev_qid > 0:
                # cumulative_scores[prev_arg] = cumulative_score
                val = 0
                if counter:
                    val = cumulative_score / counter
                cumulative_scores.append((val, prev_arg, prev_qid))
                cumulative_score = -1
                counter = 0

        score = row["score"]

        if cumulative_score == -1:
            cumulative_score = score
        elif cumulative_score:
            cumulative_score += score
        prev_qid = q_id
        prev_arg = row["argument"]
    return pd.DataFrame(cumulative_scores, columns=["Cumulative", "attribute", "q_id"])
    # return cumulative_scores


def compute_leaf_score(dataset):
    leaf_scores = []
    prev_qid = -1
    prev_arg = ""
    leaf_score = 0
    # TODO groupby question_id
    counter = 0
    for row in dataset.iterrows():
        row = row[1]
        q_id = row["question_id"]
        counter += 1
        if q_id != prev_qid:
            if prev_qid > 0:
                # leaf_scores[prev_arg] = leaf_score
                val = 0
                if counter:
                    val = leaf_score / counter
                leaf_scores.append((val, prev_arg, prev_qid))
                leaf_score = 0
                counter = 0

        score = row["score"]

        leaf_score += score
        prev_qid = q_id
        prev_arg = row["argument"]
    return pd.DataFrame(leaf_scores, columns=["Acc", "attribute", "q_id"])
    # return leaf_scores


def compute_hca_score(dataset):
    hca_scores = []
    prev_qid = -1
    prev_arg = ""
    hca_score = 1
    # TODO groupby question_id
    for row in dataset.iterrows():
        row = row[1]
        q_id = row["question_id"]
        if q_id != prev_qid:
            if prev_qid > 0:
                # hca_scores[prev_arg] = hca_score
                hca_scores.append((hca_score, prev_arg, prev_qid))
                hca_score = 1

        score = row["score"]

        hca_score *= score
        prev_qid = q_id
        prev_arg = row["argument"]
    return pd.DataFrame(hca_scores, columns=["HCA", "attribute", "q_id"])
    # return hca_scores


def merge_scores(scores, score_name="HCA"):
    df_list = []
    for file_name, df in scores.items():
        df = df.copy()  # Ensure original DataFrame isn't modified
        df["model_name"] = file_name.split("/")[-1].replace(
            ".csv", ""
        )  # Extract filename
        df_list.append(df)

    # Merge all DataFrames
    merged_df = pd.concat(df_list, ignore_index=True)
    model_scores = merged_df.groupby(["model_name"])[score_name].mean()
    return model_scores


def compute_all_scores(args):

    # dataset = load_dataset("fgqa_hs", split="test[:5000]")

    hca_scores = {}
    leaf_scores = {}
    cumulative_scores = {}

    for csv_file in args.csv_files:

        dataset = pd.read_csv(csv_file, sep=None)
        dataset["model_output"] = dataset["model_output"].str.lower()

        # __import__("ipdb").set_trace(context=31)
        dataset["response"] = dataset["model_output"].str.contains("yes")
        dataset["answer"] = dataset["ground_truth"].str.contains("yes")
        dataset["substitution"] = dataset["argument"]
        dataset["argument"] = dataset["original_arg"]
        dataset.loc[:, "score"] = (dataset["answer"] == dataset["response"]).astype(int)

        dataset = dataset.sort_values(["question_id", "substitution_hop"])
        hca_scores[csv_file] = compute_hca_score(dataset)
        leaf_scores[csv_file] = compute_leaf_score(dataset)
        cumulative_scores[csv_file] = compute_cumulative_score(dataset)

    hca_model_scores = merge_scores(hca_scores)
    leaf_model_scores = merge_scores(leaf_scores, "Acc")
    cumulative_model_scores = merge_scores(cumulative_scores, "Cumulative")

    all_scores = pd.concat(
        [leaf_model_scores, cumulative_model_scores, hca_model_scores], axis=1
    )

    return all_scores


def build_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_files",
        nargs="*",
        default=["data/0302_llava_7B.csv"],
        help="Input file as csv",
    )
    parser.add_argument(
        "--tree_dir",
        default="trees",
        help="Directory to store all generated tree images.",
    )
    parser.add_argument(
        "--result_csv",
        default="scores.csv",
        help="Path to JSON file where results should be stored.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = build_arguments()
    all_scores = compute_all_scores(args)
    plt.figure(figsize=(8, 5))
    all_scores.plot(kind="bar", legend=True, colormap="viridis")
    all_scores.to_csv(args.result_csv)
    __import__("ipdb").set_trace(context=31)
