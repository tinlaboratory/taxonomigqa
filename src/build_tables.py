import argparse
import pandas as pd
from statsmodels.stats.weightstats import ztest as ztest

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
            "path": "data/model_outputs/0426_vlm_llava_next",
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


def filter_df(df, use_max=True):
    idx = df.groupby("argument")["substitution_hop"].idxmax()
    if not use_max:
        idx = df.groupby("argument")["substitution_hop"].idxmin()
    final_sub_map = df.loc[idx].set_index("argument")["substitution"]
    df["final_sub"] = df["argument"].map(final_sub_map)
    df = df[df["final_sub"] == df["substitution"]]
    return df


def compute_ztest(data, model_pair, use_max=True):
    lm = model_pair["language_model"]
    vlm = model_pair["vision_model"]

    df_lm = filter_df(data[data["source"] == lm["path"]], use_max)
    df_vlm = filter_df(data[data["source"] == vlm["path"]], use_max)

    hca_test = ztest(df_lm["hca"], df_vlm["hca"])
    acc_test = ztest(df_lm["score"], df_vlm["score"])

    return acc_test, hca_test


def filter_mean(means_lm):
    means_lm = means_lm.loc["mean"]

    means_lm = means_lm.loc[["mean_acc", "mean_hca"]]
    means_lm = means_lm.reset_index()
    means_lm = means_lm.T

    # If you want to keep the values as a single row DataFrame:
    means_lm = pd.DataFrame(means_lm.values[1], index=means_lm.values[0]).T
    means_lm = means_lm.rename(columns={"mean_acc": "acc", "mean_hca": "hca"})
    return means_lm


def compute_means(data, model_pair, withstds=True, grouping_col="substitution"):
    """
    Compute means and standard deviations for language and vision models.
    Merges acc_mean and acc_std with a plus-minus sign.
    """
    lm = model_pair["language_model"]
    vlm = model_pair["vision_model"]

    # Filter data for each model
    df_lm = data[data["source"] == lm["path"]]
    df_vlm = data[data["source"] == vlm["path"]]

    # Process language model data
    means_lm = df_lm.groupby(grouping_col)[["score", "hca"]].mean().agg(["mean", "std"])

    # Process vision model data
    means_vlm = (
        df_vlm.groupby(grouping_col)[["score", "hca"]].mean().agg(["mean", "std"])
    )

    # Merge acc_mean and acc_std with plus-minus sign
    means_lm["mean_acc"] = f"{means_lm['score']['mean'] * 100:.1f}"
    if withstds:
        means_lm["mean_acc"] += f"±{means_lm['score']['std'] * 100:.1f}"
    means_vlm["mean_acc"] = f"{means_vlm['score']['mean'] * 100:.1f}"
    if withstds:
        means_vlm["mean_acc"] += f"±{means_vlm['score']['std'] * 100:.1f}"
    means_lm["mean_hca"] = f"{means_lm['hca']['mean'] * 100:.1f}"
    if withstds:
        means_lm["mean_hca"] += f"±{means_lm['hca']['std'] * 100:.1f}"
    means_vlm["mean_hca"] = f"{means_vlm['hca']['mean'] * 100:.1f}"
    if withstds:
        means_vlm["mean_hca"] += f"±{means_vlm['hca']['std'] * 100:.1f}"

    # Merge acc_mean and acc_std with plus-minus sign
    # means_lm["mean_hca"] = (
    #     f"{means_lm['hca']['mean'] * 100:.1f}±{means_lm['hca']['std'] * 100:.1f}"
    # )
    # means_vlm["mean_hca"] = (
    #     f"{means_vlm['hca']['mean'] * 100:.1f}±{means_vlm['hca']['std'] * 100:.1f}"
    # )

    # __import__("ipdb").set_trace(context=31)
    means_lm = filter_mean(means_lm)
    means_vlm = filter_mean(means_vlm)

    return means_lm, means_vlm


def compute_tests(data, models, args):  # text_only=False, use_max=True):

    tests = {}
    for model_pair in models:
        if args.withtext:
            model_pair["vision_model"]["path"] = model_pair["vision_model"][
                "path"
            ].replace("_vlm_", "_vlm_text_")
        elif args.qonly:
            model_pair["vision_model"]["path"] = model_pair["vision_model"][
                "path"
            ].replace("_vlm_", "_vlm_q_only_")
        acc, hca = compute_ztest(data, model_pair, use_max=True)
        means_lm, means_vlm = compute_means(
            data, model_pair, withstds=not args.withoutstds
        )
        # __import__("ipdb").set_trace(context=31)
        print(means_lm)
        tests[model_pair["name"]] = {
            "acc_mean_lm": means_lm["acc"].iloc[0],
            "hca_mean_lm": means_lm["hca"].iloc[0],
            "acc_mean_vlm": means_vlm["acc"].iloc[0],
            "hca_mean_vlm": means_vlm["hca"].iloc[0],
            "acc_ztest_score": acc[0],
            "acc_ztest_pval": acc[1],
            "hca_ztest_score": hca[0],
            "hca_ztest_pval": hca[1],
        }
    return pd.DataFrame(tests).T


def style_dataframe(df, p_threshold=0.05):
    """
    Create a beautifully styled LaTeX table highlighting significant results.

    Highlights LM ACC when Z-test is positive and significant,
    otherwise highlights VLM ACC when the difference is significant.
    """
    # Create multi-level column headers for better organization
    df.columns = pd.MultiIndex.from_tuples(
        [
            ("LM", "ACC"),
            ("LM", "HCA"),
            ("VLM", "ACC"),
            ("VLM", "HCA"),
            ("Z-Test", "Eff. Size_{acc}"),
            ("Z-Test", "$p$"),
            ("Z-Test", "Eff. Size_{HCA}"),
            ("Z-Test", "$p$"),
        ]
    )

    # Create styling function for conditional formatting
    def highlight_significant_acc(row):
        # Get p-value for ACC comparison
        p_value = row[("Z-Test", "$p$")][0]  # First p-value is for ACC
        effect_size = row[("Z-Test", "Eff. Size_{acc}")]

        # Initialize empty style series
        styles = pd.Series("", index=row.index)

        # Apply bold to the winning model if result is significant
        if p_value < p_threshold:
            if effect_size > 0:  # Positive effect size means LM > VLM
                styles[("LM", "ACC")] = "font-weight: bold"
            else:  # Negative effect size means VLM > LM
                styles[("VLM", "ACC")] = "font-weight: bold"

        return styles

    # Apply formatting to numbers
    styled_df = df.style.format(
        {
            # Format numbers with 2 decimal places
            ("LM", "ACC"): "{:.2f}",
            ("LM", "HCA"): "{:.2f}",
            ("VLM", "ACC"): "{:.2f}",
            ("VLM", "HCA"): "{:.2f}",
            ("Z-Test", "Eff. Size_{acc}"): "{:.2f}",
            ("Z-Test", "Eff. Size_{HCA}"): "{:.2f}",
            # Format p-values with scientific notation if very small
            ("Z-Test", "$p$"): lambda x: (
                "{:.3f}".format(x) if x >= 0.001 else "{:.2e}".format(x)
            ),
        }
    )

    # Apply conditional styling based on significance
    styled_df = styled_df.apply(highlight_significant_acc, axis=1)

    # For LaTeX, we need to use a different approach for conditional formatting
    def get_latex_command(df):
        # Create a copy of the dataframe for LaTeX commands
        latex_df = df.copy()

        # Iterate through rows and apply bold formatting based on p-values
        for idx in latex_df.index:
            p_val = df.loc[idx, ("Z-Test", "$p$")][0]  # First p-value
            effect = df.loc[idx, ("Z-Test", "Eff. Size_{acc}")]

            if p_val < p_threshold:
                if effect > 0:
                    # Bold LM ACC value
                    val = latex_df.loc[idx, ("LM", "ACC")]
                    latex_df.loc[idx, ("LM", "ACC")] = f"\\textbf{{{val:.2f}}}"
                else:
                    # Bold VLM ACC value
                    val = latex_df.loc[idx, ("VLM", "ACC")]
                    latex_df.loc[idx, ("VLM", "ACC")] = f"\\textbf{{{val:.2f}}}"

        return latex_df

    # Get LaTeX-ready dataframe with conditional formatting
    latex_df = get_latex_command(df)

    # Generate the LaTeX table with proper settings
    with open("scores.tex", "w") as f:
        latex_df.to_latex(
            buf=f,
            multicolumn=True,
            multicolumn_format="c",
            multirow=True,
            na_rep="--",
            escape=False,
            index=True,
            float_format="%.2f",
        )

    return styled_df


# def style_dataframe(df, output_path="scores.tex"):
#     """Create a beautifully styled LaTeX table from dataframe with test results."""
#     # Create multi-level column headers for better organization
#     df.columns = pd.MultiIndex.from_tuples(
#         [
#             ("LM", "ACC"),
#             ("LM", "HCA"),
#             ("VLM", "ACC"),
#             ("VLM", "HCA"),
#             ("Z-Test", "Eff. Size_{acc}"),
#             ("Z-Test", "$p$"),
#             ("Z-Test", "Eff. Size_{HCA}"),
#             ("Z-Test", "$p$"),
#         ]
#     )

#     # Apply styling - pandas has built-in styling capabilities
#     styled_df = df.style.format(
#         {
#             # Format numbers with 2 decimal places
#             ("LM", "ACC"): "{:.2f}",
#             ("LM", "HCA"): "{:.2f}",
#             ("VLM", "ACC"): "{:.2f}",
#             ("VLM", "HCA"): "{:.2f}",
#             ("Z-Test", "Eff. Size_{acc}"): "{:.2f}",
#             ("Z-Test", "$p$"): "{:.3f}",  # Use 3 decimals for p-values
#             ("Z-Test", "Eff. Size_{HCA}"): "{:.2f}",
#             # Format p-values with scientific notation if very small
#             ("Z-Test", "$p$"): lambda x: (
#                 "{:.3f}".format(x) if x >= 0.001 else "{:.2e}".format(x)
#             ),
#         }
#     ).highlight_max(
#         axis=0,
#         subset=[("LM", "ACC"), ("LM", "HCA"), ("VLM", "ACC"), ("VLM", "HCA")],
#         props="textbf:--rwrap;",  # Bold the max values
#     )

#     # Generate the LaTeX table with proper settings
#     with open(output_path, "w") as f:
#         styled_df.to_latex(
#             f,
#             multicolumn=True,
#             multicolumn_format="c",
#             multirow=True,
#             na_rep="--",
#             escape=False,
#             index=True,
#         )

#     return styled_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--withtext", action="store_true")
    parser.add_argument("--withoutstds", action="store_true")
    parser.add_argument("--qonly", action="store_true")
    args = parser.parse_args()
    data = pd.read_csv("datasets_with_scores.csv")

    df = compute_tests(data, models, args)
    # Create multi-level column headers for better organization
    suffix = ""
    if args.withtext:
        suffix = "_{text}"
    df.columns = pd.MultiIndex.from_tuples(
        [
            (f"LM{suffix}", "ACC"),
            (f"LM{suffix}", "HCA"),
            (f"VLM{suffix}", "ACC"),
            (f"VLM{suffix}", "HCA"),
            ("Z-Test", "Eff. Size_{acc}"),
            ("Z-Test", "$p$"),
            ("Z-Test", "Eff. Size_{HCA}"),
            ("Z-Test", "$p$"),
        ]
    )
    latex_table = df.to_latex(
        open("scores.tex", "w"),
        float_format="%.2f",
        escape=False,
        multicolumn=True,
        multicolumn_format="c",
        multirow=True,
        bold_rows=False,
        na_rep="--",
        index=True,
        # caption="Comparison of model performance and statistical significance tests.",
        caption="Accuracy and HCA scores for each model pair, including effect size and $p$-value of a Z-test.",
        label="tab:model_scores",
        column_format="lcccccccc",
    )
    __import__("ipdb").set_trace(context=31)
