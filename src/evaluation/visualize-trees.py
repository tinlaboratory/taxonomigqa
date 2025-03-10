import json
import random
import argparse
import pandas as pd

import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

from datasets import load_dataset

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def graph_to_heatmap(G, file_path):
    """
    Converts a directed NetworkX graph into a heatmap representation.

    :param G: NetworkX DiGraph where edges represent (argument, substitution, score).
    :param file_path: Path to save the heatmap image.
    """
    # Extract unique nodes (arguments + substitutions) for indexing
    nodes = G.nodes

    # Create an adjacency matrix initialized with NaNs (for missing edges)
    adjacency_matrix = pd.DataFrame(np.nan, index=nodes, columns=nodes)

    # Fill the adjacency matrix with edge weights (scores)
    for u, v, data in G.edges(data=True):
        adjacency_matrix.loc[u, v] = data.get("weight", 0)

    # Convert NaNs to 0 for visualization
    adjacency_matrix.fillna(0, inplace=True)

    # Plot heatmap
    plt.figure(figsize=(15, 12))
    plt.imshow(adjacency_matrix, cmap="hot", interpolation="nearest")
    plt.colorbar(label="Score")

    # Add labels
    plt.xticks(ticks=range(len(nodes)), labels=nodes, rotation=90)
    plt.yticks(ticks=range(len(nodes)), labels=nodes)
    plt.title("Graph Heatmap (Argument → Substitution Scores)")

    # Save the figure instead of displaying it
    plt.savefig(file_path, format="png", dpi=300)
    plt.close()

    print(f"Heatmap saved to {file_path}")


def graph_to_heatmap_minimized(G, file_path):
    """
    Converts a directed NetworkX graph into a heatmap representation where:
    - The y-axis represents arguments.
    - The x-axis represents substitutions (keeping their order).

    :param G: NetworkX DiGraph where edges represent (argument, [sub1, sub2, ...], scores).
    :param file_path: Path to save the heatmap image.
    """
    # Extract unique arguments (nodes that have outgoing edges)
    arguments = sorted(set(u for u, v in G.edges))

    # Extract unique substitutions (nodes that have incoming edges) in the order they appear
    substitutions = []
    for _, v in G.edges:
        if v not in substitutions:
            substitutions.append(v)

    # Create an adjacency matrix initialized with NaNs (for missing edges)
    adjacency_matrix = pd.DataFrame(np.nan, index=arguments, columns=substitutions)

    # Fill the adjacency matrix with edge weights (scores)
    for u, v, data in G.edges(data=True):
        adjacency_matrix.loc[u, v] = data.get("weight", 0)

    # Convert NaNs to 0 for visualization
    adjacency_matrix.fillna(0, inplace=True)

    # Plot heatmap
    plt.figure(figsize=(15, 12))
    plt.imshow(adjacency_matrix, cmap="hot", interpolation="nearest")
    plt.colorbar(label="Score")

    # Add labels (preserving order)
    plt.xticks(ticks=range(len(substitutions)), labels=substitutions, rotation=90)
    plt.yticks(ticks=range(len(arguments)), labels=arguments)
    plt.title("Graph Heatmap (Arguments → Ordered Substitutions)")

    # Save the figure instead of displaying it
    plt.savefig(file_path, format="png", dpi=300)
    plt.close()

    print(f"Heatmap saved to {file_path}")


def edges_to_heatmap(edges, file_path):
    """
    Converts a directed graph's edges into a heatmap representation.

    :param edges: List of (argument, substitution, score) tuples.
    :param file_path: Path to save the heatmap image.
    """
    # Extract unique nodes (arguments + substitutions) for indexing
    nodes = list(set([arg for arg, _, _ in edges] + [sub for _, sub, _ in edges]))

    # Create an adjacency matrix initialized with NaNs (for missing edges)
    adjacency_matrix = pd.DataFrame(np.nan, index=nodes, columns=nodes)

    # Populate the matrix with scores
    for arg, sub, score in edges:
        adjacency_matrix.loc[arg, sub] = score

    # Convert NaN values to 0 for visualization
    adjacency_matrix.fillna(0, inplace=True)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(adjacency_matrix, cmap="hot", interpolation="nearest")
    plt.colorbar(label="Score")

    # Set labels
    plt.xticks(ticks=range(len(nodes)), labels=nodes, rotation=90)
    plt.yticks(ticks=range(len(nodes)), labels=nodes)
    plt.title("Graph Edge Heatmap")

    # Save the heatmap instead of showing it
    plt.savefig(file_path, format="png", dpi=300)
    plt.close()

    print(f"Heatmap saved to {file_path}")


def load_jsonl(file_path):
    """
    Loads a JSONL file into a Pandas DataFrame.

    :param file_path: Path to the .jsonl file
    :return: DataFrame containing the JSONL data
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    return pd.DataFrame(data)


def get_root_node(G):
    """
    Finds the root node(s) in a directed graph.
    The root is defined as the node with no incoming edges (in_degree == 0).

    :param G: Directed NetworkX graph (DiGraph)
    :return: List of root nodes (since there can be multiple disconnected trees)
    """
    root_nodes = [node for node in G.nodes if G.out_degree(node) == 0]
    return root_nodes

    # pos = nx.spring_layout(G)
    # pos = {
    #     k: (
    #         v[0] - min(v[0] for v in pos.values()),
    #         v[1] - min(v[1] for v in pos.values()),
    #     )
    #     for k, v in pos.items()
    ## Draw edges with transparency
    # for (u, v, d), width, alpha in zip(G.edges(data=True), edge_widths, edge_alpha):
    #     nx.draw_networkx_edges(
    #         G, pos, edgelist=[(u, v)], width=width, alpha=alpha
    # }
    threshold = 0.7
    edge_widths = [d["weight"] * 2 for _, _, d in edges]  # Scale edge thickness
    edge_alpha = [d["weight"] for _, _, d in edges]  # Directly use scores for alpha


def visualize_tree(
    G,
    file_path,
    root=None,
):
    """
    Visualizes the tree structure using NetworkX and Matplotlib.
    - Scores are displayed as edge labels.
    - Edge transparency (alpha) is determined by the score.

    :param G: Directed NetworkX graph (DiGraph)
    :param root: Root node of the tree. If None, tries to find one automatically.
    """
    if root is None:
        # Find root (node with no incoming edges)
        root_candidates = [node for node in G.nodes if G.in_degree(node) == 0]
        root = root_candidates[0] if root_candidates else None

    if root is None:
        print("No root found in the graph.")
        return

    # Compute layout using hierarchical positioning
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")

    # Get edge weights (scores) for labeling
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

    # Determine transparency based on scores (higher score = more visible, lower score = more transparent)
    edges = G.edges(data=True)
    edge_colors = [
        "black" if d["weight"] > 0.8 else "blue" if d["weight"] >= 0.6 else "red"
        for _, _, d in edges
    ]

    # Draw the graph
    plt.figure(figsize=(14, 7))

    # Draw nodes and edges
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color=edge_colors,
        node_size=2000,
        font_size=10,
        alpha=0.9,
    )

    # Add edge labels (scores)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=9, label_pos=0.5
    )
    plt.tight_layout()

    plt.title("Argument-Substitution Tree with Scores")
    plt.savefig(file_path, format="png", dpi=300)
    plt.close()


def visualize_trees(trees, file_path, args):
    num_graphs = len(trees)
    cols = min(1, num_graphs)  # Limit to 3 columns per row
    rows = (num_graphs + cols - 1) // cols  # Calculate number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten() if num_graphs > 1 else [axes]

    for i, G in enumerate(trees):
        name, G = G
        # print(f"Model {name} for root {file_path}")
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")
        # pos = nx.spring_layout(G)
        ax = axes[i]

        # Get edge weights (scores) for labeling
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

        # Determine transparency based on scores
        edges = G.edges(data=True)
        edge_colors = [
            "black" if d["weight"] > 0.8 else "blue" if d["weight"] >= 0.6 else "red"
            for _, _, d in edges
        ]

        # Draw graph
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            node_color="lightblue",
            edge_color=edge_colors,
            node_size=2000,
            font_size=10,
            alpha=0.9,
        )

        # Add edge labels (scores)
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=9, ax=ax, label_pos=0.5
        )

        title = f"Model - {name}"
        if args.conditional_accuracy:
            title += " Conditional"
        if args.cumulative_scoring:
            title += " Cumulative"

        ax.set_title(title)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(file_path, format="png", dpi=300)
    plt.close()


def build_graph(edges):
    """
    Builds a directed tree graph from a list of (argument, substitution, score) edges.

    :param edges: List of tuples (argument, substitution, score)
    :return: A directed NetworkX graph (DiGraph)
    """
    G = nx.DiGraph()

    for argument, substitution, score in edges:
        G.add_edge(argument, substitution, weight=score)

    return G


def compute_conditional_pairwise_accuracy(df, filter_conditional=True):
    """
    Computes the conditional pairwise accuracy of argument-substitution pairs,
    conditioned on the answerness of the original question, using original_question_id to group by the original question.

    :param df: Pandas DataFrame with columns:
        - 'id': Unique identifier for the substituted question
        - 'question': The question text (either original or substituted)
        - 'argument': The main noun in the original question
        - 'substitution': The substituted noun
        - 'original_question_id': The identifier linking substitutions to the same original question
        - 'answer': Boolean indicating whether the answer to the question was answer
    :return: DataFrame with accuracy scores for each (argument, substitution) pair.
    """

    # Step 1: Get the answerness of the original question
    original_answerness = df[
        df.duplicated("original_question_id", keep="first") == False
    ][["original_question_id", "answer"]]
    original_answerness.rename(columns={"answer": "original_answer"}, inplace=True)

    # Step 2: Merge back to get original answerness on the substitutions
    df = df.merge(original_answerness, on="original_question_id", how="left")

    # Step 3: Filter only those rows where the original question was answer
    filtered_df = df
    if filter_conditional:
        filtered_df = df[df["original_answer"] == True]

    # Step 4: Calculate pairwise accuracy for each (argument, substitution) pair
    pairwise_acc = (
        filtered_df.groupby(["argument", "substitution"])["answer"]
        .mean()
        .reset_index()
        .rename(columns={"answer": "conditional_accuracy"})
    )

    return pairwise_acc


def build_pairwise(results, hypernyms):
    # Assumes columns ['question', 'response', 'answer', 'argument', 'substitution']

    results.loc[:, "score"] = (results["answer"] == results["response"]).astype(int)
    aggregated_base_questions = (
        results[results["substitution"] == ""]
        .groupby("argument")
        .agg(
            {
                "score": "mean",
                #  'rouge/score': 'mean'
            }
        )
        .reset_index()
    )

    aggregated_substitutions = (
        results[results["substitution"] != ""]
        .groupby(["argument", "substitution"])
        .agg(
            {
                "score": "mean",
            }
        )
        .reset_index()
    )

    # Merge the two dataframes on a common key (in this case, 'key')
    aggregated_combined = aggregated_base_questions.rename(
        columns={
            "score": "base/score",
        }
    )
    aggregated_combined = pd.merge(
        aggregated_combined, aggregated_substitutions, on="argument", how="left"
    )

    # Fill empty values with 0.0
    aggregated_combined = aggregated_combined.fillna(0.0)
    return aggregated_combined


def build_parser():
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
        "--conditional_accuracy",
        action="store_true",
        help="Directory to store all generated tree images.",
    )
    parser.add_argument(
        "--cumulative_scoring",
        action="store_true",
        help="Directory to store all generated tree images.",
    )
    return parser


def main():

    # dataset = load_dataset("fgqa_hs", split="test[:5000]")

    parser = build_parser()

    concept_forests = {}

    args = parser.parse_args()
    for csv_file in args.csv_files:

        dataset = pd.read_csv(csv_file, sep=None)
        dataset["model_output"] = dataset["model_output"].str.lower()

        # __import__("ipdb").set_trace(context=31)
        dataset["response"] = dataset["model_output"].str.contains("yes")
        dataset["answer"] = dataset["ground_truth"].str.contains("yes")
        dataset["substitution"] = dataset["argument"]
        dataset["argument"] = dataset["original_arg"]

        hypernyms = json.load(open("../../data/gqa_entities/noun-hypernyms.json"))

        df = build_pairwise(dataset, hypernyms)

        edges = []
        skip = False
        prev_score = {}
        # TODO sort by q_id and hop?
        dataset = dataset.sort_values(["question_id", "substitution_hop"])
        for row in dataset.iterrows():
            row = row[1]
            q_id = row["question_id"]
            if q_id not in prev_score:
                prev_score[q_id] = 1
            score = row["score"]
            # __import__("ipdb").set_trace(context=31)
            if args.conditional_accuracy:
                score *= prev_score[q_id]
            elif args.cumulative_scoring:
                score *= prev_score[q_id]
                prev_score[q_id] = row["score"]

            argument = row["argument"]
            substitution = row["substitution"]
            if argument == substitution:
                if args.conditional_accuracy:
                    prev_score[q_id] = row["score"]
                skip = False
                if not score:
                    skip = True
                prev = argument
                continue
            if not skip:
                edges.append((prev, substitution, score))
            prev = substitution
        edges = pd.DataFrame(edges, columns=["argument", "substitution", "score"])
        aggregated_df = edges.groupby(["argument", "substitution"], as_index=False)[
            "score"
        ].mean()
        edge_list = []
        for row in aggregated_df.iterrows():
            edge_list.append((row[1][0], row[1][1], row[1][2]))

        # Example setup: assume 'category' and 'substitution_type' columns exist
        G = build_graph(edge_list)
        forest = [G.subgraph(comp).copy() for comp in nx.weakly_connected_components(G)]

        # Generate heatmap and save it
        file_path = "graph_heatmap.png"
        for tree in forest:
            # __import__("ipdb").set_trace(context=31)
            # graph_to_heatmap(tree, file_path)
            name = get_root_node(tree)[0]
            existing_concepts = concept_forests.get(name, [])
            existing_concepts.append([csv_file, tree])
            concept_forests[name] = existing_concepts

    # __import__("ipdb").set_trace(context=31)
    for root_name, trees in concept_forests.items():
        print("Visualizing ", root_name)
        visualize_trees(trees, f"{args.tree_dir}/{root_name}-tree.jpg", args)


def compute_pivots(df):
    # pivot_table = df.pivot_table(
    #     index="argument",
    #     columns=["argument", "substitution"],
    #     values="score",
    #     # aggfunc="mean",
    # )
    # pairwised_df = compute_conditional_pairwise_accuracy(df)

    # Plotting heatmap
    # plt.figure(figsize=(10, 6))
    # sns.heatmap(pivot_table, annot=True, cmap="coolwarm")
    # plt.title("Accuracy Heatmap by Category and Substitution Type")
    # plt.xlabel("Category")
    # plt.ylabel("Substitution Type")
    # plt.show()

    # pivot_table = aggregated_df.pivot_table(
    #     index="argument",
    #     columns=["substitution"],
    #     values="score",
    #     # aggfunc="mean",
    # )
    # # pairwised_df = compute_conditional_pairwise_accuracy(df)

    # # Plotting heatmap
    # plt.figure(figsize=(10, 6))
    # sns.heatmap(pivot_table, annot=True, cmap="coolwarm")
    # plt.title("Accuracy Heatmap by Category and Substitution Type")
    # plt.xlabel("Category")
    # plt.ylabel("Substitution Type")
    # plt.show()
    pass


# Questions
# 1. Are substitutions always made from leaf nodes?
# 2. If no, how do we compute the conditionals?
if __name__ == "__main__":
    main()
