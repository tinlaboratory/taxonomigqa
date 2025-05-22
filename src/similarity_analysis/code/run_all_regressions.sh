#!/bin/bash

# Script to launch all combinations of linear regressions
CSV_FILES=(
    "../data/llava_vlm-text_substituted_edge_accuracy_lhs.csv"
    "../data/llava_vlm-text_substituted_edge_accuracy_po.csv" 
    "../data/llava_vlm_substituted_edge_accuracy_lhs.csv"
    "../data/llava_vlm_substituted_edge_accuracy_po.csv"
    "../data/vicuna_7b_v1.5_lm_substituted_edge_accuracy_lhs.csv"
    "../data/vicuna_7b_v1.5_lm_substituted_edge_accuracy_po.csv"
)

for csv in "${CSV_FILES[@]}"; do
    echo "Processing $csv"

    # 1. No groupby, no pairwise
    python run_regression.py --csv_path "$csv"

    # 2. No groupby, with pairwise
    python run_regression.py --csv_path "$csv" --use_pairwise_sim

    # 3. With groupby, no pairwise
    python run_regression.py --csv_path "$csv" --groupby

    # 4. With groupby, with pairwise
    python run_regression.py --csv_path "$csv" --groupby --use_pairwise_sim
done