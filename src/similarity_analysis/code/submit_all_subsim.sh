#!/bin/bash

# Array of argument sets for each job
ARGS_LIST=(
  "--nonleaf_out_pkl ../data/nl_node_to_embeds_po.pkl --leaf_out_pkl ../data/leaf_node_to_embeds_po.pkl --sim_csv_out ../data/llava_vlm-text_substituted_edge_accuracy_po.csv --model llava --model_type vlm-text"
  "--nonleaf_out_pkl ../data/nl_node_to_embeds_po.pkl --leaf_out_pkl ../data/leaf_node_to_embeds_po.pkl --sim_csv_out ../data/llava_vlm_substituted_edge_accuracy_po.csv --model llava --model_type vlm"
  "--nonleaf_out_pkl ../data/nl_node_to_embeds_po.pkl --leaf_out_pkl ../data/leaf_node_to_embeds_po.pkl --sim_csv_out ../data/vicuna_7b_v1.5_lm_substituted_edge_accuracy_po.csv --model vicuna_7b_v1.5 --model_type lm"
  "--nonleaf_out_pkl ../data/nl_node_to_embeds.pkl --leaf_out_pkl ../data/leaf_node_to_embeds.pkl --sim_csv_out ../data/llava_vlm-text_substituted_edge_accuracy_lhs.csv --model llava --model_type vlm-text"
  "--nonleaf_out_pkl ../data/nl_node_to_embeds.pkl --leaf_out_pkl ../data/leaf_node_to_embeds.pkl --sim_csv_out ../data/llava_vlm_substituted_edge_accuracy_lhs.csv --model llava --model_type vlm"
  "--nonleaf_out_pkl ../data/nl_node_to_embeds.pkl --leaf_out_pkl ../data/leaf_node_to_embeds.pkl --sim_csv_out ../data/vicuna_7b_v1.5_lm_substituted_edge_accuracy_lhs.csv --model vicuna_7b_v1.5 --model_type lm"
)

for i in "${!ARGS_LIST[@]}"; do
  qsub -N calc_subsim_$i -o ../logs/calc_subsim_$i.log -e ../logs/calc_subsim_$i.err -v ARGS="${ARGS_LIST[$i]}" calc_subsim.sh
done

# run this script with the following command:
# bash submit_all_subsim.sh