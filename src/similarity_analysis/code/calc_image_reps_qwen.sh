#!/bin/bash -l
#$ -P tin-lab  
#$ -l h_rt=05:00:00  
#$ -l gpus=1
#$ -l gpu_memory=40G
#$ -l gpu=1
#$ -o ../logs/calc_image_reps_qwen_$JOB_ID.log
#$ -e ../logs/calc_image_reps_qwen_$JOB_ID.err
#$ -N calc_image_reps_qwen 


module load miniconda
module load cuda/11.8
source $(conda info --base)/etc/profile.d/conda.sh

conda activate venv
export HF_HOME=/projectnb/tin-lab/dvarghes

python compute_taxonomy_sims_image.py --nonleaf_out_pkl ../data/qwen_nl_node_to_embeds.pkl --leaf_out_pkl ../data/qwen_leaf_node_to_embeds.pkl --sim_csv_out ../data/qwen_substituted_edge_accuracy.csv --model Qwen --model_type vlm-text