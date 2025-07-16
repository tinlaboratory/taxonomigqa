#!/bin/bash -l
#$ -P tin-lab       # Specify the SCC project name you want to use
#$ -l h_rt=6:00:00   # Specify the hard time limit for the job
#$ -N model_inference         # Give job a name
#$ -o /projectnb/tin-lab/yuluq/data/subset_combined_stats_data/output_file/testing_2b.txt             # output file name
#$ -l gpus=1
#$ -l gpu=1
#$ -l gpu_memory=80G
#$ -j y               # Merge the error and output streams into a single file

module load miniconda
conda activate /projectnb/tin-lab/yuluqin/multimodal_semantic
module load cuda/11.8
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python /projectnb/tin-lab/yuluq/multimodal-representations/src/prompting/run_inference.py --config="$1"
