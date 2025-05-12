#!/bin/bash -l
#$ -P tin-lab  
#$ -l h_rt=12:00:00  
#$ -l gpus=1
#$ -l gpu_memory=80G
#$ -l gpu=1
#$ -o ../logs/calc_subsim_$JOB_ID.log
#$ -e ../logs/calc_subsim_$JOB_ID.err
#$ -N calc_subsim # name of your process


module load miniconda
module load cuda/11.8
source $(conda info --base)/etc/profile.d/conda.sh

conda activate venv
export HF_HOME=/projectnb/tin-lab/dvarghes

# echo the arguments passed from qsub
echo "Arguments passed from qsub: $ARGS"

# Run the python script with the arguments passed from qsub
python compute_taxonomy_sims_image.py $ARGS