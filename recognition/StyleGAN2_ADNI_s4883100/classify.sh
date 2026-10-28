#!/bin/bash
#SBATCH --job-name=classify
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=classify_%j.out
#SBATCH --error=classify_%j.err

# AD/CN classifier for class fidelity of generated cohorts (classify.py). Arguments pass through:
#   sbatch classify.sh --runs sg2_adni64:10000 sg2_adni64_augp02:17500 sg2_adni64_cond:15100
# Trains the classifier (about 10 min) unless --eval_only is given.

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch

cd "$SLURM_SUBMIT_DIR"
python -u classify.py "$@"

echo "Finished $(date)"