#!/bin/bash
#SBATCH --job-name=autopsy
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=autopsy_%j.out
#SBATCH --error=autopsy_%j.err

# Rule-based failure autopsy and uncurated grids (autopsy.py). Arguments pass through:
#   sbatch autopsy.sh --run_name sg2_adni64 --iter 10000 --copy_iter 12500
# Needs audit.py's outputs at the same resolution (outputs/audit/).

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch

cd "$SLURM_SUBMIT_DIR"
python -u autopsy.py "$@"

echo "Finished $(date)"