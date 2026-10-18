#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=predict_%j.out
#SBATCH --error=predict_%j.err

# Generate from a trained run and audit the samples (predict.py). Arguments pass through:
#   sbatch predict.sh --run_name wgan_adni64 --snapshots all
# Requires audit.py's outputs at the same resolution (outputs/audit/).

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch

cd "$SLURM_SUBMIT_DIR"
python -u predict.py "$@"

echo "Finished $(date)"