#!/bin/bash
#SBATCH --job-name=latent
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=latent_%j.out
#SBATCH --error=latent_%j.err

# Latent interpolation and inversion membership test (latent.py). Arguments pass through:
#   sbatch latent.sh --run_name sg2_adni64 --iter 10000
# Needs audit.py's outputs at the same resolution (outputs/audit/).

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch

cd "$SLURM_SUBMIT_DIR"
python -u latent.py "$@"

echo "Finished $(date)"