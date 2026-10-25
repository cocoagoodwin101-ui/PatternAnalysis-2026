#!/bin/bash
#SBATCH --job-name=embed
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=embed_%j.out
#SBATCH --error=embed_%j.err

# Joint t-SNE embedding of real and generated slices (embed.py). Arguments pass through:
#   sbatch embed.sh --runs wgan_adni64:10000 sg2_adni64:10000
# Needs scikit-learn in the torch environment (t-SNE runs on the CPUs).

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch

cd "$SLURM_SUBMIT_DIR"
python -u embed.py "$@"

echo "Finished $(date)"