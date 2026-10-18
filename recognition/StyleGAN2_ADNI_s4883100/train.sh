#!/bin/bash
#SBATCH --job-name=train-gan
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# Train a model with train.py on one A100. Submit from this folder; arguments pass through:
#   sbatch train.sh --dataset oasis --iters 200 --run_name smoke_oasis
#   sbatch train.sh --dataset oasis --run_name wgan_oasis64
#   sbatch train.sh --dataset adni --run_name wgan_adni64
# If a run hits the time limit, resubmit the same command with --resume.
# No --mem: comp3710 nodes expose no schedulable memory (README Section 7.5).

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch

cd "$SLURM_SUBMIT_DIR"
python -u train.py "$@"

echo "Finished $(date)"