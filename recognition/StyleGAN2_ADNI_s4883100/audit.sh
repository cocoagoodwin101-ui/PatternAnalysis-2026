#!/bin/bash
#SBATCH --job-name=audit-ladder
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=audit_%j.out
#SBATCH --error=audit_%j.err

# Memorisation-audit calibration (audit.py part 1) on one A100.
# Submit from this folder:  sbatch audit.sh
# Extra arguments are passed through, e.g.  sbatch audit.sh --max_test_queries 500
# Logs (*.out, *.err) are gitignored.

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

source "$HOME/miniconda3/bin/activate"
conda activate torch

cd "$SLURM_SUBMIT_DIR"
python audit.py --resolution 64 "$@"

echo "Finished $(date)"