#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=DL2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=scripts/slurm_logs/slurm_output_%A.out

cd $HOME/development/dl2
source .venv/bin/activate
# run script from above
srun HYDRA_FULL_ERROR=1 python -m src.compute_measures_v2 spectrum=True ckpt_path=$1
deactivate
