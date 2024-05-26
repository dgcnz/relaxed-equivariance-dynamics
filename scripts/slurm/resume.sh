#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=DL2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=scripts/slurm_logs/slurm_output_%A.out

cd $HOME/development/dl2
source .venv/bin/activate
# run script from above
srun python -m src.train experiment=$1 +logger.wandb.resume=must logger.wandb.id=$2 ckpt_path=$3 seed=$4
deactivate