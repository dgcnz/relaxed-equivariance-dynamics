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

# sbatch scripts/slurm/resume.sh wang2022/rotation/e2conv 653akrku /gpfs/home6/scur0399/development/dl2/logs/train/runs/2024-05-26_20-59-53/wang2022/653akrku/checkpoints/last.ckpt 12
srun python -m src.train experiment=$1 +logger.wandb.resume=must logger.wandb.id=$2 ckpt_path=$3 seed=$4
deactivate