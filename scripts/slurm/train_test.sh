#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=DL2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:00:
#SBATCH --output=scripts/slurm_logs/slurm_output_%A.out

cd $HOME/development/dl2
source .venv/bin/activate
# run script from above
srun python train.py --experiment=configs/experiment/wang2022/equivariance_test/convnet --trainer.max_epochs=1
deactivate