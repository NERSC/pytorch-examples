#!/bin/bash
#SBATCH -C gpu
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --time 30
#SBATCH -J train-cgpu
#SBATCH -o logs/%x-%j.out

# Setup software
module load pytorch/1.10.2-gpu
# Workaround for cudnn module lib path bug
export LD_LIBRARY_PATH=/usr/common/software/sles15_cgpu/cudnn/8.3.2/lib:$LD_LIBRARY_PATH

# Run the training
srun -l -u python train_pl.py $@
