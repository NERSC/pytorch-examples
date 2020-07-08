#!/bin/bash
#SBATCH -C gpu
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --time 4:00:00
#SBATCH -J train-cgpu
#SBATCH -o logs/%x-%j.out

# Setup software
module load pytorch/v1.5.1-gpu

# Run the training
srun -l -u python train.py -d nccl --rank-gpu $@
