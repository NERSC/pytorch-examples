#!/bin/bash
#SBATCH -C gpu
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time 30
#SBATCH -J train-pm
#SBATCH -o logs/%x-%j.out

# Setup software
module load pytorch/1.11.0
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Run the training
srun -l -u python train.py -d nccl --rank-gpu --ranks-per-node=${SLURM_NTASKS_PER_NODE} $@
