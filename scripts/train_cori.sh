#!/bin/bash
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -J train-cori
#SBATCH -o logs/%x-%j.out

# Setup software
module load pytorch/v1.5.0
export OMP_NUM_THREADS=32
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1

# Run the training
srun -l -u python train.py -d mpi $@
