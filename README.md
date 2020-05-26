# NERSC PyTorch examples

This repository contains some PyTorch example models and training code
with support for distributed training on NERSC systems.

The layout of this package can also serve as a template for PyTorch
projects and the provided BaseTrainer and train.py script can be used to
reduce boiler plate.

## Package layout

The directory layout of this repo is designed to be flexible:
- Configuration files (in YAML format) go in `configs/`
- Dataset specifications using PyTorch's Dataset API go into `datasets/`
- Model implementations go into `models/`
- Trainer implementations go into `trainers/`. Trainers inherit from
  `BaseTrainer` and are responsible for constructing models as well as training
  and evaluating them.

All examples are run with the generic training script, `train.py`.

## Examples

This package currently contains the following examples:
- Minimal Hello World example.

## How to run

To run the examples on the Cori supercomputer, you may use the provided
example SLURM batch script. Here's how to run the Hello World example on 4
Haswell nodes:

`sbatch -N 4 scripts/train_cori.sh configs/hello.yaml`
