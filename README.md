# NERSC PyTorch examples

This repository contains some PyTorch example models and training code
with support for distributed training on NERSC systems.

The layout of this package can also serve as a template for PyTorch
projects and the provided BaseTrainer and main.py script can be used to
reduce boiler plate.

## Package layout

The directory layout of this repo is designed to be flexible:
- Configuration files (in YAML format) go in `configs/`
- Dataset specifications using PyTorch's Dataset API go into `datasets/`
- Model definitions go into `models/`
- Trainers (see below) go into `trainers/`
