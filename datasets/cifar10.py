"""
PyTorch dataset code for Cifar10
"""

# System
import os

# Externals
import torch
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms

def get_datasets(data_path, n_train=None, n_valid=None):
    """
    Get the training and test datasets
    """
    data_path = os.path.expandvars(data_path)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=True, transform=transform)
    validset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                            download=True, transform=transform)

    # Generate random subset of the training dataset
    total_train = len(trainset)
    if n_train is not None and n_train > 0 and n_train < total_train:
        trainset = Subset(trainset, torch.randperm(n_train))

    # Generate random subset of the validation dataset
    total_valid = len(validset)
    if n_valid is not None and n_valid > 0 and n_valid < total_valid:
        validset = Subset(validset, torch.randperm(n_valid))

    return trainset, validset
