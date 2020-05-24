"""
This module contains the code to retrieve the MNIST dataset using the
existing PyTorch Dataset specification in torchvision.
"""

# System
import os

# Externals
import torch
import torchvision
import torchvision.transforms as transforms

def get_datasets(data_path):
    data_path = os.path.expandvars(data_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(root=data_path, train=True,
                                               download=True, transform=transform)
    valid_dataset = torchvision.datasets.MNIST(root=data_path, train=False,
                                               download=True, transform=transform)
    return train_dataset, valid_dataset, {}
