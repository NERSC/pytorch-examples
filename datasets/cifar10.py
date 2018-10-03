"""
PyTorch dataset code for Cifar10
"""

# System
import os

# Externals
import torchvision
import torchvision.transforms as transforms

def get_datasets(data_path):
    """
    Get the training and test datasets
    """
    data_path = os.path.expandvars(data_path)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                           download=True, transform=transform)
    return trainset, testset
