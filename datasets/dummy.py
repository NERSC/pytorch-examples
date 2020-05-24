"""
PyTorch dataset description for a random dummy dataset.
"""

# Compatibility
from __future__ import print_function

# Externals
import torch
from torch.utils.data import TensorDataset

def get_datasets(n_train=1024, n_valid=1024,
                 input_shape=[3, 32, 32], target_shape=[],
                 n_classes=None):
    """Construct and return random number datasets"""
    train_x = torch.randn([n_train] + input_shape)
    valid_x = torch.randn([n_valid] + input_shape)
    if n_classes is not None:
        train_y = torch.randint(n_classes, [n_train] + target_shape, dtype=torch.long)
        valid_y = torch.randint(n_classes, [n_valid] + target_shape, dtype=torch.long)
    else:
        train_y = torch.randn([n_train] + target_shape)
        valid_y = torch.randn([n_valid] + target_shape)
    train_dataset = TensorDataset(train_x, train_y)
    valid_dataset = TensorDataset(valid_x, valid_y)
    return train_dataset, valid_dataset, {}

def _test():
    t, v = get_datasets()
    for d in t.tensors + v.tensors:
        print(d.size())
