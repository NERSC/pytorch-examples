"""
PyTorch dataset description for a random synthetic dataset.
"""

# Compatibility
from __future__ import print_function

# Externals
import torch

class RandomDataset(torch.utils.data.Dataset):
    """Random number synthetic dataset.

    For now, generating all requested samples up front.
    """

    def __init__(self, n_samples, input_shape, target_shape=None, n_classes=None):
        self.x = torch.randn([n_samples] + input_shape)
        self.y = None
        if target_shape is not None:
            if n_classes is not None:
                self.y = torch.randint(n_classes, [n_samples] + target_shape,
                                       dtype=torch.long)
            else:
                self.y = torch.randn([n_samples] + target_shape)

    def __getitem__(self, index):
        if self.y is not None:
            return self.x[index], self.y[index]
        else:
            return self.x[index]

    def __len__(self):
        return len(self.x)

def get_datasets(n_train, n_valid, input_shape, **kwargs):
    """Construct and return random number datasets"""
    train_dataset = RandomDataset(n_train, input_shape, **kwargs)
    if n_valid > 0:
        valid_dataset = RandomDataset(n_valid, input_shape, **kwargs)
    else:
        valid_dataset = None
    return train_dataset, valid_dataset, {}
