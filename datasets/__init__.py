"""
PyTorch dataset specifications.
"""

import importlib

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_datasets(name, **data_args):
    """Factory function for importing datasets from local modules"""
    module = importlib.import_module('.' + name, 'datasets')
    return module.get_datasets(**data_args)

def get_data_loaders(name, batch_size, distributed=False,
                     use_dist_sampler_train=True,
                     use_dist_sampler_valid=False,
                     **dataset_args):
    """Construct training and validation datasets and data loaders"""

    # Get the datasets
    train_dataset, valid_dataset, loader_args = get_datasets(name=name, **dataset_args)

    # Distributed samplers
    train_sampler, valid_sampler = None, None
    if distributed and use_dist_sampler_train:
        train_sampler = DistributedSampler(train_dataset)
    if distributed and use_dist_sampler_valid and valid_dataset is not None:
        valid_sampler = DistributedSampler(valid_dataset)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              shuffle=(train_sampler is None),
                              **loader_args)
    valid_loader = (DataLoader(valid_dataset, batch_size=batch_size,
                               sampler=valid_sampler,
                               **loader_args)
                    if valid_dataset is not None else None)
    return train_loader, valid_loader
