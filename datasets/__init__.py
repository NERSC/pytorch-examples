"""
PyTorch dataset specifications.
"""

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_datasets(name, **data_args):
    if name == 'dummy':
        from .dummy import get_datasets
        return get_datasets(**data_args)
    else:
        raise Exception('Dataset %s unknown' % name)

def get_data_loaders(name, batch_size, distributed=False,
                     use_dist_sampler_train=True,
                     use_dist_sampler_valid=False,
                     **dataset_args):

    # Get the datasets
    train_dataset, valid_dataset = get_datasets(name=name, **dataset_args)

    # Distributed samplers
    train_sampler, valid_sampler = None, None
    if distributed and use_dist_sampler_train:
        train_sampler = DistributedSampler(train_dataset)
    if distributed and use_dist_sampler_valid and valid_dataset is not None:
        valid_sampler = DistributedSampler(valid_dataset)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=train_sampler)
    valid_loader = (DataLoader(valid_dataset, batch_size=batch_size)
                    if valid_dataset is not None else None)
    return train_loader, valid_loader
