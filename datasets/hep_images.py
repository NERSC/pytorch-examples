"""
ATLAS Delphes images for RPV (''HEP-CNN'') classification task.
"""

# Externals
import numpy as np
import h5py
import torch
from torch.utils.data import TensorDataset

def _load_file(filename, n_samples):
    """Load one file from the dataset"""
    with h5py.File(filename, 'r') as f:
        data_group = f['all_events']
        data = data_group['hist'][:n_samples][:,None,:,:].astype(np.float32)
        labels = data_group['y'][:n_samples].astype(np.float32)
        weights = data_group['weight'][:n_samples].astype(np.float32)
    return data, labels, weights

def get_dataset(input_file, n_samples, include_weights=False):
    """Prepare a dataset for one file"""
    x, y, w = _load_file(input_file, n_samples)
    np_inputs = [x, y, w] if include_weights else [x, y]
    tensors = [torch.from_numpy(a) for a in np_inputs]
    return TensorDataset(*tensors)

def get_datasets(train_file, valid_file, n_train, n_valid):
    train_dataset = get_dataset(train_file, n_train)
    valid_dataset = get_dataset(valid_file, n_valid)
    return train_dataset, valid_dataset, {}
