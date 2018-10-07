"""
This module defines the dataset for the RPV HEP-GAN.
Note the data is pretty much the same as the HEP-CNN classifier,
but for historical reasons for now this stored in different files
and needs special handling.

Maybe eventually we can unify things.
"""

# External imports
import torch
from torch.utils.data import Dataset

def load_file(file_path, n_samples=None, dtype=np.float32):
    with np.load(file_path) as f:
        data = f['hist']
        if n_samples is not None and n_samples > 0:
            data = data[:n_samples]
        return torch.from_numpy(data[:, None].astype(dtype))

class RPVImages(Dataset):
    """Dataset for RPV image tensors"""

    def __init__(self, input_file, n_samples, scale=None):
        # Load the data
        self.data = load_file(input_file)
        if scale is not None:
            self.data = self.data / scale

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.data.size(0)
