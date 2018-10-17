"""
Dataset specification for the human protein atlas image classification
challenge on Kaggle:
https://www.kaggle.com/c/human-protein-atlas-image-classification

I'm currently using PIL to load the images, but should consider using
PyTorch's accimage
"""

# System
import os

# Externals
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image

def _load_img(path):
    with open(path, 'rb') as f:
        return Image.open(f).convert()

class HumanProteinAtlas(torch.utils.data.Dataset):
    """
    Dataset specification for the human protein atlas image classification
    Kaggle challenge.
    """

    def __init__(self, image_folder, truth_file):
        self.image_folder = image_folder
        # Load the truth file
        self.truth = np.loadtxt(truth_file, dtype='U', delimiter=',', skiprows=1)
        self.transform = ToTensor()

    def __getitem__(self, index):
        ident = self.truth[index, 0]
        labels = list(map(int, self.truth[index, 1].split()))

        # Prepare the image
        img_file = os.path.join(self.image_folder, ident + '_green.png')
        x = self.transform(_load_img(img_file))

        # Prepare the target
        y = torch.zeros(28)
        y[labels] = 1

        return x, y

    def __len__(self):
        return len(self.truth)

def get_datasets(image_folder, truth_file, n_train, n_valid):
    dataset = HumanProteinAtlas(image_folder, truth_file)
    n_leftover = len(dataset) - n_train - n_valid
    assert n_leftover >= 0
    train_dataset, valid_dataset, _ = torch.utils.data.random_split(
        dataset, [n_train, n_valid, n_leftover])
    return train_dataset, valid_dataset
