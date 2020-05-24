"""
This module contains PyTorch model code for the HEP-CNN
RPV classifier.
"""

# Externals
import torch.nn as nn

class HEPCNNClassifier(nn.Module):
    """
    HEP-CNN RPV classifier model.
    """

    def __init__(self, input_shape, conv_sizes, dense_sizes, dropout):
        """HEP-CNN classifier constructor"""
        super(HEPCNNClassifier, self).__init__()

        # Add the convolutional layers
        conv_layers = []
        in_size = input_shape[0]
        for conv_size in conv_sizes:
            conv_layers.append(nn.Conv2d(in_size, conv_size,
                                         kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(2))
            in_size = conv_size
        self.conv_net = nn.Sequential(*conv_layers)

        # Add the dense layers
        dense_layers = []
        in_height = input_shape[1] // (2 ** len(conv_sizes))
        in_width = input_shape[2] // (2 ** len(conv_sizes))
        in_size = in_height * in_width * in_size
        for dense_size in dense_sizes:
            dense_layers.append(nn.Linear(in_size, dense_size))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(dropout))
            in_size = dense_size
        dense_layers.append(nn.Linear(in_size, 1))
        dense_layers.append(nn.Sigmoid())
        self.dense_net = nn.Sequential(*dense_layers)

    def forward(self, x):
        h = self.conv_net(x)
        h = h.view(h.size(0), -1)
        return self.dense_net(h).squeeze(-1)

def build_model(**kwargs):
    return HEPCNNClassifier(**kwargs)
