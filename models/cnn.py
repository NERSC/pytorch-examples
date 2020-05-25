"""
This module defines a generic CNN model.
"""

# Externals
import torch.nn as nn

class CNNClassifier(nn.Module):
    """
    Generic CNN model with convolutions, max-pooling, fully connected layers,
    and a multi-class linear output (logits) layer.
    """
    def __init__(self, input_shape, output_size, conv_sizes, fc_sizes, dropout=0):
        """Model constructor"""
        super(CNNClassifier, self).__init__()

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
        fc_layers = []
        in_height = input_shape[1] // (2 ** len(conv_sizes))
        in_width = input_shape[2] // (2 ** len(conv_sizes))
        in_size = in_height * in_width * in_size
        for fc_size in fc_sizes:
            fc_layers.append(nn.Linear(in_size, fc_size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            in_size = fc_size
        fc_layers.append(nn.Linear(in_size, output_size))
        self.fc_net = nn.Sequential(*fc_layers)

    def forward(self, x):
        h = self.conv_net(x)
        h = h.view(h.size(0), -1)
        return self.fc_net(h)

def build_model(**kwargs):
    return CNNClassifier(**kwargs)
