"""
Python module for holding our PyTorch Lightning modules.

Trainers here inherit from the BaseTrainer and implement the logic for
constructing the model as well as training and evaluation.
"""

import importlib

def get_module(name, config):
    """Factory function for constructing a Trainer"""
    module = importlib.import_module('.' + name, 'modules')
    return module.get_module(config)
