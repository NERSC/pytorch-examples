"""
Python module for holding our PyTorch trainers.

Trainers here inherit from the BaseTrainer and implement the logic for
constructing the model as well as training and evaluation.
"""

import importlib

def get_trainer(name, **trainer_args):
    """Factory function for constructing a Trainer"""
    module = importlib.import_module('.' + name, 'trainers')
    return module.get_trainer(**trainer_args)
