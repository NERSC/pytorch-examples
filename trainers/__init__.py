"""
Python module for holding our PyTorch trainers.

Trainers here inherit from the BaseTrainer and implement the logic for
constructing the model as well as training and evaluation.
"""

def get_trainer(name, **trainer_args):
    """
    Factory function for retrieving a trainer.
    """
    if name == 'hello':
        from .hello import HelloTrainer
        return HelloTrainer(**trainer_args)
    elif name == 'basic':
        from .basic import BasicTrainer
        return BasicTrainer(**trainer_args)
    elif name == 'cifar10':
        from .cifar10 import Cifar10Trainer
        return Cifar10Trainer(**trainer_args)
    elif name == 'hepcnn':
        from .hepcnn import HEPCNNTrainer
        return HEPCNNTrainer(**trainer_args)
    elif name == 'rpvgan':
        from .gan_trainer import GANTrainer
        return GANTrainer(**trainer_args)
    else:
        raise Exception('Trainer %s unknown' % name)
