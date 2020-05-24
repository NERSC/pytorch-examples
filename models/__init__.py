"""
Python module for holding our PyTorch models.
"""

import importlib
import torchvision

def get_model(name, **model_args):
    """Top-level factory function for getting your models"""

    # First look in our local modules using importlib
    try:
        module = importlib.import_module('.' + name, 'models')
        return module.build_model(**model_args)

    # If the import fails, try getting it from torchvision
    except ImportError:
        ModelType = getattr(torchvision.models, name)
        return ModelType(**model_args)
