"""
Python module for holding our PyTorch models.
"""

def get_model(name, **model_args):
    """
    Top-level factory function for getting your models.
    """
    if name == 'resnet50_cifar10':
        from .resnet_cifar10 import ResNet50
        return ResNet50(**model_args)
    elif name == 'cnn_classifier':
        from .cnn_classifier import CNNClassifier
        return CNNClassifier(**model_args)
    else:
        raise Exception('Model %s unknown' % name)
