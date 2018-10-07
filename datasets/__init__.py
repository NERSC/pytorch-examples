"""
PyTorch dataset specifications.
"""

def get_datasets(name, **data_args):
    if name == 'dummy':
        from .dummy import get_datasets
        return get_datasets(**data_args)
    elif name == 'cifar10':
        from .cifar10 import get_datasets
        return get_datasets(**data_args)
    elif name == 'hep_images':
        from .hep_images import get_datasets
        return get_datasets(**data_args)
    elif name == 'rpv_images':
        from .rpv_images import get_datasets
        return get_datasets(**data_args)
    else:
        raise Exception('Dataset %s unknown' % name)
