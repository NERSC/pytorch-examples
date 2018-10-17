"""
PyTorch dataset specifications.
"""

def get_datasets(name, **data_args):
    if name == 'dummy':
        from .dummy import get_datasets
        return get_datasets(**data_args)
    elif name == 'hpa':
        from .hpa import get_datasets
        return get_datasets(**data_args)
    else:
        raise Exception('Dataset %s unknown' % name)
