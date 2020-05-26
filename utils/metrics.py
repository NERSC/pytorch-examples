"""Classes and utilities for tracking metrics during training"""

import torch

class Metric():
    """Abstract base class for metrics that can be computed in batches.

    The interface is inspired by the Keras Metric class.
      - New data is added via the update method
      - Results returned by the result method
      - State should be reset with the reset method
    """

    def update(self, outputs, targets):
        """Update results with new batch data"""
        raise NotImplementedError('Must be implemented in subclass')

    def result(self):
        """Return metric result"""
        raise NotImplementedError('Must be implemented in subclass')

    def reset(self):
        """Reset metric"""
        raise NotImplementedError('Must be implemented in subclass')

class Accuracy(Metric):
    """Classification accuracy

    Targets are assumed to be integer labels.
    Assumes last axis of predictions is the class vector.
    I.e., won't work for binary labels.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.n_correct = self.n_total = 0

    def update(self, outputs, targets):
        _, preds = torch.max(outputs, dim=-1)
        self.n_total += targets.numel()
        self.n_correct += (preds == targets).sum().item()

    def result(self):
        return self.n_correct / self.n_total

def get_metrics(metrics_config):
    """Get a dictionary of requested metrics instances"""
    return dict((key, globals()[m]())
                for (key, m) in metrics_config.items())

def reset_metrics(metrics):
    """Reset all metrics in the metrics dict"""
    for metric in metrics.values():
        metric.reset()

def update_metrics(metrics, outputs, targets):
    for metric in metrics.values():
        metric.update(outputs, targets)

def get_results(metrics, prefix=''):
    return dict((prefix + key, metric.result())
                for (key, metric) in metrics.items())
