"""Classes and utilities for tracking metrics during training"""

import abc

class Metric(abc.ABCMeta):
    """Abstract base class for metrics that can be computed in batches.

    The interface is inspired by the Keras Metric class.
      - New data is added via the update method
      - Results returned by the result method
      - State should be reset with the reset method
    """

    def init(self):
        self.reset()

    @abc.abstractmethod
    def update(self, outputs, targets):
        """Update results with new batch data"""
        raise NotImplementedError('Must be implemented in subclass')

    @abc.abstractmethod
    def result(self):
        """Return metric result"""
        raise NotImplementedError('Must be implemented in subclass')

    @abc.abstractmethod
    def reset(self):
        """Reset metric"""
        raise NotImplementedError('Must be implemented in subclass')

class Accuracy(Metric):
    """Classification accuracy

    Targets are assumed to be integer labels.
    Assumes last axis of predictions is the class vector.
    I.e., won't work for binary labels.
    """

    def reset(self):
        self.n_correct = self.n_total = 0

    def update(self, outputs, targets):
        _, preds = torch.max(outputs, dim=-1)
        self.n_total += targets.numel()
        self.n_correct += (preds == targets).sum().item()

    def result(self):
        return self.n_correct / self.n_total
