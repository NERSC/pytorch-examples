"""
This module defines the trainer for the CIFAR10 classification problem.
"""

# System
import time

# Externals
import torch

# Locals
from .base_trainer import BaseTrainer
from models import get_model

class Cifar10Trainer(BaseTrainer):
    """Trainer code for the CIFAR10 classification problem."""

    def __init__(self, distributed=False, **kwargs):
        super(Cifar10Trainer, self).__init__(**kwargs)
        self.distributed = distributed

    def build_model(self, model_type='resnet50_cifar10',
                    optimizer='Adam', learning_rate=0.001):
        """
        Instantiate our model.
        Just supporting resnet50 directly for this first pass.
        """
        model = get_model(model_type)
        if self.distributed:
            model = nn.parallel.DistributedDataParallelCPU(model)
        self.model = model.to(self.device)
        opt_type = dict(Adam=torch.optim.Adam)[optimizer]
        self.optimizer = opt_type(self.model.parameters(), lr=learning_rate)
        self.loss_func = torch.nn.CrossEntropyLoss()
    
    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()
        summary = dict()
        sum_loss = 0
        start_time = time.time()
        # Loop over training batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            self.model.zero_grad()
            batch_output = self.model(batch_input)
            batch_loss = self.loss_func(batch_output, batch_target)
            batch_loss.backward()
            self.optimizer.step()
            sum_loss += batch_loss.item()
        self.logger.debug('  Processed %i batches' % (i + 1))
        summary['train_time'] = time.time() - start_time
        summary['train_loss'] = sum_loss / (i + 1)
        self.logger.info('  Training loss: %.3f' % summary['train_loss'])
        return summary

    def evaluate(self, data_loader):
        """"Evaluate the model"""
        pass

def _test():
    t = Cifar10Trainer(output_dir='./')
    t.build_model()
