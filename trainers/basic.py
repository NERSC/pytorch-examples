"""
This module defines a generic trainer for simple models and datasets.
"""

# System
import time

# Externals
import torch
from torch import nn

# Locals
from .base_trainer import BaseTrainer
from models import get_model

class BasicTrainer(BaseTrainer):
    """Trainer code for basic classification problems."""

    def __init__(self, **kwargs):
        super(BasicTrainer, self).__init__(**kwargs)

    def build_model(self, model_type='cnn_classifier',
                    optimizer='Adam', learning_rate=0.001,
                    **model_args):
        """Instantiate our model"""
        self.model = get_model(name=model_type, **model_args).to(self.device)
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallelCPU(self.model)
        # TODO: add support for more optimizers and loss functions here
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
            self.logger.debug('  batch %i', i)
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            self.model.zero_grad()
            batch_output = self.model(batch_input)
            batch_loss = self.loss_func(batch_output, batch_target)
            batch_loss.backward()
            self.optimizer.step()
            sum_loss += batch_loss.item()
        summary['train_time'] = time.time() - start_time
        summary['train_loss'] = sum_loss / (i + 1)
        self.logger.debug(' Processed %i batches' % (i + 1))
        self.logger.info('  Training loss: %.3f' % summary['train_loss'])
        return summary

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""
        self.model.eval()
        summary = dict()
        sum_loss = 0
        sum_correct = 0
        start_time = time.time()
        # Loop over batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            self.logger.debug(' batch %i', i)
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            batch_output = self.model(batch_input)
            sum_loss += self.loss_func(batch_output, batch_target).item()
            # Count number of correct predictions
            _, batch_preds = torch.max(batch_output, 1)
            sum_correct += (batch_preds == batch_target).sum().item()
        summary['valid_time'] = time.time() - start_time
        summary['valid_loss'] = sum_loss / (i + 1)
        summary['valid_acc'] = sum_correct / len(data_loader.sampler)
        self.logger.debug(' Processed %i samples in %i batches',
                          len(data_loader.sampler), i + 1)
        self.logger.info('  Validation loss: %.3f acc: %.3f' %
                         (summary['valid_loss'], summary['valid_acc']))
        return summary

def _test():
    t = BasicTrainer(output_dir='./')
    t.build_model()
