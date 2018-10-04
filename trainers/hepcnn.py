"""
Trainer code for the HEP-CNN RPV classifier.
"""

# System
import time

# Externals
import torch
import torch.nn as nn

# Locals
from .base_trainer import BaseTrainer
from models.hepcnn import HEPCNNClassifier

class HEPCNNTrainer(BaseTrainer):
    """Trainer code for the HEP-CNN classifier."""

    def __init__(self, distributed=False, **kwargs):
        super(HEPCNNTrainer, self).__init__(**kwargs)
        self.distributed = distributed

    def build_model(self, input_shape, conv_sizes, dense_sizes, dropout,
                    optimizer='Adam', learning_rate=0.001):
        """Instantiate our model"""
        model = HEPCNNClassifier(input_shape=input_shape,
                                 conv_sizes=conv_sizes,
                                 dense_sizes=dense_sizes,
                                 dropout=dropout)
        if self.distributed:
            model = nn.parallel.DistributedDataParallelCPU(model)
        self.model = model.to(self.device)
        opt_type = dict(Adam=torch.optim.Adam)[optimizer]
        self.optimizer = opt_type(self.model.parameters(), lr=learning_rate)
        self.loss_func = torch.nn.BCELoss()

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

    @torch.no_grad()
    def evaluate(self, data_loader):
        """Evaluate the model"""
        self.model.eval()
        summary = dict()
        sum_loss = 0
        sum_correct = 0
        start_time = time.time()
        # Loop over batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            batch_output = self.model(batch_input)
            sum_loss += self.loss_func(batch_output, batch_target)
            # Count number of correct predictions
            preds, labels = batch_output > 0.5, batch_target > 0.5
            sum_correct += preds.eq(labels).sum().item()
        summary['valid_time'] = time.time() - start_time
        summary['valid_loss'] = sum_loss / (i + 1)
        summary['valid_acc'] = sum_correct / len(data_loader.sampler)
        self.logger.info('  Validation loss: %.3f acc: %.3f' %
                         (summary['valid_loss'], summary['valid_acc']))
        return summary
