"""
This module defines a generic trainer for simple models and datasets.
"""

# Externals
import torch
from torch import nn

# Locals
from .base_trainer import BaseTrainer
from models import get_model
import utils.metrics

class BasicTrainer(BaseTrainer):
    """Trainer code for basic classification problems."""

    def __init__(self, **kwargs):
        super(BasicTrainer, self).__init__(**kwargs)

    def build(self, model_config, loss_config, optimizer_config, metrics_config):
        """Instantiate our model, optimizer, loss function"""

        # Construct the model
        self.model = get_model(**model_config).to(self.device)
        if self.distributed:
            device_ids = [self.gpu] if self.gpu is not None else None
            self.model = DistributedDataParallel(self.model, device_ids=device_ids)

        # Construct the optimizer
        Optim = getattr(torch.optim, optimizer_config.pop('name'))
        self.optimizer = Optim(self.model.parameters(), **optimizer_config)

        # Construct the loss function
        Loss = getattr(torch.nn, loss_config.pop('name'))
        self.loss_func = Loss(**loss_config)

        # Construct the metrics
        self.metrics = utils.metrics.get_metrics(metrics_config)

        # Print a model summary
        if self.rank == 0:
            self.logger.info(self.model)
            self.logger.info('Number of parameters: %i',
                             sum(p.numel() for p in self.model.parameters()))
    
    def train_epoch(self, data_loader):
        """Train for one epoch"""

        self.model.train()

        # Reset metrics
        sum_loss = 0
        utils.metrics.reset_metrics(self.metrics)

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
            utils.metrics.update_metrics(self.metrics, batch_output, batch_target)
            self.logger.debug('batch %i loss %.3f', i, batch_loss.item())

        train_loss = sum_loss / (i + 1)
        metrics_summary = utils.metrics.get_results(self.metrics, 'train_')
        self.logger.debug('Processed %i batches' % (i + 1))

        # Return summary
        return dict(train_loss=train_loss, **metrics_summary)

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""

        self.model.eval()

        # Reset metrics
        sum_loss = 0
        utils.metrics.reset_metrics(self.metrics)

        # Loop over batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            batch_output = self.model(batch_input)
            batch_loss = self.loss_func(batch_output, batch_target).item()
            sum_loss += batch_loss
            utils.metrics.update_metrics(self.metrics, batch_output, batch_target)
            self.logger.debug('batch %i loss %.3f', i, batch_loss)

        # Summarize validation metrics
        metrics_summary = utils.metrics.get_results(self.metrics, 'valid_')

        valid_loss = sum_loss / (i + 1)
        self.logger.debug('Processed %i samples in %i batches',
                          len(data_loader.sampler), i + 1)

        # Return summary
        return dict(valid_loss=valid_loss, **metrics_summary)

def get_trainer(**kwargs):
    return BasicTrainer(**kwargs)

def _test():
    t = BasicTrainer(output_dir='./')
    t.build_model()
