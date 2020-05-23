"""
This module defines a generic trainer for simple models and datasets.
"""

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

    def build(self, model_config, optimizer_config):
        """Instantiate our model, optimizer, loss function"""

        loss_function = model_config.pop('loss_function')

        # Construct the model
        self.model = get_model(**model_config).to(self.device)
        if self.distributed:
            device_ids = [self.gpu] if self.gpu is not None else None
            self.model = DistributedDataParallel(self.model, device_ids=device_ids)

        # Construct the optimizer
        Optim = getattr(torch.optim, optimizer_config.pop('name'))
        self.optimizer = Optim(self.model.parameters(), **optimizer_config)

        # Construct the loss function
        self.loss_func = getattr(torch.nn.functional, loss_function)
    
    def train_epoch(self, data_loader):
        """Train for one epoch"""

        self.model.train()
        sum_loss = 0

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
            self.logger.debug('batch %i loss %.3f', i, batch_loss.item())

        train_loss = sum_loss / (i + 1)
        self.logger.debug('Processed %i batches' % (i + 1))
        self.logger.info('Training loss: %.3f' % train_loss)

        # Return summary
        return dict(train_loss=train_loss)

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""

        self.model.eval()
        sum_loss = 0
        sum_correct = 0

        # Loop over batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            batch_output = self.model(batch_input)
            batch_loss = self.loss_func(batch_output, batch_target).item()
            sum_loss += batch_loss
            # Count number of correct predictions
            _, batch_preds = torch.max(batch_output, 1)
            n_correct = (batch_preds == batch_target).sum().item()
            sum_correct += n_correct
            self.logger.debug('batch %i loss %.3f correct %i', i, batch_loss, n_correct)

        valid_loss = sum_loss / (i + 1)
        valid_acc = sum_correct / len(data_loader.sampler)
        self.logger.debug('Processed %i samples in %i batches',
                          len(data_loader.sampler), i + 1)
        self.logger.info('Validation loss: %.3f acc: %.3f',
                         valid_loss, valid_acc)

        # Return summary
        return dict(valid_loss=valid_loss, valid_acc=valid_acc)

def _test():
    t = BasicTrainer(output_dir='./')
    t.build_model()
