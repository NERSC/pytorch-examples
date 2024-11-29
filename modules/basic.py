# System
import logging

# Externals
import torch
import pytorch_lightning as pl

# Locals
from models import get_model


class BasicModule(pl.LightningModule):
    """PL Module for basic single-model examples"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Construct the model
        self.model = get_model(**config['model'])

        # Construct the loss function
        loss_config = config['loss']
        Loss = getattr(torch.nn, loss_config.pop('name'))
        self.loss_func = Loss(**loss_config)

    def configure_optimizers(self):
        logging.info('configure_optimizers')
        optimizer_config = self.config['optimizer']
        Optim = getattr(torch.optim, optimizer_config.pop('name'))
        return Optim(self.model.parameters(), **optimizer_config)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        batch_input, batch_target = batch
        batch_output = self.model(batch_input)
        batch_loss = self.loss_func(batch_output, batch_target)
        self.log('train_loss', batch_loss)
        return batch_loss

    def validation_step(self, batch, batch_idx):
        batch_input, batch_target = batch
        batch_output = self.model(batch_input)
        batch_loss = self.loss_func(batch_output, batch_target)
        self.log("valid_loss", batch_loss)

def get_module(config):
    return BasicModule(config)
