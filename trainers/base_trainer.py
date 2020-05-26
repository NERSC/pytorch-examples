"""
Common PyTorch trainer code.
"""

# System
import os
import time
import logging

# Externals
import numpy as np
import pandas as pd
import torch

def _format_summary(summary):
    """Make a formatted string for logging summary info"""
    return ' '.join(f'{k} {v:.4g}' for (k, v) in summary.items())

class BaseTrainer(object):
    """
    Base class for PyTorch trainers.
    This implements the common training logic,
    logging of summaries, and checkpoints.
    """

    def __init__(self, output_dir=None, gpu=None,
                 distributed=False, rank=0):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = (os.path.expandvars(output_dir)
                           if output_dir is not None else None)
        self.gpu = gpu
        if gpu is not None:
            self.device = 'cuda:%i' % gpu
            torch.cuda.set_device(gpu)
        else:
            self.device = 'cpu'
        self.distributed = distributed
        self.rank = rank
        self.summaries = None

    def _get_summary_file(self):
        return os.path.join(self.output_dir, 'summaries_%i.csv' % self.rank)

    def save_summary(self, summary, write_file=True):
        """Save new summary information"""

        # First summary
        if self.summaries is None:
            self.summaries = pd.DataFrame([summary])

        # Append a new summary row
        else:
            self.summaries = self.summaries.append([summary], ignore_index=True)

        # Write current summaries to file (note: overwrites each time)
        if write_file and self.output_dir is not None:
            self.summaries.to_csv(self._get_summary_file(), index=False,
                                  float_format='%.6f', sep='\t')

    def load_summaries(self):
        self.summaries = pd.read_csv(self._get_summary_file(), delim_whitespace=True)

    # TODO: move to derived type and utils
    def write_checkpoint(self, checkpoint_id):
        """Write a checkpoint for the model"""
        assert self.output_dir is not None
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        checkpoint_file = 'model_checkpoint_%03i.pth.tar' % checkpoint_id
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(dict(model=self.model.state_dict()),
                   os.path.join(checkpoint_dir, checkpoint_file))

    def build(self, model_config, loss_config, optimizer_config, metrics_config):
        """Virtual method to build model, optimizer, etc."""
        raise NotImplementedError

    def train_epoch(self, data_loader):
        """Virtual method to train a model"""
        raise NotImplementedError

    def evaluate(self, data_loader):
        """Virtual method to evaluate a model"""
        raise NotImplementedError

    def train(self, train_data_loader, n_epochs, valid_data_loader=None):
        """Run the model training"""

        # Loop over epochs
        for i in range(n_epochs):
            self.logger.info('Epoch %i' % i)
            summary = dict(epoch=i)

            # Train on this epoch
            start_time = time.time()
            summary.update(self.train_epoch(train_data_loader))
            summary['train_time'] = time.time() - start_time

            # Evaluate on this epoch
            if valid_data_loader is not None:
                start_time = time.time()
                summary.update(self.evaluate(valid_data_loader))
                summary['valid_time'] = time.time() - start_time

            # Save summary, checkpoint
            self.logger.info('Summary: %s', _format_summary(summary))
            self.save_summary(summary)
            if self.output_dir is not None and self.rank==0:
                self.write_checkpoint(checkpoint_id=i)

        return self.summaries
