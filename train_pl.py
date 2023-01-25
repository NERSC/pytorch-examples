"""
Main training script for NERSC PyTorch lightning examples
"""

# System
import os
import argparse
import logging

# Externals
import yaml
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import DeviceStatsMonitor

# Locals
from datasets import get_data_loaders
from modules import get_module
from utils.logging import config_logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/mnist_pl.yaml',
            help='YAML configuration file')
    return parser.parse_args()

def load_config(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['output_dir'] = os.path.expandvars(config['output_dir'])
    return config

def main():
    """Main training script function"""

    # Initialization
    args = parse_args()

    # Load configuration
    config = load_config(args)

    # Setup logging
    config_logging(verbose=False)

    # Load the datasets
    train_data_loader, valid_data_loader = get_data_loaders(**config['data'])

    # Load the PL module
    module = get_module(config['module'], config)

    # Prepare callbacks
    callbacks = [
        DeviceStatsMonitor(),
    ]

    # Create the trainer
    pl_logger = pl.loggers.CSVLogger(config['output_dir'], name=config['name'])
    num_nodes = os.environ['SLURM_JOB_NUM_NODES']
    trainer = pl.Trainer(gpus=-1, num_nodes=num_nodes,
                         strategy=DDPStrategy(find_unused_parameters=False),
                         logger=pl_logger,
                         enable_progress_bar=False,
                         callbacks=callbacks,
                         **config['trainer'])
    trainer.fit(module, train_data_loader, valid_data_loader)

    logging.info('All done!')

if __name__ == '__main__':
    main()
