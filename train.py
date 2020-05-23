"""
Main training script for NERSC PyTorch examples
"""

# System
import os
import argparse
import logging

# Externals
import yaml
import numpy as np

# Locals
from datasets import get_data_loaders
from trainers import get_trainer
from utils.logging import config_logging
from utils.distributed import init_workers

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/hello.yaml',
            help='YAML configuration file')
    add_arg('-d', '--distributed-backend', choices=['mpi', 'nccl', 'gloo'],
            help='Specify the distributed backend to use')
    add_arg('-v', '--verbose', action='store_true',
            help='Enable verbose logging')
    add_arg('--gpu', type=int,
            help='Choose a specific GPU by ID')
    add_arg('--ranks-per-node', type=int, default=8,
            help='Specifying number of ranks per node')
    add_arg('--rank-gpu', action='store_true',
            help='Choose GPU according to local rank')
    return parser.parse_args()

def load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    """Main function"""

    # Initialization
    args = parse_args()
    rank, n_ranks = init_workers(args.distributed_backend)

    # Load configuration
    config = load_config(args.config)
    data_config = config['data_config']
    model_config = config.get('model_config', {})
    train_config = config['train_config']

    # Prepare output directory
    output_dir = config.get('output_dir', None)
    if output_dir is not None:
        output_dir = os.path.expandvars(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_file = (os.path.join(output_dir, 'out_%i.log' % rank)
                if output_dir is not None else None)
    config_logging(verbose=args.verbose, log_file=log_file)
    logging.info('Initialized rank %i out of %i', rank, n_ranks)
    if rank == 0:
        logging.info('Configuration: %s' % config)

    # Load the datasets
    distributed = args.distributed_backend is not None
    train_data_loader, valid_data_loader = get_data_loaders(
        distributed=distributed, **data_config)

    # Load the trainer
    gpu = (rank % args.ranks_per_node) if args.rank_gpu else args.gpu
    if gpu is not None:
        logging.info('Using GPU %i', gpu)
    trainer = get_trainer(name=config['trainer'], distributed=distributed,
                          rank=rank, output_dir=output_dir, gpu=gpu)
    # Build the model
    trainer.build(**model_config)
    if rank == 0:
        trainer.print_model_summary()

    # Run the training
    summary = trainer.train(train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            **train_config)
    if output_dir is not None:
        trainer.write_summaries()

    # Print some conclusions
    n_train_samples = len(train_data_loader.sampler)
    logging.info('Finished training')
    train_time = np.mean(summary['train_time'])
    logging.info('Train samples %g time %g s rate %g samples/s',
                 n_train_samples, train_time, n_train_samples / train_time)
    if valid_data_loader is not None:
        n_valid_samples = len(valid_data_loader.sampler)
        valid_time = np.mean(summary['valid_time'])
        logging.info('Valid samples %g time %g s rate %g samples/s',
                     n_valid_samples, valid_time, n_valid_samples / valid_time)

    logging.info('All done!')

if __name__ == '__main__':
    main()
