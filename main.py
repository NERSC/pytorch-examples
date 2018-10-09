"""
Main training script for NERSC PyTorch examples
"""

# System
import argparse
import logging

# Externals
import yaml
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Locals
from datasets import get_datasets
from trainers import get_trainer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('main.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/hello.yaml')
    add_arg('-d', '--distributed', action='store_true')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    return parser.parse_args()

def main():
    """Main function"""

    # Parse the command line
    args = parse_args()

    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)
    logging.info('Initializing')
    if args.show_config:
        logging.info('Command line config: %s' % args)

    # Initialize MPI
    if args.distributed:
        dist.init_process_group(backend='mpi')
        logging.info('MPI rank %i' % dist.get_rank())

    # Load configuration
    with open(args.config) as f:
        config = yaml.load(f)
    if not args.distributed or (dist.get_rank() == 0):
        logging.info('Configuration: %s' % config)
    data_config = config['data_config']
    model_config = config.get('model_config', {})
    train_config = config['train_config']

    # Load the datasets
    train_dataset, valid_dataset = get_datasets(**data_config)
    batch_size = train_config.pop('batch_size')
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,
                                   sampler=train_sampler)
    logging.info('Loaded %g training samples', len(train_dataset))
    if valid_dataset is not None:
        valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size)
        logging.info('Loaded %g validation samples', len(valid_dataset))
    else:
        valid_data_loader = None

    # Load the trainer
    experiment_config = config['experiment_config']
    output_dir = experiment_config.pop('output_dir', None)
    if args.distributed and dist.get_rank() != 0:
        output_dir = None
    trainer = get_trainer(distributed=args.distributed, output_dir=output_dir,
                          **experiment_config)
    # Build the model
    trainer.build_model(**model_config)
    if not args.distributed or (dist.get_rank() == 0):
        trainer.print_model_summary()

    # Run the training
    summary = trainer.train(train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            **train_config)
    if not args.distributed or (dist.get_rank() == 0):
        trainer.write_summaries()

    # Print some conclusions
    n_train_samples = len(train_data_loader.sampler)
    logging.info('Finished training')
    train_time = np.mean(summary['train_time'])
    logging.info('Train samples %g time %gs rate %g samples/s',
                 n_train_samples, train_time, n_train_samples / train_time)
    if valid_data_loader is not None:
        n_valid_samples = len(valid_data_loader.sampler)
        valid_time = np.mean(summary['valid_time'])
        logging.info('Valid samples %g time %g s rate %g samples/s',
                     n_valid_samples, valid_time, n_valid_samples / valid_time)

    # Drop to IPython interactive shell
    if args.interactive:
        logging.info('Starting IPython interactive session')
        import IPython
        IPython.embed()

    logging.info('All done!')

if __name__ == '__main__':
    main()
