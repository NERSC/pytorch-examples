"""
This module contains trainer functionality for basic GAN-like models.
"""

# System
import os
import time

# Locals
from .base_trainer import BaseTrainer
from models import get_model

# Externals
import numpy as np
import torch

class GANTrainer(BaseTrainer):
    """
    Trainer class for GAN-like models.
    This currently implements the basic GAN training approach.
    I.e., not Wasserstein GAN or any other improved GAN approach.
    """

    def __init__(self, **kwargs):
        super(GANTrainer, self).__init__(**kwargs)

    def print_model_summary(self):
        """Override as needed"""
        self.logger.info('Generator module: \n%s\nParameters: %i',
                         self.generator,
                         sum(p.numel() for p in self.generator.parameters()))
        self.logger.info('Discriminator module: \n%s\nParameters: %i',
                         self.discriminator,
                         sum(p.numel() for p in self.discriminator.parameters()))

    def write_checkpoint(self, checkpoint_id):
        """Write a checkpoint for the model"""
        assert self.output_dir is not None
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        checkpoint_file = 'model_checkpoint_%03i.pth.tar' % checkpoint_id
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(dict(generator=self.generator.state_dict(),
                        discriminator=self.discriminator.state_dict()),
                   os.path.join(checkpoint_dir, checkpoint_file))

    def build_model(self, name='dcgan', noise_dim=64,
                    optimizer='Adam', learning_rate=0.0002,
                    beta1=0.5, beta2=0.999,
                    label_flip_rate=0, **model_args):
        """Construct the GAN"""
        # TODO: add distributed support
        g, d = get_model(name=name, noise_dim=noise_dim, **model_args)
        self.generator, self.discriminator = g.to(self.device), d.to(self.device)
        self.noise_dim = noise_dim
        self.label_flip_rate = label_flip_rate
        self.loss_func = torch.nn.BCELoss()
        opt_type = dict(Adam=torch.optim.Adam)[optimizer]
        self.gen_optimizer = opt_type(self.generator.parameters(), lr=learning_rate)
        self.dis_optimizer = opt_type(self.discriminator.parameters(), lr=learning_rate)

    def train_epoch(self, data_loader):
        """Train the GAN for one epoch"""
        self.generator.train()
        self.discriminator.train()
        summary = dict()
        d_sum_loss, g_sum_loss = 0, 0
        d_sum_out_real, d_sum_out_fake = 0, 0
        start_time = time.time()

        # Loop over training batches
        for i, data in enumerate(data_loader):

            self.logger.debug(' Batch %i', i)
            real_data = data.to(self.device)
            batch_size = real_data.size(0)

            # Label flipping
            flip = (np.random.random_sample() < self.label_flip_rate)
            real_label = 0 if flip else 1
            fake_label = 1 if flip else 0

            # Train discriminator with real samples
            labels = torch.full((batch_size,), real_label, device=self.device)
            self.discriminator.zero_grad()
            d_out_real = self.discriminator(real_data)
            d_loss_real = self.loss_func(d_out_real, labels)
            d_loss_real.backward()

            # Train discriminator with fake generated samples
            noise = torch.randn(batch_size, self.noise_dim, 1, 1, device=self.device)
            fake_data = self.generator(noise)
            labels.fill_(fake_label)
            d_out_fake = self.discriminator(fake_data.detach())
            d_loss_fake = self.loss_func(d_out_fake, labels)
            d_loss_fake.backward()

            # Update discriminator parameters
            d_loss = (d_loss_real + d_loss_fake) / 2
            self.dis_optimizer.step()

            # Train generator to fool discriminator
            self.generator.zero_grad()
            labels.fill_(real_label)
            g_out_fake = self.discriminator(fake_data)
            # We use 'real' labels for the generator loss
            g_loss = self.loss_func(g_out_fake, labels)
            g_loss.backward()

            # Update generator parameters
            self.gen_optimizer.step()

            # Accumulate summaries
            d_sum_loss += d_loss.item()
            g_sum_loss += g_loss.item()
            d_sum_out_real += d_out_real.mean().item()
            d_sum_out_fake += d_out_fake.mean().item()

        # TODO: save subset of generated data
        n_batches = i + 1
        summary['train_time'] = time.time() - start_time
        summary['d_train_loss'] = d_sum_loss / n_batches
        summary['g_train_loss'] = g_sum_loss / n_batches
        summary['d_train_out_real'] = d_sum_out_real / n_batches
        summary['d_train_out_fake'] = d_sum_out_fake / n_batches

        # Print some information
        self.logger.debug(' Processed %i batches', n_batches)
        self.logger.info('  Avg discriminator real output: %.4f', summary['d_train_out_real'])
        self.logger.info('  Avg discriminator fake output: %.4f', summary['d_train_out_fake'])
        self.logger.info('  Avg discriminator loss: %.4f', summary['d_train_loss'])
        self.logger.info('  Avg generator loss: %.4f', summary['g_train_loss'])
        return summary
