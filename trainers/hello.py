"""
Hello world PyTorch trainer.
"""

# Locals
from .base_trainer import BaseTrainer

class HelloTrainer(BaseTrainer):
    """Hello world trainer object"""

    def __init__(self, **kwargs):
        super(HelloTrainer, self).__init__(**kwargs)

    def build(self, **kwargs):
        self.logger.info('Hello world')

    def write_checkpoint(self, checkpoint_id):
        pass

    def train_epoch(self, data_loader):
        # Loop over training batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            self.logger.info('Train batch input %s, target %s',
                             batch_input.size(), batch_target.size())
        self.logger.info('Processed %i training batches' % (i + 1))
        return dict(train_loss=0)

    def evaluate(self, data_loader):
        """"Evaluate the model"""
        # Loop over validation batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            self.logger.info('Valid batch input %s, target %s',
                             batch_input.size(), batch_target.size())
        self.logger.info('Processed %i validation batches' % (i + 1))
        return dict(valid_loss=0, valid_acc=1)

def get_trainer(**kwargs):
    return HelloTrainer(**kwargs)
