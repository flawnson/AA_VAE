import numpy as np

from utils.logger import log


class StepOptim(object):
    """
        A simple wrapper class for learning rate scheduling
    """

    def __init__(self, optimizer, lr=1e-5, step_size=4000):
        self.optimizer = optimizer
        self.lr = lr
        self.step_size = step_size
        self.n_current_steps = 0

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()
        self.update_learning_rate()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def update_learning_rate(self):
        """ Learning rate scheduling per step """

        self.n_current_steps += 1
        if self.n_current_steps % self.step_size == 0:
            new_lr = self.lr * np.power((self.n_current_steps / self.step_size), -0.5)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            log.debug("Current learning rate is {}".format(new_lr))


class ScheduledOptim(object):
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr=1e-5, n_warmup_steps=4000):
        self.optimizer = optimizer
        self.lr = lr
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0

    def step(self):
        "Step by the inner optimizer"
        self.optimizer.step()
        self.update_learning_rate()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        new_lr = self.lr * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        if self.n_current_steps % 1000 == 0:
            log.debug("Current learning rate is {}".format(new_lr))
