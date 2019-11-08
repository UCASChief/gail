import datetime
import os
from functools import wraps
from collections import namedtuple
import torch
import torch.cuda
import torch.nn as nn
import random
import configparser
import numpy as np

__all__ = ['FloatTensor', 'IntTensor', 'LongTensor', 'Transition', 'add_method', 'to_device', 'ModelArgs',
           'to_FloatTensor', 'to_IntTensor', 'to_LongTensor', 'time_this', '_init_weight', 'Memory', 'device']

IntTensor = torch.IntTensor
LongTensor = torch.LongTensor

Transition = namedtuple('Transition', (
    'discrete_state', 'continuous_state', 'discrete_action', 'continuous_action',
    'next_discrete_state', 'next_continuous_state', 'old_log_prob', 'mask'
))
use_gpu = True
device = torch.device('cuda') if use_gpu else torch.device('cpu')
FloatTensor = torch.FloatTensor if not use_gpu else torch.cuda.FloatTensor


# device = torch.device('cpu')


def add_method(cls, name=None):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(*args, **kwargs)

        if name is None:
            setattr(cls, func.__name__, wrapper)
        else:
            setattr(cls, name, wrapper)
        return func

    return decorator


def to_device(device, *args):
    return [x.to(device) for x in args]


def to_IntTensor(*args):
    return [x.type(IntTensor) for x in args]


def to_FloatTensor(*args):
    return [x.type(FloatTensor) for x in args]


def to_LongTensor(*args):
    return [x.type(LongTensor) for x in args]


def _init_weight(m):
    if type(m) == nn.Linear:
        size = m.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
        m.bias.data.fill_(0.0)


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)

    def clear_memory(self):
        del self.memory[:]


class Singleton(type):
    def __init__(cls, *args, **kwargs):
        cls.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__call__(*args, **kwargs)
            return cls.__instance
        else:
            return cls.__instance


class ModelArgs(metaclass=Singleton):
    def __init__(self):
        args = configparser.ConfigParser()
        args.read(os.path.dirname(__file__) + os.sep + 'config.ini')
        print('read config')
        self.n_discrete_state = args.getint('policy_net', 'n_discrete_state')
        self.n_continuous_state = args.getint('policy_net', 'n_continuous_state')
        self.n_discrete_action = args.getint('policy_net', 'n_discrete_action')
        self.n_continuous_action = args.getint('policy_net', 'n_continuous_action')
        self.n_policy_hidden = args.getint('policy_net', 'n_policy_hidden')
        self.n_transition_hidden = args.getint('transition_net', 'n_transition_hidden')
        self.n_value_hidden = args.getint('value_net', 'n_value_hidden')
        self.n_discriminator_hidden = args.getint('discriminator_net', 'n_discriminator_hidden')

        self.value_lr = args.getfloat('value_net', 'value_net_learning_rate')
        self.policy_lr = args.getfloat('policy_net', 'policy_net_learning_rate')
        self.discrim_lr = args.getfloat('discriminator_net', 'discriminator_net_learning_rate')
        self.ppo_buffer_size = args.getint('ppo', 'buffer_size')
        self.ppo_optim_epoch = args.getint('ppo', 'ppo_optim_epoch')
        self.ppo_mini_batch_size = args.getint('ppo', 'ppo_mini_batch_size')
        self.ppo_clip_epsilon = args.getfloat('ppo', 'ppo_clip_epsilon')
        self.gamma = args.getfloat('ppo', 'gamma')
        self.lam = args.getfloat('ppo', 'lam')
        self.expert_activities_data_path = args.get('data_path', 'expert_activities_data_path')
        self.expert_cost_data_path = args.get('data_path', 'expert_cost_data_path')
        self.expert_batch_size = args.getint('general', 'expert_batch_size')
        self.training_epochs = args.getint('general', 'training_epochs')


def time_this(func):
    @wraps(func)
    def int_time(*args, **kwargs):
        start_time = datetime.datetime.now()
        ret = func(*args, **kwargs)
        over_time = datetime.datetime.now()
        total_time = (over_time - start_time).total_seconds()
        print('Function %s\'s total running time : %s s.' % (func.__name__, total_time))
        return ret

    return int_time
