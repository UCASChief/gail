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
import torch.nn.functional as F

__all__ = ['FloatTensor', 'IntTensor', 'LongTensor', 'Transition', 'add_method', 'to_device', 'ModelArgs',
           'to_FloatTensor', 'to_IntTensor', 'to_LongTensor', 'time_this', '_init_weight', 'Memory', 'device']

IntTensor = torch.IntTensor
LongTensor = torch.LongTensor

SimpleTransition = namedtuple('SimpleTransition', (
    'state', 'action', 'next_state', 'old_log_prob', 'mask'
))

Transition = namedtuple('Transition', (
    'onehot_state', 'multihot_state', 'continuous_state', 'onehot_action','multihot_action', 'continuous_action',
    'next_onehot_state','next_multihot_state', 'next_continuous_state', 'old_log_prob', 'mask'
))
# use_gpu = False
use_gpu = True
device = torch.device('cuda:1') if use_gpu else torch.device('cpu')
FloatTensor = torch.FloatTensor if not use_gpu else torch.cuda.FloatTensor

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

class SimpleMemory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(SimpleTransition(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return SimpleTransition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return SimpleTransition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)

    def clear_memory(self):
        del self.memory[:]


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
        args.read(os.path.dirname(__file__) + os.sep + 'config/config_scaled_new_v2.ini')
        print('read config')
        self.n_state = args.getint('policy_net', 'n_state')
        self.n_action = args.getint('policy_net', 'n_action')
        self.n_onehot_state = args.getint('policy_net', 'n_onehot_state')
        self.n_multihot_state = args.getint('policy_net', 'n_multihot_state')
        self.n_continuous_state = args.getint('policy_net', 'n_continuous_state')
        self.n_onehot_action = args.getint('policy_net', 'n_onehot_action')
        self.n_multihot_action = args.getint('policy_net', 'n_multihot_action')
        self.n_continuous_action = args.getint('policy_net', 'n_continuous_action')
        self.n_policy_hidden = args.getint('policy_net', 'n_policy_hidden')
        self.n_transition_hidden = args.getint('transition_net', 'n_transition_hidden')
        self.n_value_hidden = args.getint('value_net', 'n_value_hidden')
        self.n_discriminator_hidden = args.getint('discriminator_net', 'n_discriminator_hidden')

        self.value_lr = args.getfloat('value_net', 'value_net_learning_rate')
        self.policy_lr = args.getfloat('policy_net', 'policy_net_learning_rate')
        self.discrim_lr = args.getfloat('discriminator_net', 'discriminator_net_learning_rate')
        self.sample_batch_size = args.getint('ppo', 'sample_batch_size')
        self.sample_traj_length = args.getint('ppo', 'sample_traj_length')
        self.ppo_optim_epoch = args.getint('ppo', 'ppo_optim_epoch')
        self.ppo_mini_batch_size = args.getint('ppo', 'ppo_mini_batch_size')
        self.ppo_clip_epsilon = args.getfloat('ppo', 'ppo_clip_epsilon')
        self.gamma = args.getfloat('ppo', 'gamma')
        self.lam = args.getfloat('ppo', 'lam')
        self.data_set_path = args.get('data_path', 'data_set_path')
        self.expert_batch_size = args.getint('general', 'expert_batch_size')
        self.training_epochs = args.getint('general', 'training_epochs')
        self.drop_prob = args.getfloat('general', 'drop_prob')
        self.noise_std = args.getfloat('general', 'noise_std')
        self.concat_action = args.getint('general', 'concat_action')

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

def one_hot_decode(one_hot_data):
    tmp = (one_hot_data == 1).nonzero()
    one_hot_decode_data = tmp[:, tmp.shape[1]-1].unsqueeze(1)
    return one_hot_decode_data

if __name__ == '__main__':
    import io, os
    features, labels = [], []
    with io.open(os.path.dirname(__file__) + '/dataset.txt', 'r') as file:
        for line in file:
            features_l, labels_l, clicks_l = line.split('\t')
            features.append([float(x) for x in features_l.split(',')])
            labels.append([float(x) for x in labels_l.split(',')])
    features_tensor = FloatTensor(features)
    first = features_tensor[..., :88]
    attr_1 = features_tensor[..., 88].to(torch.int64)
    attr_2 = features_tensor[..., 89].to(torch.int64)
    attr_3 = features_tensor[..., 90].to(torch.int64)
    a1 = F.one_hot(attr_1, 30)
    a2 = F.one_hot(attr_2, 10)
    a3 = F.one_hot(attr_3, 101)
    state = torch.cat([first] + to_FloatTensor(a1, a2, a3), dim=-1)

    decode_a1 = one_hot_decode(a1).type(torch.FloatTensor).to(device)
    decode_a2 = one_hot_decode(a2).type(torch.FloatTensor).to(device)
    decode_a3 = one_hot_decode(a3).type(torch.FloatTensor).to(device)
    data = torch.cat((first, decode_a1, decode_a2, decode_a3), dim=-1)
    assert torch.allclose(data, features_tensor)

