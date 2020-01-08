
import gym
import math
import torch
import random
import time, sys
import configparser
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gym import wrappers
from ddpg import DDPG
from copy import deepcopy
from collections import namedtuple
from tqdm import tqdm
from env import WebEye
from utils import device
from torch.utils.tensorboard import SummaryWriter

FLOAT = torch.FloatTensor
LONG = torch.LongTensor

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale

env = WebEye()

env.seed(0)
np.random.seed(0)
torch.manual_seed(0)

agent = DDPG(gamma = 0.95, tau = 0.1, hidden_size = 256,
                    num_inputs = env.observation_space.shape[0], action_space = env.action_space)

memory = ReplayMemory(1000000)

ounoise = OUNoise(env.action_space.shape[0])
param_noise = None

rewards = []
total_numsteps = 0
updates = 0
writer = SummaryWriter()

for i_episode in tqdm(range(100000)):
    state = env.reset()

    episode_reward = 0
    while True:
        action = agent.select_action(state, ounoise, param_noise)
        next_state, reward, done, _ = env.step(action)
        total_numsteps += 1
        episode_reward += reward

        mask = torch.Tensor([not done]).to(device)

        for i in range(state.shape[0]):
            memory.push(state[i], action[i], mask, next_state[i], reward[i])

        state = next_state
        if i_episode % 10 == 0 and len(memory) > 128:
            for _ in range(2):
                transitions = memory.sample(128)
                batch = Transition(*zip(*transitions))
                value_loss, policy_loss = agent.update_parameters(batch)

                updates += 1
        if done:
            writer.add_scalar('reward', episode_reward, i_episode)
            break

    rewards.append(episode_reward.item())
    if i_episode % 10 == 0:
        # rewards.append(episode_reward)
        print("Episode: {}, average return: {}".format(i_episode, sum(rewards[-20:]) / 20))
    if i_episode % 100 == 0:
        agent.save_model("policy", suffix = "test")
    
env.close()