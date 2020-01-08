import math

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import time
from torch.distributions import MultivariateNormal
from model.critic import Value
from model.policy import Policy
from model.discriminator import Discriminator
from ppo_for_rl import ppo_step
from utils import ModelArgs, device, FloatTensor, to_FloatTensor, one_hot_decode, _init_weight
from tqdm import tqdm
from collections import namedtuple
from matplotlib import pyplot as plt
import pandas as pd
from env import WebEye

load_model = True
save_model = False
write_scalar = False
args = ModelArgs()
env = WebEye()
# discrete sections of state and action
# sas
discrete_action_sections = [0]
discrete_state_sections = [5, 2, 4, 3, 2, 9, 2, 32, 35, 7, 2, 21, 2, 3, 3]
# ungroup 
# discrete_state_sections = [0]
# discrete_action_sections = [5, 2, 4, 3, 2, 10, 2, 33, 39, 8, 2, 25, 2, 3, 3]
Transition = namedtuple('Transition', (
    'discrete_state', 'continuous_state', 'discrete_action', 'continuous_action',
    'next_discrete_state', 'next_continuous_state', 'old_log_prob', 'reward', 'mask'
))
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
class ExpertDataSet(torch.utils.data.Dataset):
    def __init__(self, data_set_path):
        self.expert_data = np.array(pd.read_csv(data_set_path))
        self.state = torch.FloatTensor(self.expert_data[:, :args.n_discrete_state + args.n_continuous_state])
        self.action = torch.FloatTensor(self.expert_data[:, args.n_discrete_state + args.n_continuous_state:args.n_discrete_state + args.n_continuous_state + args.n_discrete_action + args.n_continuous_action])
        self.next_state = torch.FloatTensor(self.expert_data[:, args.n_discrete_state + args.n_continuous_state + args.n_discrete_action + args.n_continuous_action:])
        self.length = self.state.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.state[idx], self.action[idx]

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.n_action = args.n_continuous_action + args.n_discrete_action
        self.n_state = args.n_discrete_state + args.n_continuous_state
        self.policy = nn.Sequential(
            nn.Linear(self.n_state, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_action),
        ).to(device)
        self.policy.apply(_init_weight)
        self.policy_net_action_std = nn.Parameter(
            torch.ones(1, args.n_continuous_action, device=device) * 0)

    def get_action(self, state, num_trajs=1):
        net = self.policy
        n_action_dim = args.n_continuous_action + args.n_discrete_action
        discrete_action_dim = args.n_discrete_action
        sections = discrete_action_sections
        continuous_action_log_std = self.policy_net_action_std

        discrete_action_probs_with_continuous_mean = self.policy(state)
        discrete_actions = torch.empty((num_trajs,0), device=device)
        continuous_actions = torch.empty((num_trajs,0), device=device)
        discrete_actions_log_prob = 0
        continuous_actions_log_prob = 0
        if discrete_action_dim != 0:
            dist = MultiOneHotCategorical(discrete_action_probs_with_continuous_mean[..., :discrete_action_dim],
                                          sections)
            discrete_actions = dist.sample()
            discrete_actions_log_prob = dist.log_prob(discrete_actions)
        if n_action_dim - discrete_action_dim != 0:
            continuous_actions_mean = discrete_action_probs_with_continuous_mean[..., discrete_action_dim:]
            continuous_log_std = continuous_action_log_std.expand_as(continuous_actions_mean)
            continuous_actions_std = torch.exp(continuous_log_std)
            continuous_dist = MultivariateNormal(continuous_actions_mean, torch.diag_embed(continuous_actions_std))
            continuous_actions = continuous_dist.sample()
            continuous_actions_log_prob = continuous_dist.log_prob(continuous_actions)

        return discrete_actions, continuous_actions, FloatTensor(
                discrete_actions_log_prob + continuous_actions_log_prob).unsqueeze(-1)

    def get_policy_net_log_prob(self, net_input_state, net_input_discrete_action, net_input_continuous_action):
        net = self.policy
        n_action_dim = args.n_continuous_action + args.n_discrete_action
        discrete_action_dim = args.n_discrete_action
        sections = discrete_action_sections
        continuous_action_log_std = self.policy_net_action_std
        
        discrete_action_probs_with_continuous_mean = net(net_input_state)
        discrete_actions_log_prob = 0
        continuous_actions_log_prob = 0
        if discrete_action_dim != 0:
            dist = MultiOneHotCategorical(discrete_action_probs_with_continuous_mean[..., :discrete_action_dim],
                                          sections)
            discrete_actions_log_prob = dist.log_prob(net_input_discrete_action)
        if n_action_dim - discrete_action_dim != 0:
            continuous_actions_mean = discrete_action_probs_with_continuous_mean[..., discrete_action_dim:]
            continuous_log_std = continuous_action_log_std.expand_as(continuous_actions_mean)
            continuous_actions_std = torch.exp(continuous_log_std)
            continuous_dist = MultivariateNormal(continuous_actions_mean, torch.diag_embed(continuous_actions_std))
            continuous_actions_log_prob = continuous_dist.log_prob(net_input_continuous_action)
        return FloatTensor(discrete_actions_log_prob + continuous_actions_log_prob).unsqueeze(-1)
        
def collect_samples(actor, batch_size=1):
    memory = Memory()
    returns = []
    episode_reward = 0
    num_trajs = (batch_size + args.sample_traj_length - 1) // args.sample_traj_length
    state = env.reset(num_trajs)
    discrete_state, continuous_state = state[:, :args.n_discrete_state], state[:, args.n_discrete_state:]
    while True:
        with torch.no_grad():
            discrete_action, continuous_action, old_log_prob = actor.get_action(state, num_trajs)
            next_state, reward, done, _ = env.step(torch.cat((discrete_action, continuous_action), dim=-1))
            next_discrete_state, next_continuous_state = next_state[:, :args.n_discrete_state], next_state[:, args.n_discrete_state:]
            episode_reward = episode_reward + reward
        if not done:
            mask = torch.ones((num_trajs, 1), device=device)
        else:
            mask = torch.zeros((num_trajs, 1), device=device)
        memory.push(discrete_state.type(FloatTensor), 
                    continuous_state, 
                    discrete_action.type(FloatTensor),
                    continuous_action,
                    next_discrete_state.type(FloatTensor),
                    next_continuous_state,
                    old_log_prob,
                    reward.reshape(num_trajs, 1),
                    mask)
        discrete_state, continuous_state = next_discrete_state, next_continuous_state
        state = torch.cat((discrete_state, continuous_state), dim = -1)
        if done: 
            returns.append(episode_reward)
            break
    return memory, returns, num_trajs

def main():
    # define actor/critic/discriminator net and optimizer
    policy = Actor()
    value = Value()
    if load_model:
        policy.policy.load_state_dict(torch.load('./model_pkl/ppo/Policy_model_sas_train.pkl'))
        policy.policy_net_action_std = torch.load('./model_pkl/ppo/policy_net_action_std_model_sas_train.pkl')
        value.load_state_dict(torch.load('./model_pkl/ppo/Value_model_sas_train.pkl'))
    optimizer_policy = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
    optimizer_value = torch.optim.Adam(value.parameters(), lr=args.value_lr)
    discriminator_criterion = nn.BCELoss()
    if write_scalar:
        writer = SummaryWriter()

    print('#############  start training  ##############')

    num = 0
    for ep in tqdm(range(args.training_epochs)):
        # collect data from environment for ppo update
        start_time = time.time()
        memory, returns, num_trajs = collect_samples(policy, batch_size=args.sample_batch_size)
        batch = memory.sample()
        discrete_state = torch.cat(batch.discrete_state, dim = 1).reshape(num_trajs * args.sample_traj_length, -1).detach()
        continuous_state = torch.cat(batch.continuous_state, dim = 1).reshape(num_trajs * args.sample_traj_length, -1).detach()
        discrete_action = torch.cat(batch.discrete_action, dim = 1).reshape(num_trajs * args.sample_traj_length, -1).detach()
        continuous_action = torch.cat(batch.continuous_action, dim = 1).reshape(num_trajs * args.sample_traj_length, -1).detach()
        next_discrete_state = torch.cat(batch.next_discrete_state, dim = 1).reshape(num_trajs * args.sample_traj_length, -1).detach()
        next_continuous_state = torch.cat(batch.next_continuous_state, dim = 1).reshape(num_trajs * args.sample_traj_length, -1).detach()

        old_log_prob = torch.cat(batch.old_log_prob, dim = 1).reshape(num_trajs * args.sample_traj_length, -1).detach()
        mask = torch.cat(batch.mask, dim = 1).reshape(num_trajs * args.sample_traj_length, -1).detach()
        
        gen_r = torch.cat(batch.reward, dim = 1).reshape(num_trajs * args.sample_traj_length, -1).detach()
        optimize_iter_num = int(math.ceil(discrete_state.shape[0] / args.ppo_mini_batch_size))
        if write_scalar:
            writer.add_scalar('reward', gen_r.mean(), ep)
        # gen_r = -(1 - gen_r + 1e-10).log()
        for ppo_ep in range(args.ppo_optim_epoch):
            for i in range(optimize_iter_num):
                num += 1
                index = slice(i * args.ppo_mini_batch_size,
                            min((i + 1) * args.ppo_mini_batch_size, discrete_state.shape[0]))
                discrete_state_batch, continuous_state_batch, discrete_action_batch, continuous_action_batch, \
                old_log_prob_batch, mask_batch, next_discrete_state_batch, next_continuous_state_batch, gen_r_batch = \
                    discrete_state[index], continuous_state[index], discrete_action[index], continuous_action[index], \
                    old_log_prob[index], mask[index], next_discrete_state[index], next_continuous_state[index], gen_r[
                        index]
                v_loss, p_loss = ppo_step(policy, value, optimizer_policy, optimizer_value,
                                        discrete_state_batch,
                                        continuous_state_batch,
                                        discrete_action_batch, continuous_action_batch,
                                        next_discrete_state_batch,
                                        next_continuous_state_batch,
                                        gen_r_batch, old_log_prob_batch,
                                        mask_batch, args.ppo_clip_epsilon)
            if write_scalar:
                writer.add_scalar('p_loss', p_loss, ep)
                writer.add_scalar('v_loss', v_loss, ep)
                writer.add_scalar('return', returns[0].mean(), ep)
        print('#' * 5 + 'training episode:{}'.format(ep) + '#' * 5)
        print('gen_r:', gen_r.mean())
        print('gen_r std:', gen_r.std())
        print('returns:', returns[0].mean())
        if ep % 10000 == 0: 
            print(discrete_state, continuous_state, discrete_action, continuous_action)
        # save models
        if save_model:
            torch.save(policy.policy.state_dict(), './model_pkl/ppo/Policy_model_sas_train.pkl')
            torch.save(policy.policy_net_action_std, './model_pkl/ppo/policy_net_action_std_model_sas_train.pkl')
            torch.save(value.state_dict(), './model_pkl/ppo/Value_model_sas_train.pkl')
        memory.clear_memory()

if __name__ == '__main__':
    main()
