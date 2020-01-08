import math

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from model.critic import Value
from model.multidiscriminator import MultiDiscriminator
from model.multipolicy import MultiPolicy
from ppo_multi import ppo_step
from utils import ModelArgs, device, FloatTensor, to_FloatTensor, one_hot_decode
import gym
from tqdm import tqdm
import time
from matplotlib import pyplot as plt
import pandas as pd

args = ModelArgs()
assert args.n_state == args.n_onehot_state + args.n_multihot_state + args.n_continuous_state
assert args.n_action == args.n_onehot_action + args.n_multihot_action + args.n_continuous_action
# onehot sections of state and action
# sas
#onehot_action_sections = [0]
#onehot_state_sections = [5, 2, 4, 3, 2, 9, 2, 32, 35, 7, 2, 21, 2, 3, 3]

# ungroup 
# onehot_state_sections = [0]
# onehot_action_sections = [5, 2, 4, 3, 2, 10, 2, 33, 39, 8, 2, 25, 2, 3, 3]

# v3
#onehot_state_sections = [5, 2, 4, 2, 2, 10, 2, 34, 40, 8, 2, 26, 2, 3, 3]
#onehot_action_sections = [5, 2, 4, 2, 2, 10, 2, 34, 40, 8, 2, 26, 2, 3, 3]

# unscaled
onehot_state_sections = [5, 2, 3, 2, 5, 2]
onehot_action_sections = [5, 2, 3, 2, 5, 2]
model_name = 'scaled_multip_noise_0.25_gpu7'
write_scalar = True
load_model = False

class ExpertDataSet(torch.utils.data.Dataset):
    def __init__(self, data_set_path):
        self.expert_data = np.array(pd.read_csv(data_set_path))
        self.state = torch.FloatTensor(self.expert_data[:, :args.n_state])
        self.action = torch.FloatTensor(self.expert_data[:, args.n_state:args.n_state + args.n_action])
        self.next_state = torch.FloatTensor(self.expert_data[:, args.n_state + args.n_action:])
        self.length = self.state.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.state[idx], self.action[idx], self.next_state[idx]

def make_d_inputs(state, action, next_state):
    d_state_input = [state, state, state, state]
    # d_state_input = [state, state, state, torch.cat((state, action), dim = -1)]
    d_action_input = [action[:, :args.n_onehot_action],
                      action[:, args.n_onehot_action:-args.n_continuous_action],
                      action[:, -args.n_continuous_action:],
                      next_state[:, -(args.n_continuous_state - args.n_continuous_action):]
                     ]
    return d_state_input, d_action_input


def main():
    # load expert data
    print(args.data_set_path)
    dataset = ExpertDataSet(args.data_set_path)
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=args.expert_batch_size,
        shuffle=True,
        num_workers=0
    )
    p_state_sizes = [args.n_state, args.n_state, args.n_state, args.n_state + args.n_action]
    p_action_sizes = [args.n_onehot_action, args.n_multihot_action, args.n_continuous_action, args.n_continuous_state - args.n_continuous_action]
    
    d_state_sizes = [args.n_state, args.n_state, args.n_state, args.n_state]
    d_action_sizes = [args.n_onehot_action, args.n_multihot_action, args.n_continuous_action, args.n_continuous_state - args.n_continuous_action]

    policy = MultiPolicy(p_state_sizes, p_action_sizes, onehot_action_sections, onehot_state_sections, state_0 = dataset.state)
    discriminator = MultiDiscriminator(d_state_sizes, d_action_sizes)
    discriminator_criterion = nn.BCELoss()
    if write_scalar:
        writer = SummaryWriter(log_dir='runs/' + model_name)

    # load net  models
    if load_model:
        discriminator = torch.load('./model_pkl/multi_policy/D_' + model_name + '.pkl')
        policy = torch.load('./model_pkl/multi_policy/P_' + model_name + '.pkl')
    print('#############  start training  ##############')

    # update discriminator
    num = 0
    for ep in tqdm(range(args.training_epochs)):
        # collect data from environment for ppo update
        policy.train()
        discriminator.train()
        start_time = time.time()
        memory, n_trajs = policy.collect_samples(batch_size=args.sample_batch_size)
        # print('sample_data_time:{}'.format(time.time()-start_time))
        batch = memory.sample()
        gen_state = torch.cat(batch.state, dim=1).reshape(n_trajs * args.sample_traj_length, -1).detach()
        gen_action = torch.cat(batch.action, dim=1).reshape(n_trajs * args.sample_traj_length, -1).detach()
        gen_next_state = torch.cat(batch.next_state, dim=1).reshape(n_trajs * args.sample_traj_length, -1).detach()
        old_log_prob = torch.cat(batch.old_log_prob, dim=1).reshape(n_trajs * args.sample_traj_length, -1).detach()
        mask = torch.cat(batch.mask, dim=1).reshape(n_trajs * args.sample_traj_length, -1).detach()
        
        gen_d_state, gen_d_action = make_d_inputs(gen_state, gen_action, gen_next_state)
        if ep % 1 == 0:
        # if (d_slow_flag and ep % 50 == 0) or (not d_slow_flag and ep % 1 == 0):
            d_loss = torch.empty(0, device=device)
            p_loss = torch.empty(0, device=device)
            v_loss = torch.empty(0, device=device)
            gen_r = torch.empty(0, device=device)
            expert_r = torch.empty(0, device=device)
            for expert_state_batch, expert_action_batch, expert_next_state_batch in data_loader:
                expert_d_state, expert_d_action = make_d_inputs(expert_state_batch.to(device), expert_action_batch.to(device), expert_next_state_batch.to(device))
                gen_r = discriminator(gen_d_state, gen_d_action)
                expert_r = discriminator(expert_d_state, expert_d_action)

                discriminator.optimizer.zero_grad()
                d_loss = discriminator_criterion(gen_r, torch.zeros(gen_r.shape, device=device)) + \
                            discriminator_criterion(expert_r,torch.ones(expert_r.shape, device=device))
                variance = 0.5 * torch.var(gen_r.to(device)) + 0.5 * torch.var(expert_r.to(device))
                total_d_loss = d_loss - 10 * variance
                d_loss.backward()
                # total_d_loss.backward()
                discriminator.optimizer.step()
            if write_scalar:
                writer.add_scalar('loss/d_loss', d_loss, ep)
                writer.add_scalar('loss/total_d_loss', total_d_loss, ep)
                writer.add_scalar('loss/variance', 10 * variance, ep)

        if ep % 1 == 0:
            # update PPO
            gen_r = discriminator(gen_d_state, gen_d_action)
            optimize_iter_num = int(math.ceil(gen_state.shape[0] / args.ppo_mini_batch_size))
            for ppo_ep in range(args.ppo_optim_epoch):
                for i in range(optimize_iter_num):
                    num += 1
                    index = slice(i * args.ppo_mini_batch_size, min((i + 1) * args.ppo_mini_batch_size, gen_state.shape[0]))
                    gen_state_batch, gen_action_batch, gen_next_state_batch, old_log_prob_batch, mask_batch, gen_r_batch = \
                        gen_state[index], gen_action[index], gen_next_state[index], old_log_prob[index], mask[index], gen_r[index]
                    v_loss, p_loss = ppo_step(policy,
                                            gen_state_batch,
                                            gen_action_batch, 
                                            gen_next_state_batch,
                                            gen_r_batch, old_log_prob_batch,
                                            mask_batch, args.ppo_clip_epsilon)
        policy.eval()
        discriminator.eval()
        gen_d_state, gen_d_action = make_d_inputs(gen_state, gen_action, gen_next_state)
        expert_d_state, expert_d_action = make_d_inputs(expert_state_batch.to(device), expert_action_batch.to(device), expert_next_state_batch.to(device))
        gen_r = discriminator(gen_d_state, gen_d_action)
        expert_r = discriminator(expert_d_state, expert_d_action)
        gen_r_noise = gen_r.mean(dim=0)
        expert_r_noise = expert_r.mean(dim=0)
        gen_r = discriminator(gen_d_state, gen_d_action, noise=False)
        expert_r = discriminator(expert_d_state, expert_d_action, noise=False)
        if write_scalar:
            writer.add_scalar('gen_r_accurate/onehot', gen_r.mean(dim=0)[0], ep)
            writer.add_scalar('gen_r_accurate/multihot', gen_r.mean(dim=0)[1], ep)
            writer.add_scalar('gen_r_accurate/continuous', gen_r.mean(dim=0)[2], ep)
            writer.add_scalar('gen_r_accurate/next_state', gen_r.mean(dim=0)[3], ep)
            writer.add_scalar('expert_r_accurate/onehot', expert_r.mean(dim=0)[0], ep)
            writer.add_scalar('expert_r_accurate/multihot', expert_r.mean(dim=0)[1], ep)
            writer.add_scalar('expert_r_accurate/continuous', expert_r.mean(dim=0)[2], ep)
            writer.add_scalar('expert_r_accurate/next_state', expert_r.mean(dim=0)[3], ep)
            writer.add_scalar('gen_r_with_noise/onehot', gen_r_noise[0], ep)
            writer.add_scalar('gen_r_with_noise/multihot', gen_r_noise[1], ep)
            writer.add_scalar('gen_r_with_noise/continuous', gen_r_noise[2], ep)
            writer.add_scalar('gen_r_with_noise/next_state', gen_r_noise[3], ep)
            writer.add_scalar('expert_r_with_noise/onehot', expert_r_noise[0], ep)
            writer.add_scalar('expert_r_with_noise/multihot', expert_r_noise[1], ep)
            writer.add_scalar('expert_r_with_noise/continuous', expert_r_noise[2], ep)
            writer.add_scalar('expert_r_with_noise/next_state', expert_r_noise[3], ep)
            writer.add_scalar('total/gen_r_accurate', gen_r.mean(), ep)
            writer.add_scalar('total/expert_r_accurate', expert_r.mean(), ep)
            writer.add_scalar('total/gen_r_with_noise', gen_r_noise.mean(), ep)
            writer.add_scalar('total/expert_r_with_noise', expert_r_noise.mean(), ep)
        print('#' * 5 + 'training episode:{}'.format(ep) + '#' * 5)
        print('gen_r_noise:', gen_r_noise)
        print('expert_r_noise:', expert_r_noise)
        print('gen_r:', gen_r.mean(dim=0))
        print('expert_r:', expert_r.mean(dim=0))
        print('d_loss', d_loss.item())
        # save models
        if model_name is not None:
            torch.save(discriminator, './model_pkl/multi_policy/D_' + model_name + '.pkl')
            torch.save(policy, './model_pkl/multi_policy/P_' + model_name + '.pkl')
        memory.clear_memory()

if __name__ == '__main__':
    main()
