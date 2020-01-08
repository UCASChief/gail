import math

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from model.critic import Value
from model.discriminator import Discriminator
from model.policy import Policy
from ppo import ppo_step
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
model_name = 'scaled_new_train_v3_noise_0.25_gpu1'
#model_name = 'sas_train_test_gpu7'
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
        return self.state[idx], self.action[idx]


# todo: test GAE and ppo_step function works when collect_samples size > 1
# todo: use model_dict or some others nn.Module class adjust policy class method to forward
def main():
    ## load std models
    # policy_log_std = torch.load('./model_pkl/policy_net_action_std_model_1.pkl')
    # transition_log_std = torch.load('./model_pkl/transition_net_state_std_model_1.pkl')

    # load expert data
    print(args.data_set_path)
    dataset = ExpertDataSet(args.data_set_path)
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=args.expert_batch_size,
        shuffle=True,
        num_workers=0
    )
    # define actor/critic/discriminator net and optimizer
    policy = Policy(onehot_action_sections, onehot_state_sections, state_0 = dataset.state)
    value = Value()
    discriminator = Discriminator()
    optimizer_policy = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
    optimizer_value = torch.optim.Adam(value.parameters(), lr=args.value_lr)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.discrim_lr)
    discriminator_criterion = nn.BCELoss()
    if write_scalar:
        writer = SummaryWriter(log_dir='runs/' + model_name)

    # load net  models
    if load_model:
        discriminator.load_state_dict(torch.load('./model_pkl/Discriminator_model_' + model_name + '.pkl'))
        policy.transition_net.load_state_dict(torch.load('./model_pkl/Transition_model_' + model_name + '.pkl'))
        policy.policy_net.load_state_dict(torch.load('./model_pkl/Policy_model_' + model_name + '.pkl'))
        value.load_state_dict(torch.load('./model_pkl/Value_model_' + model_name + '.pkl'))

        policy.policy_net_action_std = torch.load('./model_pkl/Policy_net_action_std_model_' + model_name + '.pkl')
        policy.transition_net_state_std = torch.load('./model_pkl/Transition_net_state_std_model_' + model_name + '.pkl')
    print('#############  start training  ##############')

    # update discriminator
    num = 0
    for ep in tqdm(range(args.training_epochs)):
        # collect data from environment for ppo update
        policy.train()
        value.train()
        discriminator.train()
        start_time = time.time()
        memory, n_trajs = policy.collect_samples(batch_size=args.sample_batch_size)
        # print('sample_data_time:{}'.format(time.time()-start_time))
        batch = memory.sample()
        onehot_state = torch.cat(batch.onehot_state, dim=1).reshape(n_trajs * args.sample_traj_length, -1).detach()
        multihot_state = torch.cat(batch.multihot_state, dim=1).reshape(n_trajs * args.sample_traj_length, -1).detach()
        continuous_state = torch.cat(batch.continuous_state, dim=1).reshape(n_trajs * args.sample_traj_length, -1).detach()

        onehot_action = torch.cat(batch.onehot_action, dim=1).reshape(n_trajs * args.sample_traj_length, -1).detach()
        multihot_action = torch.cat(batch.multihot_action, dim=1).reshape(n_trajs * args.sample_traj_length, -1).detach()
        continuous_action = torch.cat(batch.continuous_action, dim=1).reshape(n_trajs * args.sample_traj_length, -1).detach()
        next_onehot_state = torch.cat(batch.next_onehot_state, dim=1).reshape(n_trajs * args.sample_traj_length, -1).detach()
        next_multihot_state = torch.cat(batch.next_multihot_state, dim=1).reshape(n_trajs * args.sample_traj_length, -1).detach()
        next_continuous_state = torch.cat(batch.next_continuous_state, dim=1).reshape(n_trajs * args.sample_traj_length, -1).detach()

        old_log_prob = torch.cat(batch.old_log_prob, dim=1).reshape(n_trajs * args.sample_traj_length, -1).detach()
        mask = torch.cat(batch.mask, dim=1).reshape(n_trajs * args.sample_traj_length, -1).detach()
        gen_state = torch.cat((onehot_state, multihot_state, continuous_state), dim=-1)
        gen_action = torch.cat((onehot_action, multihot_action, continuous_action), dim=-1)
        if ep % 1 == 0:
        # if (d_slow_flag and ep % 50 == 0) or (not d_slow_flag and ep % 1 == 0):
            d_loss = torch.empty(0, device=device)
            p_loss = torch.empty(0, device=device)
            v_loss = torch.empty(0, device=device)
            gen_r = torch.empty(0, device=device)
            expert_r = torch.empty(0, device=device)
            for expert_state_batch, expert_action_batch in data_loader:
                noise1 = torch.normal(0, args.noise_std, size=gen_state.shape,device=device)
                noise2 = torch.normal(0, args.noise_std, size=gen_action.shape,device=device)
                noise3 = torch.normal(0, args.noise_std, size=expert_state_batch.shape,device=device)
                noise4 = torch.normal(0, args.noise_std, size=expert_action_batch.shape,device=device)
                gen_r = discriminator(gen_state + noise1, gen_action + noise2)
                expert_r = discriminator(expert_state_batch.to(device) + noise3, expert_action_batch.to(device) + noise4)

                # gen_r = discriminator(gen_state, gen_action)
                # expert_r = discriminator(expert_state_batch.to(device), expert_action_batch.to(device))
                optimizer_discriminator.zero_grad()
                d_loss = discriminator_criterion(gen_r, torch.zeros(gen_r.shape, device=device)) + \
                            discriminator_criterion(expert_r,torch.ones(expert_r.shape, device=device))
                variance = 0.5 * torch.var(gen_r.to(device)) + 0.5 * torch.var(expert_r.to(device))
                total_d_loss = d_loss - 10 * variance
                d_loss.backward()
                # total_d_loss.backward()
                optimizer_discriminator.step()
            if write_scalar:
                writer.add_scalar('d_loss', d_loss, ep)
                writer.add_scalar('total_d_loss', total_d_loss, ep)
                writer.add_scalar('variance', 10 * variance, ep)
        if ep % 1 == 0:
            # update PPO
            noise1 = torch.normal(0, args.noise_std, size=gen_state.shape,device=device)
            noise2 = torch.normal(0, args.noise_std, size=gen_action.shape,device=device)
            gen_r = discriminator(gen_state + noise1, gen_action + noise2)
            #if gen_r.mean().item() < 0.1:
            #    d_stop = True
            #if d_stop and gen_r.mean()
            optimize_iter_num = int(math.ceil(onehot_state.shape[0] / args.ppo_mini_batch_size))
            # gen_r = -(1 - gen_r + 1e-10).log()
            for ppo_ep in range(args.ppo_optim_epoch):
                for i in range(optimize_iter_num):
                    num += 1
                    index = slice(i * args.ppo_mini_batch_size,
                                min((i + 1) * args.ppo_mini_batch_size, onehot_state.shape[0]))
                    onehot_state_batch, multihot_state_batch, continuous_state_batch, onehot_action_batch, multihot_action_batch, continuous_action_batch, \
                    old_log_prob_batch, mask_batch, next_onehot_state_batch, next_multihot_state_batch, next_continuous_state_batch, gen_r_batch = \
                        onehot_state[index], multihot_state[index], continuous_state[index], onehot_action[index], multihot_action[index], continuous_action[index], \
                        old_log_prob[index], mask[index], next_onehot_state[index], next_multihot_state[index], next_continuous_state[index], gen_r[
                            index]
                    v_loss, p_loss = ppo_step(policy, value, optimizer_policy, optimizer_value,
                                            onehot_state_batch,
                                            multihot_state_batch,
                                            continuous_state_batch,
                                            onehot_action_batch, 
                                            multihot_action_batch,
                                            continuous_action_batch,
                                            next_onehot_state_batch,
                                            next_multihot_state_batch,
                                            next_continuous_state_batch,
                                            gen_r_batch, old_log_prob_batch,
                                            mask_batch, args.ppo_clip_epsilon)
                    if write_scalar:
                        writer.add_scalar('p_loss', p_loss, ep)
                        writer.add_scalar('v_loss', v_loss, ep)
        policy.eval()
        value.eval()
        discriminator.eval()
        noise1 = torch.normal(0, args.noise_std, size=gen_state.shape,device=device)
        noise2 = torch.normal(0, args.noise_std, size=gen_action.shape,device=device)
        gen_r = discriminator(gen_state + noise1, gen_action + noise2)
        expert_r = discriminator(expert_state_batch.to(device) + noise3, expert_action_batch.to(device) + noise4)
        gen_r_noise = gen_r.mean().item()
        expert_r_noise = expert_r.mean().item()
        gen_r = discriminator(gen_state, gen_action)
        expert_r = discriminator(expert_state_batch.to(device), expert_action_batch.to(device))
        if write_scalar:
            writer.add_scalar('gen_r', gen_r.mean(), ep)
            writer.add_scalar('expert_r', expert_r.mean(), ep)
            writer.add_scalar('gen_r_noise', gen_r_noise, ep)
            writer.add_scalar('expert_r_noise', expert_r_noise, ep)
        print('#' * 5 + 'training episode:{}'.format(ep) + '#' * 5)
        print('gen_r_noise', gen_r_noise)
        print('expert_r_noise', expert_r_noise)
        print('gen_r:', gen_r.mean().item())
        print('expert_r:', expert_r.mean().item())
        print('d_loss', d_loss.item())
        # save models
        if model_name is not None:
            torch.save(discriminator.state_dict(), './model_pkl/Discriminator_model_' + model_name + '.pkl')
            torch.save(policy.transition_net.state_dict(), './model_pkl/Transition_model_' + model_name + '.pkl')
            torch.save(policy.policy_net.state_dict(), './model_pkl/Policy_model_' + model_name + '.pkl')
            torch.save(policy.policy_net_action_std, './model_pkl/Policy_net_action_std_model_' + model_name + '.pkl')
            torch.save(policy.transition_net_state_std, './model_pkl/Transition_net_state_std_model_' + model_name + '.pkl')
            torch.save(value.state_dict(), './model_pkl/Value_model_' + model_name + '.pkl')
        memory.clear_memory()

if __name__ == '__main__':
    main()
