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
from sklearn.metrics import hamming_loss

args = ModelArgs()
# discrete sections of state and action
# sas
discrete_action_sections = [0]
discrete_state_sections = [5, 2, 4, 3, 2, 9, 2, 32, 35, 7, 2, 21, 2, 3, 3]

# ungroup 
# discrete_state_sections = [0]
# discrete_action_sections = [5, 2, 4, 3, 2, 10, 2, 33, 39, 8, 2, 25, 2, 3, 3]

class ExpertDataSet(torch.utils.data.Dataset):
    def __init__(self, data_set_path):
        self.expert_data = np.array(pd.read_csv(data_set_path))
        self.expert_sa = torch.FloatTensor(self.expert_data[:, :161])
        self.expert_s = torch.FloatTensor(self.expert_data[:, 161:])
        self.expert_state = self.expert_data[:, :155]
        self.expert_action = self.expert_data[:, 155:161]
        self.expert_next_state = self.expert_s
        self.length = self.expert_data.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.FloatTensor(self.expert_state[idx]), torch.FloatTensor(self.expert_action[idx]), torch.FloatTensor(self.expert_next_state[idx])

def multi_one_hot_decode(multi_one_hot_data, sections: list):
    one_hot_data_list = torch.split(multi_one_hot_data, sections, dim=-1)
    return list(map(one_hot_decode, one_hot_data_list))

def test():
    # load std models
    # policy_log_std = torch.load('./model_pkl/policy_net_action_std_model_1.pkl')
    # transition_log_std = torch.load('./model_pkl/transition_net_state_std_model_1.pkl')

    # define actor/critic/discriminator net and optimizer
    policy = Policy(discrete_action_sections, discrete_state_sections)
    value = Value()
    discriminator = Discriminator()
    discriminator_criterion = nn.BCELoss()

    # load expert data
    dataset = ExpertDataSet(args.data_set_path)
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # load net  models
    discriminator.load_state_dict(torch.load('./model_pkl/Discriminator_model_2.pkl'))
    policy.transition_net.load_state_dict(torch.load('./model_pkl/Transition_model_2.pkl'))
    policy.policy_net.load_state_dict(torch.load('./model_pkl/Policy_model_2.pkl'))
    value.load_state_dict(torch.load('./model_pkl/Value_model_2.pkl'))

    discrete_state_loss_list = []
    continous_state_loss_list = []
    action_loss_list = []
    cnt = 0
    for expert_state_batch, expert_action_batch, expert_next_state in data_loader:
        cnt += 1
        expert_state_action = torch.cat((expert_state_batch, expert_action_batch), dim=-1).type(FloatTensor)
        next_discrete_state, next_continuous_state, _ = policy.get_transition_net_state(expert_state_action)
        gen_next_state = torch.cat((next_discrete_state.to(device), next_continuous_state.to(device)), dim=-1)

        loss_func = torch.nn.MSELoss()
        continous_state_loss = loss_func(gen_next_state[:, 132:], expert_next_state[:, 132:])
        discrete_state_loss = hamming_loss(gen_next_state[:, :132], expert_next_state[:, :132].type(torch.LongTensor))


        discrete_action, continuous_action, _ = policy.get_policy_net_action(expert_state_batch.type(FloatTensor))
        gen_action = torch.FloatTensor(continuous_action)
        loss_func = torch.nn.MSELoss()
        action_loss = loss_func(gen_action, expert_action_batch)

        discrete_state_loss_list.append(discrete_state_loss)
        continous_state_loss_list.append(continous_state_loss.item())
        action_loss_list.append(action_loss)
    print(sum(discrete_state_loss_list) / cnt)
    print(sum(continous_state_loss_list) / cnt)
    #print(sum(action_loss_list) / cnt)



if __name__ == '__main__':
    test()
