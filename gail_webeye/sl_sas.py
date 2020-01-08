import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import configparser
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import torch.nn.functional as F
from utils import ModelArgs, device
from env import WebEye

args = ModelArgs()
env = WebEye()
use_gpu = True
discrete_state_sections = [5, 2, 4, 3, 2, 9, 2, 32, 35, 7, 2, 21, 2, 3, 3]
# discrete_state_sections = [0]
# discrete_action_sections = [5,2,4,3,2,10,2,33,39,8,2,25,2,3,3]
discrete_state_dim = sum(discrete_state_sections)

# S：132离散+23连续
# a：6连续
exp_version = '_in_train_env'

class TrainingDataSet(torch.utils.data.Dataset):
    def __init__(self):
        train_data = np.array(pd.read_csv('./train_data_sas.csv'))
        test_data = np.array(pd.read_csv('./test_data_sas.csv'))
        state_dim = args.n_continuous_state + args.n_discrete_state
        action_dim = args.n_continuous_action + args.n_discrete_action
    
        train_state = torch.FloatTensor(train_data[:, :state_dim])
        train_action = torch.FloatTensor(train_data[:, state_dim:state_dim + action_dim])
        train_next_state = torch.FloatTensor(train_data[:, state_dim + action_dim:])
        test_state = torch.FloatTensor(test_data[:, :state_dim])
        test_action = torch.FloatTensor(test_data[:, state_dim:state_dim + action_dim])
        test_next_state = torch.FloatTensor(test_data[:, state_dim + action_dim:])
        x_train = train_state
        y_train_true = train_action
        x_test = test_state
        y_test_true = test_action

        self.train_sa = x_train
        self.train_s = y_train_true
        self.test_sa = x_test
        self.test_s = y_test_true
        self.length = self.train_sa.shape[0]
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return self.train_sa[idx].to(device), self.train_s[idx].to(device)

def one_hot_decode(one_hot_data):
    tmp = (one_hot_data == 1).nonzero()
    one_hot_decode_data = tmp[:, tmp.shape[1] - 1].unsqueeze(1)
    return one_hot_decode_data

def multi_one_hot_decode(multi_one_hot_data, sections: list):
    one_hot_data_list = torch.split(multi_one_hot_data, sections, dim=-1)
    return list(map(one_hot_decode, one_hot_data_list))

def mlp_hybrid_loss(ypred, ytrue):
    mse = torch.nn.MSELoss()
    continuous_loss = mse(ypred, ytrue)
    return continuous_loss

class Net(nn.Module):
    def __init__(self, activation=nn.LeakyReLU()):
        super(Net, self).__init__()
        self.input_dim = args.n_continuous_state + args.n_discrete_state
        self.hidden_dim = args.n_policy_hidden 
        self.output_dim = args.n_continuous_action + args.n_discrete_action
        self.activation = activation
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            activation,
            nn.Linear(self.hidden_dim, self.output_dim)
            # nn.Dropout(0.5)
        ).to(device)
    def forward(self, input):
        out = self.net(input)
        return out

def training():
    net = Net()
    writer = SummaryWriter(log_dir = 'runs/sl_sas' + exp_version)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss_func = nn.MSELoss()
    training_dataset = TrainingDataSet()
    train_data_loader = torch.utils.data.DataLoader(
        dataset=training_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )
    training_epochs = 100000
    reward_sum = 0
    step_cnt = 0
    reward_lst = [0] * 1000
    for ep in tqdm(range(training_epochs)):
        for x_train, y_train_true in train_data_loader:
            y_train_predict = net(x_train)
            optimizer.zero_grad()
            train_total_loss = mlp_hybrid_loss(y_train_predict, y_train_true)
            train_total_loss.backward()
            optimizer.step()
        
        if ep % 10 == 0:
            state = env.reset()
            episode_reward = 0
            episode_cnt = 0
            for step in range(200):
                action = net(state)
                nxt_state, reward, done, _ = env.step(action)
                episode_reward += reward

                reward_lst[step_cnt % 1000] = reward.item()
                state = nxt_state
                if done:
                    state = env.reset()
                    reward_sum += episode_reward
                    episode_reward = 0
                    episode_cnt += 1
                step_cnt += 1
            writer.add_scalar('reward/per_episode', reward_sum / episode_cnt, ep)
            writer.add_scalar('reward/per_step', sum(reward_lst) / len(reward_lst), ep)
            writer.add_scalar('loss', train_total_loss, ep)
            print('reward per episode',reward_sum / episode_cnt)
            print('reward per step', sum(reward_lst)/ len(reward_lst)) 
            print(reward_lst[:10])
            print('reward per step std', torch.FloatTensor(reward_lst).std())
            print('loss:', train_total_loss)

if __name__ == '__main__':
    training()
