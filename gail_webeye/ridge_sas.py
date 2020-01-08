import torch
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


def one_hot_decode(one_hot_data):
    tmp = (one_hot_data == 1).nonzero()
    one_hot_decode_data = tmp[:, tmp.shape[1] - 1].unsqueeze(1)
    return one_hot_decode_data

def multi_one_hot_decode(multi_one_hot_data, sections: list):
    one_hot_data_list = torch.split(multi_one_hot_data, sections, dim=-1)
    return list(map(one_hot_decode, one_hot_data_list))

def hybrid_loss(ypred, ytrue):
    def cross_entropy(logits, labels):
        return F.cross_entropy(logits, labels.type(torch.int64).squeeze())

    y_discrete_pred = torch.split(torch.tensor(ypred[:, :discrete_state_dim]), discrete_state_sections, dim=-1)
    y_discrete_true = multi_one_hot_decode(torch.Tensor(ytrue[:, :discrete_state_dim]), discrete_state_sections)
    discrete_loss_list = list(map(cross_entropy, y_discrete_pred, y_discrete_true))
    # print(f'discrete_loss_list: {discrete_loss_list}')
    discrete_loss_sum = sum(discrete_loss_list)
    continuous_loss = mean_squared_error(ypred[:, discrete_state_dim:], ytrue[:, discrete_state_dim:])
    return discrete_loss_sum.numpy() + continuous_loss, discrete_loss_sum, continuous_loss

def loss(ypred, ytrue):
    loss = mean_squared_error(ypred[:, discrete_state_dim:], ytrue[:, discrete_state_dim:])
    return loss

def supervised_learning(algorithm):
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
    
    print(train_next_state[:, args.n_discrete_state - 1 + 8].mean())
    print(test_next_state[:, args.n_discrete_state - 1 + 8].mean())

    print(train_state.shape, train_action.shape, train_next_state.shape)
    print(test_state.shape, test_action.shape, test_next_state.shape)
    # x_train = torch.cat((state, action), dim=-1)[:-test_dataset_num, :]
    # y_train_true = next_state[:-test_dataset_num, :]
    # x_test = torch.cat((state, action), dim=-1)[-test_dataset_num:, :]
    # y_test_true = next_state[-test_dataset_num:, :]
    # x_train = torch.empty(state.shape[1]+action.shape[1]).view(1, -1)
    # y_train_true = torch.empty(state.shape[1]).view(1, -1)
    # x_test = torch.empty(state.shape[1]+action.shape[1]).view(1, -1)
    # y_test_true = torch.empty(state.shape[1]).view(1, -1)

    x_train = train_state
    y_train_true = train_action
    x_test = test_state
    y_test_true = test_action
    train_data = torch.cat((x_train, y_train_true), dim=-1)
    test_data = torch.cat((x_test, y_test_true), dim=-1)
    # np.savetxt('./tarin_data_sas.txt', train_data)
    # np.savetxt('./test_data_sas.txt', test_data)
    
    algorithm.fit(x_train, y_train_true)
    state = env.reset()
    episode_reward, reward_sum = 0, 0
    episode_cnt = 0
    for step in range(5000):
        action = algorithm.predict(state)
        nxt_state, reward, done, _ = env.step(torch.FloatTensor(action))
        episode_reward += reward

        state = nxt_state
        if done:
            state = env.reset()
            reward_sum += episode_reward
            episode_reward = 0
            episode_cnt += 1
    print(reward_sum / episode_cnt)
    print(reward_sum / 5000)
    print(episode_cnt)
        

def find_best_feature(alpha, max_alpha, step=0.01, loss_idx=1):
    alphas = []
    losses = []
    best_alpha = []
    range_list = [alpha + i * step for i in range(int((max_alpha - alpha) / step))]
    for alpha in tqdm(range_list):
        rg = Ridge(alpha=alpha)
        hybrid_train_loss, hybrid_test_loss = supervised_learning(rg)
        all_loss = hybrid_train_loss, hybrid_test_loss
        losses.append(all_loss[loss_idx])
        alphas.append(alpha)
    best_alpha = list(zip(losses, alphas))
    best_alpha.sort()
    fig = plt.figure()
    plt.title(f"best_alpha:{best_alpha[0][1]},best loss:{best_alpha[0][0]}")
    plt.xlabel("alpha")
    plt.ylabel("loss")
    plt.plot(alphas, losses)
    plt.savefig(f'find_best_alpha')
    plt.close(fig)

if __name__ == '__main__':
    # find_best_feature(0, 0.2, step=0.05, loss_idx=1)

    rg = Ridge(alpha=0.1)
    supervised_learning(rg)
