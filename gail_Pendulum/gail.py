import math

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter

from model.critic import Value
from model.discriminator import Discriminator
from model.policy import Policy
from ppo import ppo_step
from utils import ModelArgs, device, FloatTensor
import gym
from tqdm import tqdm
import time
from matplotlib import pyplot as plt

args = ModelArgs()
# discrete sections of state and action
discrete_action_sections = [0]
discrete_state_sections = [0]

class ExpertDataSet(torch.utils.data.Dataset):
    def __init__(self, activity_file_name, cost_file_name):
        self.c = torch.from_numpy(np.genfromtxt(cost_file_name))
        self.a = torch.from_numpy(np.genfromtxt(activity_file_name)).unsqueeze(1)
        self.length = len(self.c)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.c[idx], self.a[idx]

    @staticmethod
    def normalization(x):
        scalar = StandardScaler()
        x = scalar.fit_transform(x)
        return x


# todo: test GAE and ppo_step function works when collect_samples size > 1
# todo: use model_dict or some others nn.Module class adjust policy class method to forward
def main():
    # define actor/critic/discriminator net and optimizer
    policy = Policy(discrete_action_sections, discrete_state_sections)
    value = Value()
    discriminator = Discriminator()
    optimizer_policy = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
    optimizer_value = torch.optim.Adam(value.parameters(), lr=args.value_lr)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.discrim_lr)
    discriminator_criterion = nn.BCELoss()
    writer = SummaryWriter()

    # load expert data
    dataset = ExpertDataSet(args.expert_activities_data_path, args.expert_cost_data_path)
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=args.expert_batch_size,
        shuffle=False,
        num_workers=1
    )

    # load models
    # discriminator.load_state_dict(torch.load('./model_pkl/Discriminator_model_3.pkl'))
    # policy.transition_net.load_state_dict(torch.load('./model_pkl/Transition_model_3.pkl'))
    # policy.policy_net.load_state_dict(torch.load('./model_pkl/Policy_model_3.pkl'))
    # value.load_state_dict(torch.load('./model_pkl/Value_model_3.pkl'))

    print('#############  start training  ##############')

    # update discriminator
    num = 0
    for ep in tqdm(range(args.training_epochs)):
        # collect data from environment for ppo update
        start_time = time.time()
        memory = policy.collect_samples(args.ppo_buffer_size, size=10000)
        # print('sample_data_time:{}'.format(time.time()-start_time))
        batch = memory.sample()
        continuous_state = torch.stack(batch.continuous_state).squeeze(1).detach()
        discrete_action = torch.stack(batch.discrete_action).squeeze(1).detach()
        continuous_action = torch.stack(batch.continuous_action).squeeze(1).detach()
        next_discrete_state = torch.stack(batch.next_discrete_state).squeeze(1).detach()
        next_continuous_state = torch.stack(batch.next_continuous_state).squeeze(1).detach()
        old_log_prob = torch.stack(batch.old_log_prob).detach()
        mask = torch.stack(batch.mask).squeeze(1).detach()
        discrete_state = torch.stack(batch.discrete_state).squeeze(1).detach()
        d_loss = torch.empty(0, device=device)
        p_loss = torch.empty(0, device=device)
        v_loss = torch.empty(0, device=device)
        gen_r = torch.empty(0, device=device)
        expert_r = torch.empty(0, device=device)
        for _ in range(1):
            for expert_state_batch, expert_action_batch in data_loader:
                gen_state = torch.cat((discrete_state, continuous_state), dim=-1)
                gen_action = torch.cat((discrete_action, continuous_action), dim=-1)
                gen_r = discriminator(gen_state, gen_action)
                expert_r = discriminator(expert_state_batch, expert_action_batch)
                optimizer_discriminator.zero_grad()
                d_loss = discriminator_criterion(gen_r,
                                                 torch.zeros(gen_r.shape, device=device)) + \
                         discriminator_criterion(expert_r,
                                                 torch.ones(expert_r.shape, device=device))
                total_d_loss = d_loss - 10 * torch.var(gen_r.to(device))
                d_loss.backward()
                # total_d_loss.backward()
                optimizer_discriminator.step()
        writer.add_scalar('d_loss', d_loss, ep)
        # writer.add_scalar('total_d_loss', total_d_loss, ep)
        writer.add_scalar('expert_r', expert_r.mean(), ep)

        # update PPO
        gen_r = discriminator(torch.cat((discrete_state, continuous_state), dim=-1),
                              torch.cat((discrete_action, continuous_action), dim=-1))
        optimize_iter_num = int(math.ceil(discrete_state.shape[0] / args.ppo_mini_batch_size))
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
            writer.add_scalar('p_loss', p_loss, num)
            writer.add_scalar('v_loss', v_loss, num)
            writer.add_scalar('gen_r', gen_r.mean(), num)

        print('#' * 5 + 'training episode:{}'.format(ep) + '#' * 5)
        print('d_loss', d_loss.item())
        # print('p_loss', p_loss.item())
        # print('v_loss', v_loss.item())
        print('gen_r:', gen_r.mean().item())
        print('expert_r:', expert_r.mean().item())

        memory.clear_memory()
        # save models
        torch.save(discriminator.state_dict(), './model_pkl/Discriminator_model_4.pkl')
        torch.save(policy.transition_net.state_dict(), './model_pkl/Transition_model_4.pkl')
        torch.save(policy.policy_net.state_dict(), './model_pkl/Policy_model_4.pkl')
        torch.save(value.state_dict(), './model_pkl/Value_model_4.pkl')


def test():
    env_name = "Pendulum-v0"
    env = gym.make(env_name)
    render = True
    ppo = Policy([0], [0])
    ppo.policy_net.load_state_dict(torch.load('./model_pkl/Policy_model_4.pkl'))
    for ep in range(500):
        ep_reward = 0
        state = env.reset()
        # env.render()
        for t in range(200):
            state = torch.unsqueeze(torch.from_numpy(state).type(torch.FloatTensor), 0).to(device)
            # _, action, _ = ppo.get_policy_net_action(state,size=10000)
            discrete_action_probs_with_continuous_mean = ppo.policy_net(state)
            action = discrete_action_probs_with_continuous_mean[:, 0:]
            action = torch.squeeze(action, 1)
            # print(action)
            action = action.cpu().detach().numpy()
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            env.render()
            if done:
                break
        # writer.add_scalar('ep_reward', ep_reward, ep)
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        env.close()

def evaluate_env():
    env_name = "Pendulum-v0"
    env = gym.make(env_name)
    ppo = Policy([0], [0])
    ppo.policy_net.load_state_dict(torch.load('./model_pkl/Policy_model_2.pkl'))
    ppo.transition_net.load_state_dict(torch.load('./model_pkl/Transition_model_2.pkl'))
    state = env.reset()
    t = 0
    value_list = []
    for j in range(50):
        state = env.reset()
        for i in range(200):
            t += 1
            if(i == 0):
                gen_state = torch.unsqueeze(torch.from_numpy(state).type(torch.FloatTensor), 0)
                real_state = torch.unsqueeze(torch.from_numpy(state).type(torch.FloatTensor), 0)
            else:
                _, action, _ = ppo.get_policy_net_action(gen_state.to(device), size=10000)
                _, gen_state, _ = ppo.get_transition_net_state(torch.cat((gen_state.to(device), action), dim=-1), size=10000)
                action = torch.squeeze(action, 1)
                action = action.cpu().numpy()
                real_state, reward, done, _ = env.step(action)
                value = torch.dist((gen_state.to(device)).float(), (torch.from_numpy(real_state).unsqueeze(0)).float().to(device), p=2)
                value_list.append(value)
                if done:
                    i = 0
    plt.plot(np.linspace(0,100,len(value_list)), value_list)
    plt.show()


if __name__ == '__main__':
    # evaluate_env()
    test()
    # main()
