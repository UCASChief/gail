import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import torch
import torch.nn as nn
from model.policy import Policy
from gail import ExpertDataSet

from utils import (_init_weight, FloatTensor, Memory, device, ModelArgs)
args = ModelArgs()
mode = 'train'
# sas
discrete_action_sections = [0]
discrete_state_sections = [5, 2, 4, 3, 2, 9, 2, 32, 35, 7, 2, 21, 2, 3, 3]

# ungroup 
# discrete_state_sections = [0]
# discrete_action_sections = [5, 2, 4, 3, 2, 10, 2, 33, 39, 8, 2, 25, 2, 3, 3]

dataset = ExpertDataSet(args.data_set_path)

class WebEye(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.n_item = 5
        self.max_c = 100
        self.obs_low = np.concatenate(([0] * args.n_discrete_state, [-5] * args.n_continuous_state))
        self.obs_high = np.concatenate(([1] * args.n_discrete_state, [5] * args.n_continuous_state))
        self.observation_space = spaces.Box(low = self.obs_low, high = self.obs_high, dtype = np.float32)
        self.action_space = spaces.Box(low = -5, high = 5, shape = (args.n_discrete_action + args.n_continuous_action,), dtype = np.float32)
        self.trans_model = Policy(discrete_action_sections, discrete_state_sections, state_0 = dataset.state)
        if mode == 'train':
            self.trans_model.transition_net.load_state_dict(torch.load('./model_pkl/Transition_model_sas_train_4.pkl'))
            self.trans_model.policy_net.load_state_dict(torch.load('./model_pkl/Policy_model_sas_train_4.pkl'))
            self.trans_model.policy_net_action_std = torch.load('./model_pkl/policy_net_action_std_model_sas_train_4.pkl')
        elif mode == 'test':
            self.trans_model.transition_net.load_state_dict(torch.load('./model_pkl/Transition_model_sas_test.pkl'))
            self.trans_model.policy_net.load_state_dict(torch.load('./model_pkl/Policy_model_sas_test.pkl'))
            self.trans_model.policy_net_action_std = torch.load('./model_pkl/policy_net_action_std_model_sas_test.pkl')
        else:
            assert False
        self.reset()

    def seed(self, sd = 0):
        torch.manual_seed(sd)

    @property
    def state(self):
        return torch.cat((self.discrete_state, self.continuous_state), axis = -1)

    def __user_generator(self):
        # with shape(n_user_feature,)
        user = self.user_model.generate()
        self.__leave = self.user_leave_model.predict(user)
        return user

    def _calc_reward(self):
        return self.state[:, args.n_discrete_state - 1 + 8].to(device)
        
    def step(self, action):
        assert action.shape[0] == self.batch_size
        self.length += 1
        self.discrete_state, self.continuous_state, _ = self.trans_model.get_transition_net_state(
            torch.cat((self.state, action), dim = -1)
        )
        done = (self.length >= 5)
        #if done:
        reward = self._calc_reward()
        #else:
        #    reward = torch.zeros(size=(self.batch_size,)).to(device)
        return self.state, reward, done, {}

    def reset(self, batch_size = 1):
        self.batch_size = batch_size
        self.length = 0
        self.discrete_state, self.continuous_state = self.trans_model.reset(num_trajs = self.batch_size)
        return self.state

    def render(self, mode='human', close=False):
        pass
        
