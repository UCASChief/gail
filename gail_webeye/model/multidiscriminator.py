import torch
import torch.nn as nn
from utils import _init_weight, device, ModelArgs, FloatTensor

args = ModelArgs()


class MultiDiscriminator(nn.Module):
    def __init__(self, state_sizes, action_sizes, device=device):
        super(MultiDiscriminator, self).__init__()
        assert len(state_sizes) == len(action_sizes)
        self.device = device
        self.n_discriminators = len(state_sizes)
        self.state_sizes = state_sizes  
        self.action_sizes = action_sizes
        self.discriminator_arr = []
        self.params = []
        for i in range(self.n_discriminators):
            self.discriminator_arr.append(self.__create_d_network(state_sizes[i], action_sizes[i]))
            self.params.append({'params':self.discriminator_arr[-1].parameters()})
        self.optimizer = torch.optim.Adam(self.params, lr = args.discrim_lr)
    
    def __create_d_network(self, state_dim, action_dim):
        net = nn.Sequential(
            nn.Linear(state_dim + action_dim, args.n_discriminator_hidden),
            nn.LeakyReLU(),
            nn.Linear(args.n_discriminator_hidden, 1),
            nn.Sigmoid()
        ).to(self.device)
        net.apply(_init_weight)
        return net

    def forward(self, state, action, noise=True):
        ret_value = torch.empty(size=(state[0].shape[0], self.n_discriminators)).to(self.device)
        for i in range(self.n_discriminators):
            state_action = torch.cat((state[i].type(FloatTensor), action[i].type(FloatTensor)), dim=-1)
            if noise:
                state_action += torch.normal(0, args.noise_std, size=state_action.shape, device=self.device)
            ret_value[:, i:i+1] = self.discriminator_arr[i](state_action)
        return ret_value

__all__ = ['Discriminator']
