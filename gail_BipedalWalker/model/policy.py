import torch
import torch.cuda
import torch.nn as nn
from torch.distributions import MultivariateNormal

from distributions.MultiOneHotCategorical import MultiOneHotCategorical
from utils import (_init_weight, FloatTensor, Memory, device, ModelArgs)

args = ModelArgs()


class CustomSoftMax(nn.Module):
    def __init__(self, discrete_action_dim, sections):
        super().__init__()
        self.discrete_action_dim = discrete_action_dim
        self.sections = sections

    def forward(self, input_tensor: torch.Tensor):
        out = torch.zeros(input_tensor.shape, device=device)
        # print(f'out_shape:{out.shape}')
        out[..., :self.discrete_action_dim] = torch.cat(
            [tensor.softmax(dim=-1) for tensor in
             torch.split(input_tensor[..., :self.discrete_action_dim], self.sections, dim=-1)],
            dim=-1)
        out[..., self.discrete_action_dim:] = input_tensor[..., self.discrete_action_dim:].tanh()
        # print(f'out_shape_2:{out.shape}')
        return out


class Policy(nn.Module):
    def forward(self, *input):
        raise NotImplementedError

    def __init__(self, discrete_action_sections: list, discrete_state_sections: list, action_log_std=0,
                 state_log_std=0, max_traj_length=2000, n_continuous_action=args.n_continuous_action,
                 n_discrete_action=args.n_discrete_action, n_discrete_state=args.n_discrete_state,
                 n_continuous_state=args.n_continuous_state, device=device):
        super(Policy, self).__init__()
        self.n_action = n_continuous_action + n_discrete_action
        self.n_state = n_discrete_state + n_continuous_state
        self.discrete_action_dim = n_discrete_action
        self.discrete_state_dim = n_discrete_state
        self.discrete_action_sections = discrete_action_sections
        self.discrete_state_sections = discrete_state_sections
        self.discrete_action_sections_len = len(discrete_action_sections)
        self.discrete_state_sections_len = len(discrete_state_sections)
        n_policy_hidden = args.n_policy_hidden
        n_transition_hidden = args.n_transition_hidden
        self.policy_net = nn.Sequential(
            nn.Linear(self.n_state, n_policy_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_policy_hidden, n_policy_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_policy_hidden, n_policy_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_policy_hidden, self.n_action),
            CustomSoftMax(self.discrete_action_dim, discrete_action_sections)
        ).to(device)
        self.device = device
        self.transition_net = nn.Sequential(
            nn.Linear(self.n_state + self.n_action, n_transition_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_transition_hidden, n_transition_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_transition_hidden, n_transition_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_transition_hidden, self.n_state),
            CustomSoftMax(self.discrete_state_dim, discrete_state_sections)
        ).to(device)
        self.policy_net.apply(_init_weight)
        self.transition_net.apply(_init_weight)
        self.max_traj_length = max_traj_length
        self.policy_net_action_std = nn.Parameter(
            torch.ones(1, n_continuous_action, device=device) * action_log_std)
        self.transition_net_state_std = nn.Parameter(
            torch.ones(1, n_continuous_state, device=device) * state_log_std)
        add_Policy_class_method(Policy, 'policy_net', 'action')
        add_Policy_class_method(Policy, 'transition_net', 'state')

    def collect_samples(self, mini_batch_size, size=1):
        num_step = 0
        memory = Memory()
        while num_step < mini_batch_size:
            discrete_state, continuous_state = self.reset(size)
            for walk_step in range(self.max_traj_length - 1):
                with torch.no_grad():
                    discrete_action, continuous_action, next_discrete_state, next_continuous_state, old_log_prob = self.step(
                        discrete_state, continuous_state,size)
                # Currently we assume the exploration step is not done until it reaches max_traj_length.
                mask = torch.ones((size, 1), device=device)
                memory.push(discrete_state.type(FloatTensor), continuous_state, discrete_action.type(FloatTensor),
                            continuous_action, next_discrete_state.type(FloatTensor),
                            next_continuous_state, old_log_prob, mask)
                discrete_state, continuous_state = next_discrete_state, next_continuous_state
                num_step += 1
                if num_step >= mini_batch_size:
                    return memory
            # one more step for push done
            with torch.no_grad():
                discrete_action, continuous_action, next_discrete_state, next_continuous_state, old_log_prob = self.step(
                    discrete_state, continuous_state,size)
                mask = torch.zeros((size, 1), device=device)
                memory.push(discrete_state.type(FloatTensor), continuous_state, discrete_action.type(FloatTensor),
                            continuous_action, next_discrete_state.type(FloatTensor),
                            next_continuous_state, old_log_prob, mask)
                num_step += 1
        return memory

    def step(self, cur_discrete_state, cur_continuous_state, size=1):
        state = torch.cat((cur_discrete_state.type(FloatTensor),
                           cur_continuous_state.type(FloatTensor)), dim=-1)
        discrete_action, continuous_action, policy_net_log_prob = self.get_policy_net_action(state,size)

        next_discrete_state, next_continuous_state, transition_net_log_prob = self.get_transition_net_state(
            torch.cat((state, discrete_action.type(FloatTensor),
                       continuous_action), dim=-1),size)
        return discrete_action, continuous_action, next_discrete_state, next_continuous_state, \
               policy_net_log_prob + transition_net_log_prob

    def reset(self, size=1):
        discrete_state = torch.empty((size, 0), device=self.device)
        continuous_state = torch.empty((size, 0), device=self.device)
        if self.discrete_action_dim != 0:
            discrete_state = torch.randn(size=(size, self.discrete_state_dim), device=self.device)
        if self.n_state - self.discrete_state_dim != 0:
            continuous_state = torch.randn(size=(size, self.n_state - self.discrete_state_dim), device=self.device)

        return discrete_state, continuous_state


def add_Policy_class_method(cls, net_name: str, action_name: str):
    def get_net_action(self, state,size=1):
        net = getattr(self, net_name)
        n_action_dim = getattr(self, 'n_' + action_name)
        discrete_action_dim = getattr(self, 'discrete_' + action_name + '_dim')
        sections = getattr(self, 'discrete_' + action_name + '_sections')
        continuous_action_log_std = getattr(self, net_name + '_' + action_name + '_std')
        discrete_action_probs_with_continuous_mean = net(state)
        discrete_actions = torch.empty((size,0), device=self.device)
        continuous_actions = torch.empty((size,0), device=self.device)
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

    def get_net_log_prob(self, net_input_state, net_input_discrete_action, net_input_continuous_action):
        net = getattr(self, net_name)
        n_action_dim = getattr(self, 'n_' + action_name)
        discrete_action_dim = getattr(self, 'discrete_' + action_name + '_dim')
        sections = getattr(self, 'discrete_' + action_name + '_sections')
        continuous_action_log_std = getattr(self, net_name + '_' + action_name + '_std')
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

    setattr(cls, 'get_' + net_name + '_' + action_name, get_net_action)
    setattr(cls, 'get_' + net_name + '_' + 'log_prob', get_net_log_prob)


__all__ = ['Policy', 'CustomSoftMax']
