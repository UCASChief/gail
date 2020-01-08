import torch
import torch.cuda
import torch.nn as nn
from torch.distributions import MultivariateNormal

from distributions.MultiOneHotCategorical import MultiOneHotCategorical
from utils import (_init_weight, FloatTensor, Memory, device, ModelArgs)

args = ModelArgs()

class CustomSoftMax(nn.Module):
    def __init__(self, onehot_action_dim, sections):
        super().__init__()
        self.onehot_action_dim = onehot_action_dim
        self.sections = sections

    def forward(self, input_tensor: torch.Tensor):
        out = torch.zeros(input_tensor.shape, device=device)
        out[..., :self.onehot_action_dim] = torch.cat(
            [tensor.softmax(dim=-1) for tensor in
             torch.split(input_tensor[..., :self.onehot_action_dim], self.sections, dim=-1)],
            dim=-1)
        out[..., self.onehot_action_dim:] = input_tensor[..., self.onehot_action_dim:].tanh()
        return out


class Policy(nn.Module):
    def forward(self, *input):
        raise NotImplementedError

    def __init__(self, onehot_action_sections: list, onehot_state_sections: list, action_log_std=0,
                 state_log_std=0, max_traj_length=args.sample_traj_length, n_continuous_action=args.n_continuous_action,
                 n_onehot_action=args.n_onehot_action, n_onehot_state=args.n_onehot_state,
                 n_continuous_state=args.n_continuous_state, device=device, state_0=None):
        super(Policy, self).__init__()
        self.n_action = args.n_action
        self.n_state = args.n_state
        self.onehot_action_dim = n_onehot_action
        self.multihot_action_dim = args.n_multihot_action
        self.onehot_state_dim = n_onehot_state
        self.multihot_state_dim = args.n_multihot_state
        self.onehot_action_sections = onehot_action_sections
        self.onehot_state_sections = onehot_state_sections
        self.onehot_action_sections_len = len(onehot_action_sections)
        self.onehot_state_sections_len = len(onehot_state_sections)
        if state_0 is not None:
            self.state_0 = state_0.to(device)
        else:
            self.state_0 = None
        n_policy_hidden = args.n_policy_hidden
        n_transition_hidden = args.n_transition_hidden
        self.policy_net = nn.Sequential(
            nn.Linear(self.n_state, n_policy_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_policy_hidden, self.n_action),
            CustomSoftMax(self.onehot_action_dim, onehot_action_sections)
        ).to(device)
        self.device = device
        self.transition_net = nn.Sequential(
            nn.Linear(self.n_state + self.n_action, n_transition_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_transition_hidden, self.n_state),
            CustomSoftMax(self.onehot_state_dim, onehot_state_sections)
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

    def collect_samples(self, batch_size=1):
        memory = Memory()
        num_trajs = (batch_size + args.sample_traj_length - 1) // args.sample_traj_length
        onehot_state, multihot_state, continuous_state = self.reset(num_trajs)
        for walk_step in range(self.max_traj_length - 1):
            with torch.no_grad():
                onehot_action, multihot_action, continuous_action, next_onehot_state, next_multihot_state, next_continuous_state, old_log_prob = self.step(onehot_state, multihot_state, continuous_state, num_trajs)
            # Currently we assume the exploration step is not done until it reaches max_traj_length.
            mask = torch.ones((num_trajs, 1), device=device)
            memory.push(onehot_state.type(FloatTensor), multihot_state.type(FloatTensor), continuous_state, onehot_action.type(FloatTensor),
                        multihot_action.type(FloatTensor), continuous_action, next_onehot_state.type(FloatTensor),
                        next_multihot_state.type(FloatTensor), next_continuous_state, old_log_prob, mask)
            onehot_state, multihot_state, continuous_state = next_onehot_state, next_multihot_state, next_continuous_state
        # one more step for push done
        with torch.no_grad():
            onehot_action, multihot_action, continuous_action, next_onehot_state, next_multihot_state, next_continuous_state, old_log_prob = self.step(onehot_state, multihot_state, continuous_state, num_trajs)
            mask = torch.zeros((num_trajs, 1), device=device)
            memory.push(onehot_state.type(FloatTensor), multihot_state.type(FloatTensor), continuous_state, onehot_action.type(FloatTensor),
                        multihot_action.type(FloatTensor), continuous_action, next_onehot_state.type(FloatTensor),
                        next_multihot_state.type(FloatTensor), next_continuous_state, old_log_prob, mask)
        return memory, num_trajs

    def step(self, cur_onehot_state, cur_multihot_state, cur_continuous_state, num_trajs=1):
        state = torch.cat((cur_onehot_state.type(FloatTensor),
                           cur_multihot_state.type(FloatTensor),
                           cur_continuous_state.type(FloatTensor)), dim=-1)
        onehot_action, multihot_action, continuous_action, policy_net_log_prob = self.get_policy_net_action(state, num_trajs)
        next_onehot_state, next_multihot_state, next_continuous_state, transition_net_log_prob = self.get_transition_net_state(
            torch.cat((state, onehot_action.type(FloatTensor),
                       multihot_action.type(FloatTensor),
                       continuous_action), dim=-1), num_trajs)
        if args.concat_action == 1:
            return onehot_action, multihot_action, continuous_action, onehot_action, multihot_action, next_continuous_state, policy_net_log_prob + transition_net_log_prob
        else:
            return onehot_action, multihot_action, continuous_action, next_onehot_state, next_multihot_state, next_continuous_state, policy_net_log_prob + transition_net_log_prob

    def reset(self, num_trajs=1):
        onehot_state = torch.empty((num_trajs, 0), device=self.device)
        multihot_state = torch.empty((num_trajs, 0), device=self.device)
        continuous_state = torch.empty((num_trajs, 0), device=self.device)
        if self.state_0 is None:
            assert False
        else:
            state = self.state_0[torch.randint(self.state_0.shape[0], size=(num_trajs,))]
            onehot_state = state[:, :self.onehot_state_dim]
            multihot_state = state[:, self.onehot_state_dim:self.onehot_state_dim + args.n_multihot_state]
            continuous_state = state[:, self.onehot_state_dim + args.n_multihot_state:]
        return onehot_state, multihot_state, continuous_state


def add_Policy_class_method(cls, net_name: str, action_name: str):
    def get_net_action(self, state, num_trajs=1):
        net = getattr(self, net_name)
        n_action_dim = getattr(self, 'n_' + action_name)
        onehot_action_dim = getattr(self, 'onehot_' + action_name + '_dim')
        multihot_action_dim = getattr(self, 'multihot_' + action_name + '_dim')
        sections = getattr(self, 'onehot_' + action_name + '_sections')
        continuous_action_log_std = getattr(self, net_name + '_' + action_name + '_std')
        onehot_action_probs_with_continuous_mean = net(state)

        onehot_actions = torch.empty((num_trajs, 0), device=self.device)
        multihot_actions = torch.empty((num_trajs, 0), device=self.device)
        continuous_actions = torch.empty((num_trajs, 0), device=self.device)
        onehot_actions_log_prob = 0
        multihot_actions_log_prob = 0
        continuous_actions_log_prob = 0
        if onehot_action_dim != 0:
            dist = MultiOneHotCategorical(onehot_action_probs_with_continuous_mean[..., :onehot_action_dim],
                                          sections)
            onehot_actions = dist.sample()
            onehot_actions_log_prob = dist.log_prob(onehot_actions)
        if multihot_action_dim != 0:
            multihot_actions_prob = torch.sigmoid(onehot_action_probs_with_continuous_mean[..., onehot_action_dim:onehot_action_dim + multihot_action_dim])
            dist = torch.distributions.bernoulli.Bernoulli(probs = multihot_actions_prob) 
            multihot_actions = dist.sample()
            multihot_actions_log_prob = dist.log_prob(multihot_actions).sum(dim = 1)
        if n_action_dim - onehot_action_dim - multihot_action_dim != 0:
            continuous_actions_mean = onehot_action_probs_with_continuous_mean[..., onehot_action_dim + multihot_action_dim:]
            continuous_log_std = continuous_action_log_std.expand_as(continuous_actions_mean)
            continuous_actions_std = torch.exp(continuous_log_std)
            continuous_dist = MultivariateNormal(continuous_actions_mean, torch.diag_embed(continuous_actions_std))
            continuous_actions = continuous_dist.sample()
            continuous_actions_log_prob = continuous_dist.log_prob(continuous_actions)

        return onehot_actions, multihot_actions, continuous_actions, FloatTensor(
                onehot_actions_log_prob + multihot_actions_log_prob + continuous_actions_log_prob).unsqueeze(-1)

    def get_net_log_prob(self, net_input_state, net_input_onehot_action, net_input_multihot_action, net_input_continuous_action):
        net = getattr(self, net_name)
        n_action_dim = getattr(self, 'n_' + action_name)
        onehot_action_dim = getattr(self, 'onehot_' + action_name + '_dim')
        multihot_action_dim = getattr(self, 'multihot_' + action_name + '_dim')
        sections = getattr(self, 'onehot_' + action_name + '_sections')
        continuous_action_log_std = getattr(self, net_name + '_' + action_name + '_std')
        onehot_action_probs_with_continuous_mean = net(net_input_state)
        onehot_actions_log_prob = 0
        multihot_actions_log_prob = 0
        continuous_actions_log_prob = 0
        if onehot_action_dim != 0:
            dist = MultiOneHotCategorical(onehot_action_probs_with_continuous_mean[..., :onehot_action_dim],
                                          sections)
            onehot_actions_log_prob = dist.log_prob(net_input_onehot_action)
        if multihot_action_dim != 0:
            multihot_actions_prob = torch.sigmoid(onehot_action_probs_with_continuous_mean[..., onehot_action_dim:onehot_action_dim + multihot_action_dim])
            dist = torch.distributions.bernoulli.Bernoulli(probs = multihot_actions_prob) 
            multihot_actions_log_prob = dist.log_prob(net_input_multihot_action).sum(dim = 1)
        if n_action_dim - onehot_action_dim - multihot_action_dim != 0:
            continuous_actions_mean = onehot_action_probs_with_continuous_mean[..., onehot_action_dim + multihot_action_dim:]
            continuous_log_std = continuous_action_log_std.expand_as(continuous_actions_mean)
            continuous_actions_std = torch.exp(continuous_log_std)
            continuous_dist = MultivariateNormal(continuous_actions_mean, torch.diag_embed(continuous_actions_std))
            continuous_actions_log_prob = continuous_dist.log_prob(net_input_continuous_action)
        return FloatTensor(onehot_actions_log_prob + multihot_actions_log_prob + continuous_actions_log_prob).unsqueeze(-1)

    setattr(cls, 'get_' + net_name + '_' + action_name, get_net_action)
    setattr(cls, 'get_' + net_name + '_' + 'log_prob', get_net_log_prob)


__all__ = ['Policy', 'CustomSoftMax']
