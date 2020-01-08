import torch
import torch.cuda
import torch.nn as nn
from torch.distributions import MultivariateNormal

from distributions.MultiOneHotCategorical import MultiOneHotCategorical
from utils import (_init_weight, FloatTensor, SimpleMemory, device, ModelArgs)

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


class MultiPolicy(nn.Module):
    def forward(self, *input):
        raise NotImplementedError

    def __init__(self, state_sizes, action_sizes, onehot_action_sections: list, onehot_state_sections: list, action_log_std=0,
                 state_log_std=0, max_traj_length=args.sample_traj_length, n_continuous_action=args.n_continuous_action,
                 n_onehot_action=args.n_onehot_action, n_onehot_state=args.n_onehot_state,
                 n_continuous_state=args.n_continuous_state, device=device, state_0=None):
        super(MultiPolicy, self).__init__()
        assert len(state_sizes) == len(action_sizes)
        self.n_policies = len(state_sizes)
        self.state_sizes = state_sizes
        self.action_sizes = action_sizes
        self.onehot_state_sections = onehot_state_sections
        self.onehot_action_sections = onehot_action_sections
        self.device = device
        if state_0 is not None:
            self.state_0 = state_0.to(device)
        else:
            self.state_0 = None

        self.hidden_arr = []
        self.policy_arr = []
        self.value_arr = []
        self.optimizer_arr = []
        self.p_params = []
        self.v_params = []
        for i in range(self.n_policies):
            self.hidden_arr.append(self.__create_network(self.state_sizes[i]))
            self.policy_arr.append(nn.Linear(args.n_policy_hidden, self.action_sizes[i]).to(device))
            self.policy_arr[-1].apply(_init_weight)
            self.value_arr.append(nn.Linear(args.n_policy_hidden, 1).to(device))
            self.value_arr[-1].apply(_init_weight)
            self.optimizer_arr.append(torch.optim.Adam(self.policy_arr[i].parameters(), lr = args.policy_lr))
            self.p_params.append({'params':self.hidden_arr[-1].parameters()})
            self.v_params.append({'params':self.hidden_arr[-1].parameters()})
            self.p_params.append({'params':self.policy_arr[-1].parameters()})
            self.v_params.append({'params':self.value_arr[-1].parameters()})
        self.action_softmax = CustomSoftMax(args.n_onehot_action, onehot_action_sections)
        self.state_softmax = CustomSoftMax(args.n_onehot_state, onehot_state_sections)
        self.v_optimizer = torch.optim.Adam(self.v_params, lr = args.value_lr)
        self.p_optimizer = torch.optim.Adam(self.p_params, lr = args.policy_lr)

        self.max_traj_length = max_traj_length
        self.action_std = nn.Parameter(torch.ones(1, n_continuous_action, device=device) * action_log_std)
        self.next_state_std = nn.Parameter(torch.ones(1, n_continuous_state - n_continuous_action, device=device) * state_log_std)

    def __create_network(self, state_dim):
        net = nn.Sequential(
            nn.Linear(state_dim, args.n_policy_hidden),
            nn.LeakyReLU(),
            nn.Linear(args.n_policy_hidden, args.n_policy_hidden),
            nn.LeakyReLU(),
        ).to(self.device)
        net.apply(_init_weight)
        return net

    def collect_samples(self, batch_size=1):
        memory = SimpleMemory()
        num_trajs = (batch_size + args.sample_traj_length - 1) // args.sample_traj_length
        state = self.reset(num_trajs)
        for walk_step in range(self.max_traj_length):
            with torch.no_grad():
                action, next_state, old_log_prob = self.step(state, num_trajs)
            if walk_step == self.max_traj_length - 1:
                mask = torch.zeros((num_trajs, 1), device = device)
            else:
                mask = torch.ones((num_trajs, 1), device=device)
            memory.push(state.type(FloatTensor), action.type(FloatTensor), next_state.type(FloatTensor), old_log_prob, mask)
            state = next_state
        return memory, num_trajs

    def step(self, state, num_trajs=1):
        action, action_log_prob = self.get_user_action(state)
        next_state, next_state_log_prob = self.get_next_state(state, action)
        if args.concat_action == 1:
            next_state = torch.cat((action, next_state[-17:]), dim = -1)
        return action, next_state, torch.cat((action_log_prob, next_state_log_prob), dim = -1)

    def reset(self, num_trajs=1):
        if self.state_0 is None:
            assert False
        else:
            state = self.state_0[torch.randint(self.state_0.shape[0], size=(num_trajs,))]
        return state
    
    def get_user_action(self, state):
        hidden = self.hidden_arr[0](state)
        action1_dist = MultiOneHotCategorical(self.action_softmax(self.policy_arr[0](hidden)), self.onehot_action_sections)
        action1 = action1_dist.sample()
        action1_log_prob = action1_dist.log_prob(action1).reshape(-1, 1)

        hidden = self.hidden_arr[1](state)
        action2_prob = torch.sigmoid(self.policy_arr[1](hidden))
        action2_dist = torch.distributions.bernoulli.Bernoulli(probs = action2_prob)
        action2 = action2_dist.sample()
        action2_log_prob = action2_dist.log_prob(action2).sum(dim = -1).reshape(-1, 1)
        
        hidden = self.hidden_arr[2](state)
        action3_mean = self.policy_arr[2](hidden)
        action3_log_std = self.action_std.expand_as(action3_mean)
        action3_std = torch.exp(action3_log_std)
        action3_dist = MultivariateNormal(action3_mean, torch.diag_embed(action3_std))
        action3 = action3_dist.sample()
        action3_log_prob = action3_dist.log_prob(action3).reshape(-1, 1)
        return torch.cat((action1, action2, action3), dim = -1), torch.cat((action1_log_prob, action2_log_prob, action3_log_prob), dim = -1)

    def get_next_state(self, state, action):
        hidden = self.hidden_arr[3](torch.cat((state, action), dim = -1))
        next_state_mean = self.policy_arr[3](hidden)
        next_state_log_std = self.next_state_std.expand_as(next_state_mean)
        next_state_std = torch.exp(next_state_log_std)
        next_state_dist = MultivariateNormal(next_state_mean, torch.diag_embed(next_state_std))
        next_state = next_state_dist.sample()
        next_state_log_prob = next_state_dist.log_prob(next_state).reshape(-1, 1)

        continuous_state = state[..., -17 - 6:-17] + action[..., -6:]
        return torch.cat((action[..., :-6], continuous_state, next_state), dim = -1), next_state_log_prob

    def get_user_action_log_prob(self, state, action):
        hidden = self.hidden_arr[0](state)
        action1_dist = MultiOneHotCategorical(self.action_softmax(self.policy_arr[0](hidden)), self.onehot_action_sections)
        action1_log_prob = action1_dist.log_prob(action[:, :args.n_onehot_action])

        hudden = self.hidden_arr[1](state)
        action2_prob = torch.sigmoid(self.policy_arr[1](hidden))
        action2_dist = torch.distributions.bernoulli.Bernoulli(probs = action2_prob)
        action2_log_prob = action2_dist.log_prob(action[:, args.n_onehot_action:-args.n_continuous_action]).sum(dim = -1)
        
        hidden = self.hidden_arr[2](state)
        action3_mean = self.policy_arr[2](hidden)
        action3_log_std = self.action_std.expand_as(action3_mean)
        action3_std = torch.exp(action3_log_std)
        action3_dist = MultivariateNormal(action3_mean, torch.diag_embed(action3_std))
        action3_log_prob = action3_dist.log_prob(action[:, -args.n_continuous_action:])

        return action1_log_prob.reshape(-1,1), action2_log_prob.reshape(-1,1), action3_log_prob.reshape(-1,1)

    def get_next_state_log_prob(self, state, action, next_state):
        next_state = next_state[:, -17:]
        hidden = self.hidden_arr[3](torch.cat((state, action), dim = -1))
        next_state_mean = self.policy_arr[3](hidden)
        next_state_log_std = self.next_state_std.expand_as(next_state_mean)
        next_state_std = torch.exp(next_state_log_std)
        next_state_dist = MultivariateNormal(next_state_mean, torch.diag_embed(next_state_std))
        next_state_log_prob = next_state_dist.log_prob(next_state)

        continuous_state = state[..., -17 - 6:-17] + action[..., -6:]
        return next_state_log_prob.reshape(-1,1)

    def get_value(self, state):
        value = []
        for i in range(self.n_policies):
            hidden = self.hidden_arr[i](state[i])
            value.append(self.value_arr[i](hidden))
        return torch.cat(value, dim = -1)

__all__ = ['Policy', 'CustomSoftMax']
