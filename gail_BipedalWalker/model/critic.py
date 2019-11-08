import torch.nn as nn
from utils import _init_weight, device, ModelArgs

args = ModelArgs()


class Value(nn.Module):
    def __init__(self, device=device):
        super(Value, self).__init__()
        self.n_state = args.n_discrete_state + args.n_continuous_state
        self.value_net = nn.Sequential(
            nn.Linear(self.n_state, args.n_value_hidden),
            nn.LeakyReLU(),
            nn.Linear(args.n_value_hidden, args.n_value_hidden),
            nn.LeakyReLU(),
            nn.Linear(args.n_value_hidden, args.n_value_hidden),
            nn.Linear(args.n_value_hidden, 1)
        ).to(device)
        self.value_net.apply(_init_weight)

    def forward(self, state):
        values = self.value_net(state.to(device))
        return values.to(device)
