from functools import reduce

import torch
from torch.distributions import Categorical, Distribution


class MultiCategorical(Distribution):

    def __init__(self, probs_vector: torch.Tensor, sections: list):
        """
        :param probs_vector: a vector contains multi categorical probs
        :param sections: probs sections, type: list of int

        """
        probs_tuple = torch.split(probs_vector, sections, dim=-1)
        self._categoricals = list(map(Categorical, probs_tuple))
        self.sections = sections

    def log_prob(self, value):
        value_samples = torch.split(value, 1, dim=-1)
        assert len(value_samples) == len(self.sections)
        log_probs = [p.log_prob(px.squeeze()) for p, px in zip(self._categoricals, value_samples)]
        return reduce(torch.add, log_probs)

    def entropy(self):
        return reduce(torch.add, [dist.entropy() for dist in self._categoricals])

    def sample(self, sample_shape=torch.Size()):
        out = torch.cat([dist.sample().unsqueeze(-1) for dist in self._categoricals], dim=-1)
        assert out.size(-1) == len(self.sections)
        return out
