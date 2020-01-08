from functools import reduce

import torch
from torch.distributions import OneHotCategorical, Distribution


class MultiOneHotCategorical(Distribution):

    def __init__(self, probs_vector: torch.Tensor, sections: list):
        """
        :param probs_vector: a vector contains multi categorical probs
        :param sections: probs sections, type: list of int

        """
        probs_tuple = torch.split(probs_vector, sections, dim=-1)
        self._categoricals = list(map(OneHotCategorical, probs_tuple))
        self.sections = sections

    def log_prob(self, value):
        value_samples = torch.split(value, self.sections, dim=-1)
        log_probs = [p.log_prob(px.squeeze()) for p, px in zip(self._categoricals, value_samples)]
        return reduce(torch.add, log_probs)

    def entropy(self):
        return reduce(torch.add, [dist.entropy() for dist in self._categoricals])

    def sample(self, sample_shape=torch.Size()):
        out = torch.cat([dist.sample() for dist in self._categoricals], dim=-1)
        return out
