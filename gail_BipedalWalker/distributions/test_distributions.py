import unittest
import torch
from distributions.MultiCategorical import MultiCategorical
from distributions.MultiOneHotCategorical import MultiOneHotCategorical
from torch.distributions import Categorical, OneHotCategorical


# noinspection PyArgumentList
class MultiCategoricalBatchTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.test_prob_tensor = torch.Tensor([[0.1, 0.2, 0.7, 0.6, 0.4],
                                              [0.2, 0.6, 0.2, 0.5, 0.5]])
        self.test_sections = [3, 2]
        self.multi_dist = MultiCategorical(self.test_prob_tensor, self.test_sections)
        self.dist_a = Categorical(self.test_prob_tensor[..., :3])
        self.dist_b = Categorical(self.test_prob_tensor[..., 3:])
        self.test_sample = torch.Tensor([[2, 0], [1, 1]]).long()
        self.test_samples = torch.split(self.test_sample, 1, dim=-1)

    def test_log_prob(self):
        log_prob_sum_1 = self.dist_a.log_prob(self.test_samples[0].squeeze())  # [-0.3567,-0.5108]
        log_prob_sum_2 = self.dist_b.log_prob(self.test_samples[1].squeeze())  # [-0.5108,-0.6931]
        multi_log_prob_correct = log_prob_sum_1 + log_prob_sum_2
        multi_log_prob = self.multi_dist.log_prob(self.test_sample)
        assert multi_log_prob.shape == torch.Size([2])
        assert multi_log_prob_correct.shape == torch.Size([2])
        assert torch.allclose(multi_log_prob, multi_log_prob_correct)

    def test_entropy(self):
        ea = self.dist_a.entropy()
        eb = self.dist_b.entropy()
        assert torch.allclose(ea + eb, self.multi_dist.entropy())

    def test_sample(self):
        sample = self.multi_dist.sample()
        assert sample.size(0) == 2
        assert sample.size(1) == 2


class MultiOneHotCategoricalBatchTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.test_prob_tensor = torch.Tensor([[0.1, 0.2, 0.7, 0.6, 0.4],
                                              [0.2, 0.6, 0.2, 0.5, 0.5]])
        self.test_sections = [3, 2]
        self.multi_dist = MultiOneHotCategorical(self.test_prob_tensor, self.test_sections)
        self.dist_a = OneHotCategorical(self.test_prob_tensor[..., :3])
        self.dist_b = OneHotCategorical(self.test_prob_tensor[..., 3:])
        self.test_sample = torch.Tensor([[0., 0., 1., 1., 0.], [0., 1., 0., 0., 1.]]).long()
        self.test_samples = torch.split(self.test_sample, self.test_sections, dim=-1)

    def test_log_prob(self):
        log_prob_sum_1 = self.dist_a.log_prob(self.test_samples[0])  # [-0.3567,-0.5108]
        log_prob_sum_2 = self.dist_b.log_prob(self.test_samples[1])  # [-0.5108,-0.6931]
        multi_log_prob_correct = log_prob_sum_1 + log_prob_sum_2
        multi_log_prob = self.multi_dist.log_prob(self.test_sample)
        assert multi_log_prob.shape == torch.Size([2])
        assert multi_log_prob_correct.shape == torch.Size([2])
        assert torch.allclose(multi_log_prob, multi_log_prob_correct)

    def test_entropy(self):
        ea = self.dist_a.entropy()
        eb = self.dist_b.entropy()
        assert torch.allclose(ea + eb, self.multi_dist.entropy())

    def test_sample(self):
        sample = self.multi_dist.sample()
        assert sample.size(0) == 2
        assert sample.size(1) == 5


if __name__ == '__main__':
    unittest.main()
