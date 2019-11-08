import unittest
import torch
from model.policy import Policy, CustomSoftMax
from random import randint

from utils import ModelArgs, device


def tensor_close(t1: torch.Tensor, t2: torch.Tensor):
    if t1.shape == t2.shape:
        return torch.allclose(t1, t2)
    return False


class CustomSoftMaxTestCast(unittest.TestCase):
    def setUp(self) -> None:
        self.no_discrete_model = CustomSoftMax(0, [0])
        self.no_continuous_model = CustomSoftMax(20, [3, 4, 5, 8])

    def test_forward(self):
        t = torch.rand(randint(20, 200), self.no_continuous_model.discrete_action_dim, device=device)
        no_discrete_forward_tensor = self.no_discrete_model(t)
        no_continuous_forward_tensor = self.no_continuous_model(t)
        assert tensor_close(torch.tanh(t), no_discrete_forward_tensor)
        expect_no_continuous_forward_tensor = torch.cat(
            [tensor.softmax(dim=-1) for tensor in
             torch.split(t[..., :self.no_continuous_model.discrete_action_dim], self.no_continuous_model.sections,
                         dim=-1)],
            dim=-1)
        assert tensor_close(expect_no_continuous_forward_tensor, no_continuous_forward_tensor)


class PolicyTestCase(unittest.TestCase):
    def generate_random_sections(self, total_dim):
        # discard a random value
        if total_dim == 0:
            return [0]
        randint(2, 8)
        sections_len = randint(2, 8)
        sections = [2] * sections_len
        remain_dim = total_dim - sections_len * 2
        if remain_dim <= 0:
            return self.generate_random_sections(total_dim)
        for i in range(sections_len - 1):
            dim = randint(1, remain_dim)
            sections[i] += dim
            remain_dim -= dim
            if remain_dim <= 1:
                break
        sections[sections_len - 1] += (total_dim - sum(sections))
        assert sum(sections) == total_dim
        return sections

    def setUp(self) -> None:
        n_discrete_state = randint(20, 30)
        n_discrete_action = randint(20, 30)

        self.policy_discrete_state_sections = self.generate_random_sections(n_discrete_state)
        self.policy_discrete_action_sections = self.generate_random_sections(n_discrete_action)
        self.policy = Policy(self.policy_discrete_action_sections, self.policy_discrete_state_sections,
                             n_discrete_state=n_discrete_state, n_discrete_action=n_discrete_action,
                             n_continuous_action=1,
                             n_continuous_state=1)
        self.no_discrete_policy = Policy([0], [0], n_discrete_action=0, n_discrete_state=0, n_continuous_state=1,
                                         n_continuous_action=1)
        self.no_continuous_policy = Policy(self.policy_discrete_action_sections,
                                           self.policy_discrete_state_sections, n_continuous_action=0,
                                           n_continuous_state=0,
                                           n_discrete_state=n_discrete_state
                                           , n_discrete_action=n_discrete_action)

    def test_collect_time(self):
        import time
        start_time = time.time()
        memory = self.policy.collect_samples(2048)
        end_time = time.time()
        print('Total time %f' % (end_time - start_time))
        start_time = time.time()
        memory = self.policy.collect_samples(4, 512)
        end_time = time.time()
        print('Total time %f' % (end_time - start_time))

    def test_collect_samples_size(self):
        memory = self.policy.collect_samples(2048, 99)
        batch = memory.sample()
        discrete_state = torch.stack(batch.discrete_state).to(self.policy.device).squeeze(1).detach()
        continuous_state = torch.stack(batch.continuous_state).to(self.policy.device).squeeze(1).detach()
        next_discrete_state = torch.stack(batch.next_discrete_state).to(self.policy.device).squeeze(1).detach()
        continuous_action = torch.stack(batch.continuous_action).to(self.policy.device).squeeze(1).detach()
        next_continuous_state = torch.stack(batch.next_continuous_state).to(self.policy.device).squeeze(1).detach()
        discrete_action = torch.stack(batch.discrete_action).to(self.policy.device).squeeze(1).detach()

        old_log_prob = torch.stack(batch.old_log_prob).to(self.policy.device).squeeze(1).detach()
        mask = torch.stack(batch.mask).to(self.policy.device).squeeze(1).detach()

        policy_log_prob_new = self.policy.get_policy_net_log_prob(torch.cat((discrete_state, continuous_state), dim=-1),
                                                                  discrete_action, continuous_action)
        transition_log_prob_new = self.policy.get_transition_net_log_prob(
            torch.cat((discrete_state, continuous_state, discrete_action, continuous_action), dim=-1),
            next_discrete_state,
            next_continuous_state)
        new_log_prob = policy_log_prob_new + transition_log_prob_new
        # Due to we are not update policy gradient, old_log_prob and new_log_prob must be equal
        assert tensor_close(new_log_prob, old_log_prob)

    def test_log_prob(self):
        memory = self.policy.collect_samples(2048)
        batch = memory.sample()
        discrete_state = torch.stack(batch.discrete_state).to(self.policy.device).squeeze(1).detach()
        continuous_state = torch.stack(batch.continuous_state).to(self.policy.device).squeeze(1).detach()
        next_discrete_state = torch.stack(batch.next_discrete_state).to(self.policy.device).squeeze(1).detach()
        continuous_action = torch.stack(batch.continuous_action).to(self.policy.device).squeeze(1).detach()
        next_continuous_state = torch.stack(batch.next_continuous_state).to(self.policy.device).squeeze(1).detach()
        discrete_action = torch.stack(batch.discrete_action).to(self.policy.device).squeeze(1).detach()

        assert discrete_state.size(1) == sum(self.policy_discrete_state_sections)
        assert continuous_state.size(1) == 1
        assert next_discrete_state.size(1) == sum(self.policy_discrete_state_sections)
        assert continuous_action.size(1) == 1
        assert next_continuous_state.size(1) == 1
        assert discrete_action.size(1) == sum(self.policy_discrete_action_sections)

        old_log_prob = torch.stack(batch.old_log_prob).to(self.policy.device).squeeze(1).detach()
        mask = torch.stack(batch.mask).to(self.policy.device).squeeze(1).detach()
        # it should contain 'done'
        assert (mask == 0).any()

        policy_log_prob_new = self.policy.get_policy_net_log_prob(torch.cat((discrete_state, continuous_state), dim=-1),
                                                                  discrete_action, continuous_action)
        transition_log_prob_new = self.policy.get_transition_net_log_prob(
            torch.cat((discrete_state, continuous_state, discrete_action, continuous_action), dim=-1),
            next_discrete_state,
            next_continuous_state)
        new_log_prob = policy_log_prob_new + transition_log_prob_new
        # Due to we are not update policy gradient, old_log_prob and new_log_prob must be equal
        assert tensor_close(new_log_prob, old_log_prob)

    def test_no_discrete_log_prob(self):
        memory = self.no_discrete_policy.collect_samples(2048)
        batch = memory.sample()
        discrete_state = torch.stack(batch.discrete_state).to(self.no_discrete_policy.device).squeeze(1).detach()
        continuous_state = torch.stack(batch.continuous_state).to(self.no_discrete_policy.device).squeeze(1).detach()
        next_discrete_state = torch.stack(batch.next_discrete_state).to(
            self.no_discrete_policy.device).squeeze(1).detach()
        continuous_action = torch.stack(batch.continuous_action).to(self.no_discrete_policy.device).squeeze(1).detach()
        next_continuous_state = torch.stack(batch.next_continuous_state).to(
            self.no_discrete_policy.device).squeeze(1).detach()
        discrete_action = torch.stack(batch.discrete_action).to(self.no_discrete_policy.device).squeeze(1).detach()

        assert discrete_state.size(1) == sum(self.no_discrete_policy.discrete_state_sections)
        assert continuous_state.size(1) == 1
        assert next_discrete_state.size(1) == sum(self.no_discrete_policy.discrete_state_sections)
        assert continuous_action.size(1) == 1
        assert next_continuous_state.size(1) == 1
        assert discrete_action.size(1) == sum(self.no_discrete_policy.discrete_action_sections)

        old_log_prob = torch.stack(batch.old_log_prob).to(self.no_discrete_policy.device).squeeze(1).detach()
        mask = torch.stack(batch.mask).to(self.no_discrete_policy.device).squeeze(1).detach()
        # it should contain 'done'
        assert (mask == 0).any()

        policy_log_prob_new = self.no_discrete_policy.get_policy_net_log_prob(
            torch.cat((discrete_state, continuous_state), dim=-1),
            discrete_action, continuous_action)
        transition_log_prob_new = self.no_discrete_policy.get_transition_net_log_prob(
            torch.cat((discrete_state, continuous_state, discrete_action, continuous_action), dim=-1),
            next_discrete_state,
            next_continuous_state)
        new_log_prob = policy_log_prob_new + transition_log_prob_new
        # Due to we are not update policy gradient, old_log_prob and new_log_prob must be equal
        assert tensor_close(new_log_prob, old_log_prob)

    def test_no_continuous_log_prob(self):
        memory = self.no_continuous_policy.collect_samples(2048)
        batch = memory.sample()
        discrete_state = torch.stack(batch.discrete_state).to(self.no_continuous_policy.device).squeeze(1).detach()
        continuous_state = torch.stack(batch.continuous_state).to(self.no_continuous_policy.device).squeeze(1).detach()
        next_discrete_state = torch.stack(batch.next_discrete_state).to(
            self.no_continuous_policy.device).squeeze(1).detach()
        continuous_action = torch.stack(batch.continuous_action).to(
            self.no_continuous_policy.device).squeeze(1).detach()
        next_continuous_state = torch.stack(batch.next_continuous_state).to(
            self.no_continuous_policy.device).squeeze(1).detach()
        discrete_action = torch.stack(batch.discrete_action).to(self.no_continuous_policy.device).squeeze(1).detach()

        assert discrete_state.size(1) == sum(self.no_continuous_policy.discrete_state_sections)
        assert continuous_state.size(1) == 0
        assert next_discrete_state.size(1) == sum(self.no_continuous_policy.discrete_state_sections)
        assert continuous_action.size(1) == 0
        assert next_continuous_state.size(1) == 0
        assert discrete_action.size(1) == sum(self.no_continuous_policy.discrete_action_sections)

        old_log_prob = torch.stack(batch.old_log_prob).to(self.no_continuous_policy.device).squeeze(1).detach()
        mask = torch.stack(batch.mask).to(self.no_continuous_policy.device).squeeze(1).detach()
        # it should contain 'done'
        assert (mask == 0).any()

        policy_log_prob_new = self.no_continuous_policy.get_policy_net_log_prob(
            torch.cat((discrete_state, continuous_state), dim=-1),
            discrete_action, continuous_action)
        transition_log_prob_new = self.no_continuous_policy.get_transition_net_log_prob(
            torch.cat((discrete_state, continuous_state, discrete_action, continuous_action), dim=-1),
            next_discrete_state,
            next_continuous_state)
        new_log_prob = policy_log_prob_new + transition_log_prob_new
        # Due to we are not update policy gradient, old_log_prob and new_log_prob must be equal
        assert tensor_close(new_log_prob, old_log_prob)


if __name__ == '__main__':
    unittest.main()
