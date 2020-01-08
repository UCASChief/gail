import torch
from utils import time_this
from utils import ModelArgs, device, FloatTensor
args = ModelArgs()

def GAE(reward, value, mask, gamma, lam):
    # adv = FloatTensor(reward.shape, device=device)
    # delta = FloatTensor(reward.shape, device=device)

    # # pre_value, pre_adv = 0, 0
    # pre_value = torch.zeros(reward.shape[1:], device=device)
    # pre_adv = torch.zeros(reward.shape[1:], device=device)
    # for i in reversed(range(reward.shape[0])):
    #     delta[i] = reward[i] + gamma * pre_value * mask[i] - value[i]
    #     adv[i] = delta[i] + gamma * lam * pre_adv * mask[i]
    #     pre_adv = adv[i, ...]
    #     pre_value = value[i, ...]
    # returns = value + adv
    # adv = (adv - adv.mean()) / adv.std()

    reward = reward.reshape(-1, args.sample_traj_length, 1)
    value = value.reshape(-1, args.sample_traj_length, 1)
    mask = mask.reshape(-1, args.sample_traj_length, 1)

    adv = FloatTensor(reward.shape, device=device)
    delta = FloatTensor(reward.shape, device=device)

    # pre_value, pre_adv = 0, 0
    pre_value = torch.zeros((reward.shape[0], 1), device=device)
    pre_adv = torch.zeros((reward.shape[0], 1), device=device)
    for i in reversed(range(reward.shape[1])):
        delta[:, i] = reward[:, i] + gamma * pre_value * mask[:, i] - value[:, i]
        adv[:, i] = delta[:, i] + gamma * lam * pre_adv * mask[:, i]
        pre_adv = adv[:, i, ...]
        pre_value = value[:, i, ...]
    returns = value + adv
    adv = (adv - adv.mean()) / adv.std()
    returns = returns.reshape(-1, 1)
    adv = adv.reshape(-1, 1)

    return adv, returns


def ppo_step(policy, value, optimizer_policy, optimizer_value, onehot_state, multihot_state, continuous_state, onehot_action,
             multihot_action, continuous_action, next_onehot_state, next_multihot_state, next_continuous_state, reward,
             fixed_log_probs, done, ppo_clip_epsilon):
    # update critic
    states = torch.cat((onehot_state, multihot_state, continuous_state), dim=-1)
    values_pred = value(states)
    advantages, returns = GAE(reward.detach(), values_pred.detach(), done, gamma=args.gamma, lam=args.lam)
    value_loss = (values_pred - returns).pow(2).mean()
    for param in value.parameters():
        value_loss += param.pow(2).sum() * 1e-3
    optimizer_value.zero_grad()
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(value.parameters(), 40)
    optimizer_value.step()

    # update actor
    policy_log_probs = policy.get_policy_net_log_prob(states,
                                                      onehot_action,
                                                      multihot_action,
                                                      continuous_action)
    transition_log_prob_new = policy.get_transition_net_log_prob(
        torch.cat((onehot_state, multihot_state, continuous_state, onehot_action, multihot_action, continuous_action), dim=-1), next_onehot_state, next_multihot_state, next_continuous_state)
    log_probs = policy_log_probs + transition_log_prob_new
    ratio = torch.exp(log_probs - fixed_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - ppo_clip_epsilon, 1.0 + ppo_clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 40)
    optimizer_policy.step()
    return value_loss, policy_loss
