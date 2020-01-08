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

    reward = reward.reshape(-1, args.sample_traj_length, 4)
    value = value.reshape(-1, args.sample_traj_length, 4)
    mask = mask.reshape(-1, args.sample_traj_length, 4)

    adv = FloatTensor(reward.shape, device=device)
    delta = FloatTensor(reward.shape, device=device)

    # pre_value, pre_adv = 0, 0
    pre_value = torch.zeros((reward.shape[0], 4), device=device)
    pre_adv = torch.zeros((reward.shape[0], 4), device=device)
    for i in reversed(range(reward.shape[1])):
        delta[:, i] = reward[:, i] + gamma * pre_value * mask[:, i] - value[:, i]
        adv[:, i] = delta[:, i] + gamma * lam * pre_adv * mask[:, i]
        pre_adv = adv[:, i, ...]
        pre_value = value[:, i, ...]
    returns = value + adv
    adv = (adv - adv.mean()) / adv.std()
    returns = returns.reshape(-1, 4)
    adv = adv.reshape(-1, 4)

    return adv, returns


def ppo_step(policy, state, action, next_state, reward, fixed_log_probs, done, ppo_clip_epsilon):
    # update critic
    states = [state, state, state, torch.cat((state, action), dim = -1)]
    values_pred = policy.get_value(states)
    advantages, returns = GAE(reward.detach(), values_pred.detach(), done.expand(reward.shape), gamma=args.gamma, lam=args.lam)
    value_loss = (values_pred - returns).pow(2).mean()
    for item in policy.v_params:
        for param in item['params']:
            value_loss += param.pow(2).sum() * 1e-3
    policy.v_optimizer.zero_grad()
    value_loss.backward()
    for item in policy.v_params:
        torch.nn.utils.clip_grad_norm_(item['params'], 40)
    policy.v_optimizer.step()

    # update actor
    log_prob1, log_prob2, log_prob3 = policy.get_user_action_log_prob(state, action)
    log_prob4 = policy.get_next_state_log_prob(state, action, next_state)
    log_probs = torch.cat((log_prob1, log_prob2, log_prob3, log_prob4), dim = -1)
    ratio = torch.exp(log_probs - fixed_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - ppo_clip_epsilon, 1.0 + ppo_clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean(dim=0).sum()
    policy.p_optimizer.zero_grad()
    policy_loss.backward()
    for item in policy.p_params:
        torch.nn.utils.clip_grad_norm_(item['params'], 40)
    policy.p_optimizer.step()
    return value_loss, policy_loss
