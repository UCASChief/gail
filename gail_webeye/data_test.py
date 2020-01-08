import torch
import numpy as np
import pandas as pd
from utils import ModelArgs

args = ModelArgs()
n_id_cols = 6
data = np.array(pd.read_csv('./trajectories_scaled.csv'))
header = [
    '180days',
    'used momo',
    'qinggan',
    'douzheng',
    'chuanyue',
    'dushi',
    'diaosi',
    'shijia',
    'yineng',
    'haomen',
    '60days30min',
    'nan',
    'vip',
    'leisi',
    '30days',
    'open180',
]
cols = [
    [35,37,39,52,56,58,63],
    [36,38,51,57,59,61,62],
    [40,46,64,65],
    [41],
    [42,67],
    [43,68,72],
    [44,69],
    [66],
    [45,70],
    [47,71],
    [50],
    [75],
    [77],
    [60],
    [55],
    [76],
]
cols = [torch.LongTensor(x) for x in cols]

ids, data = data[:, :n_id_cols], data[:, n_id_cols:]
data = data.astype(np.float64)

action_offset = args.n_state
next_state_offset = action_offset + args.n_action

data = torch.FloatTensor(data)

print('state onehot {}, multi {}, continuous {}.'.format(args.n_onehot_state, len(header) + 10, args.n_continuous_state))
new_data = torch.FloatTensor(data[:, :args.n_onehot_state])
for i in range(len(header)):
    cnt = data[:, cols[i]]
    new_data = torch.cat((new_data, cnt.sum(dim = -1).reshape(-1, 1)), dim = 1)

new_data = torch.cat((new_data, data[:, action_offset - args.n_continuous_state - 10:action_offset]), dim = 1)
print(new_data.shape)

print('action onehot {}, multi {}, continuous {}.'.format(args.n_onehot_action, len(header) + 10, args.n_continuous_action))
new_data = torch.cat((new_data, data[:, action_offset:action_offset + args.n_onehot_action]), dim = 1)
for i in range(len(header)):
    cnt = data[:, cols[i] + action_offset]
    new_data = torch.cat((new_data, cnt.sum(dim = -1).reshape(-1, 1)), dim = 1)
new_data = torch.cat((new_data, data[:, next_state_offset - args.n_continuous_action - 10:next_state_offset]), dim = 1)
print(new_data.shape)

new_data = torch.cat((new_data, data[:, next_state_offset:next_state_offset + args.n_onehot_state]), dim = 1)
for i in range(len(header)):
    cnt = data[:, cols[i] + next_state_offset]
    new_data = torch.cat((new_data, cnt.sum(dim = -1).reshape(-1, 1)), dim = 1)
new_data = torch.cat((new_data, data[:, -args.n_continuous_state - 10:]), dim = 1)
print(new_data.shape)

data_len = new_data.shape[0]
lst_adset_id = None
for i in range(data_len):
    cur_line = ids[i]
    if lst_adset_id is None or cur_line[1] != lst_adset_id:
        lst_adset_id = cur_line[1]
        if i > data_len * 4 // 5:
            target = i
            break

print(target)

train_data = torch.FloatTensor(new_data[:target, :])
test_data = torch.FloatTensor(new_data[target:, :])
print(train_data.shape)
print(test_data.shape)
with open('./train_data_scaled_new_v2.csv', 'w') as file:
    for line in train_data:
        string = ','.join([str(x.item()) for x in line]) + '\n'
        file.write(string)    
with open('./test_data_scaled_new_v2.csv', 'w') as file:
    for line in test_data:
        string = ','.join([str(x.item()) for x in line]) + '\n'
        file.write(string)
 
