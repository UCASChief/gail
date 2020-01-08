import torch
import numpy as np
import pandas as pd
from utils import ModelArgs

args = ModelArgs()
n_id_cols = 6
data = np.array(pd.read_csv('./trajectories_scaled.csv'))

ids, data = data[:, :n_id_cols], data[:, n_id_cols:]
data_len = data.shape[0]
lst_adset_id = None
for i in range(data_len):
    cur_line = ids[i]
    if lst_adset_id is None or cur_line[1] != lst_adset_id:
        lst_adset_id = cur_line[1]
        if i > data_len * 4 // 5:
            target = i
            break

print(target)
data = data.astype(np.float64)
print(data.dtype)

train_data = torch.FloatTensor(data[:target, :])
test_data = torch.FloatTensor(data[target:, :])
print(train_data[:20, 19:19 + 110])
exit(0)
print(train_data.shape)
print(test_data.shape)
with open('./train_data_scaled.csv', 'w') as file:
    for line in train_data:
        string = ','.join([str(x.item()) for x in line]) + '\n'
        file.write(string)    
with open('./test_data_scaled.csv', 'w') as file:
    for line in test_data:
        string = ','.join([str(x.item()) for x in line]) + '\n'
        file.write(string)
 
