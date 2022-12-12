import random


import torch 
from torch.utils.data import Dataset, DataLoader 
from torch_geometric.data import Data, Batch
import numpy as np


def preprocessed_split_data(time_steps, test_ratio=0.15, val_ratio=0.15):
    print('Loading dataset...')
    DATA = np.load('./data/dataset.npz', allow_pickle=True)['data']
    indices = list(range(len(DATA)))
    random.shuffle(indices)
    n_test = int(np.ceil(len(DATA) * test_ratio))
    n_val = int(np.ceil(len(DATA) * val_ratio))

    indices_test = indices[:n_test]
    indices_val = indices[n_test: n_test + n_val]
    indices_train = indices[n_test + n_val: ]

    for d in DATA:
        d['vector'] = d['vector'][:, : (time_steps + 1)]
        d['scalar'] = d['scalar'][:, : (time_steps + 1)]
        d['wdfp'] = d['wdfp'][:, : (time_steps + 1)][...,None]

    valid_list = [DATA[i] for i in indices_val]
    test_list = [DATA[i] for i in indices_test]

    train_list = []
    scalar_list = []
    scalar_fixed_list = []
    wdfp_list = []
    
    for i in indices_train:
        sample = DATA[i]
        train_list.append(sample)
        scalar_fixed_list.append(sample['scalar_fixed'])
        s = sample['scalar']
        wdfp = sample['wdfp']
        scalar_list.append(s.reshape(s.shape[0], s.shape[1], -1))
        wdfp_list.append(wdfp.reshape(wdfp.shape[0], wdfp.shape[1], -1))
    
    scalar_list = np.concatenate(scalar_list)
    scalar_fixed_list = np.concatenate(scalar_fixed_list)
    wdfp_list = np.concatenate(wdfp_list)

    s_mean, s_std = np.mean(scalar_list), np.std(scalar_list)
    s_f_mean, s_f_std = np.mean(scalar_fixed_list), np.std(scalar_fixed_list)
    w_mean, w_std = np.mean(wdfp_list), np.std(wdfp_list)
    
    print('Loaded!')
    return (
        FloodDatatset(train_list, scalar_fixed_stats=(s_f_mean, s_f_std), scalar_stats=(s_mean, s_std), wdfp_stats=(w_mean, w_std), time_steps=time_steps),
        FloodDatatset(valid_list, scalar_fixed_stats=(s_f_mean, s_f_std), scalar_stats=(s_mean, s_std), wdfp_stats=(w_mean, w_std), time_steps=time_steps),
        FloodDatatset(test_list, scalar_fixed_stats=(s_f_mean, s_f_std), scalar_stats=(s_mean, s_std), wdfp_stats=(w_mean, w_std), time_steps=time_steps),
        torch.tensor(indices_test),
    )


class FloodDatatset(Dataset):
    def __init__(self, data_list, scalar_fixed_stats, scalar_stats, wdfp_stats, time_steps):
        self.data_list = data_list 
        for g in data_list:
            g['scalar_fixed'] = torch.tensor((g['scalar_fixed'] - scalar_fixed_stats[0]) / (scalar_fixed_stats[1] + 1e-7)).float()
            g['scalar'] = torch.tensor((g['scalar'] - scalar_stats[0]) / (scalar_stats[1] + 1e-7)).float()
            g['wdfp'] = torch.tensor((g['wdfp'] - wdfp_stats[0]) / (wdfp_stats[1] + 1e-7)).float() 

            g['edges'] = torch.tensor(g['edges'].T).long()
            g['vector'] = torch.tensor(g['vector']).float()

        self.scalar_fixed_stats = tuple(torch.tensor(s) for s in scalar_fixed_stats)
        self.scalar_stats = tuple(torch.tensor(s) for s in scalar_stats)
        self.wdfp_stats = tuple(torch.tensor(s) for s in wdfp_stats)
        self.time_steps = time_steps

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index]
    
    def collate(self, batch):
        new_batch = []
        for i, g in enumerate(batch):
            new_batch.append(
                Data(edge_index=g['edges'], s_fixed=g['scalar_fixed'], 
                    s=g['scalar'], v=g['vector'], wdfp=g['wdfp'], 
                    num_nodes=g['scalar'].shape[0])
                    )
        new_batch = Batch.from_data_list(new_batch, follow_batch=['s'])

        return new_batch


class FloodDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, collate_fn=dataset.collate, **kwargs)
