## dataloader for NTU-RGBD

import torch
import torch.utils.data as data
import numpy as np
import os
import math
import pickle
import ref

class NTURGBD(data.Dataset):
    def __init__(self, split='train', benchmark='sub', seed_seq_len=40, pred_seq_len=10):
        print('==> initializing NTURGBD {} data.'.format(split))

        data_path = ref.ntuDataDir + '/x{}/{}_data.npy'.format(benchmark, split)
        label_path = ref.ntuDataDir + '/x{}/{}_label.pkl'.format(benchmark, split)
        times = ref.ntuDataDir + '/x{}/end_times_{}.npy'.format(benchmark, split)

        with open(label_path, 'rb') as f:
            labels = pickle.load(f)

        self.label = labels[1]
        data = np.load(data_path)    # shape : [N, 3, 300, 25, 2]
        self.data = data.transpose((0,4,2,3,1))
        self.end_times = np.load(times)

        self.seed_seq_len = seed_seq_len
        self.pred_seq_len = pred_seq_len

        self.split = split
        self.benchmark = benchmark
        self.max_length = 300

        self.n_samples = self.data.shape[0]
        print('Loaded {} with {} labelled samples'.format(split, self.n_samples))


    def pad_poses(self, poses):
        ''' Please Fill in the code for padding the input to get a fixed
            length sequence
        '''
        #if poses.shape[0] >= self.max_length:
        return poses[ :self.max_length]

    def __getitem__(self, index):
        #frame_mat = self.annot[index]

        poses_3d = self.data[index, 0]  #frame_mat['skel_body0']  poses_3d.shape = [300, 25, 3]
        poses_3d = poses_3d - poses_3d[:, :1, :]   # making pose root relative
        label = self.label[index]      #frame_mat['label']
        padded_poses = poses_3d        #self.pad_poses(poses_3d)

        end_time = self.end_times[index]
        total_len = self.seed_seq_len #+ self.pred_seq_len

        #mask = np.ones((total_len), dtype=float)

        if end_time > total_len:
          idx = np.random.randint(0, end_time-total_len)
          seq = poses_3d[ idx: idx + total_len]
          seed_seq = seq[ : self.seed_seq_len]
          target_seq = seq[ self.seed_seq_len:]
        else:
          idx = 0
          #mask[int(end_time):] = 0
          seq = poses_3d[idx : idx+total_len]
          seed_seq = seq[ : self.seed_seq_len]
          target_seq = seq[self.seed_seq_len : ]
        #mask = mask[self.seed_seq_len : ]

        return 1, seed_seq, 1  #, target_seq, label  #, mask

    def __len__(self):
        return self.n_samples

if __name__ == "__main__":
    ntu  = NTURGBD(split='val')
    dataloader = torch.utils.data.DataLoader(ntu, batch_size=5, shuffle=True)
    for i, (seed_seq, target_seq, label) in enumerate(dataloader):
        print(seed_seq.shape)
        #print(seed_seq)
        print(target_seq.shape)
        #print(mask.shape)
        #print(labels[0])
        print('#'*10)
        print('#'*10)
        if i == 10:
            break

