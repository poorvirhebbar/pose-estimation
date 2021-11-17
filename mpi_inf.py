## dataloader for h3.6m

import random
import cv2
import torch
import torch.utils.data as data
import numpy as np
import os
import math
import pickle
from img_utils import Crop
from viz import viz_img

class MPI_INF(data.Dataset):
    def __init__(self, data_path, split='train', max_length=100):
        print('==> initializing MPI-INF {} data.'.format(split))

        if split == 'train':
            file_path = os.path.join(
                                data_path,
                                'image_embeds_train_mpi.pkl')
                                #'mpi_inf_videowise_annot.pkl')
        else:
            file_path = os.path.join(
                                data_path,
                                'image_embeds_val_mpi.pkl')
                                #'mpi_inf_videowise_annot_val.pkl')

        with open(file_path, 'rb') as f:
            annot = pickle.load(f)

        self.data_path = data_path
        self.annot = annot
        #import pdb; pdb.set_trace()
        self.split = split
        self.max_length = max_length

        self.n_samples = len(self.annot.keys())
        print('Loaded {} with {} labelled samples'.format(split, self.n_samples))

        self.img_mean = np.array([0.485, 0.456, 0.406])
        self.std_dev = np.array([0.229, 0.224, 0.225])

    def _kps_to_h36m(self, poses):
        '''
            poses: n x n_joints x 2|3
        '''
        joint_idx = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6];
        poses = poses[:, joint_idx, :]
        return poses

    def pad_poses(self, poses):
        ''' Please Fill in the code for padding the input to get a fixed
            length sequence
        '''
        pad_vec = np.zeros((self.max_length, 1))
        if poses.shape[0] >= self.max_length:
            return poses[ :self.max_length], pad_vec
        else:
            padded_poses = np.zeros((self.max_length, poses.shape[1], 3))
            padded_poses[:poses.shape[0]] = poses
            pad_vec[:poses.shape[0]] = 1.0
            return padded_poses, pad_vec

    def get_chunk(self, poses, init_frame=None, max_len=100):
        '''
            poses: n x n_joints x 2 | 3
        '''
        if max_len == -1:
            return poses, 0

        n_frames = poses.shape[0]
        if init_frame is None:
            init_frame = random.randint(0, n_frames - self.max_length )
        chunk = poses[init_frame: init_frame + self.max_length ]
        return chunk, init_frame

    def get_images(self, index, init_frame, poses_2d):
        imgs = []
        for i in range(init_frame, init_frame + self.max_length):
            sub = self.annot['sub'][index][i]
            seq = self.annot['seq'][index][i]
            cam = self.annot['cam'][index][i]
            img_path = os.path.join(self.data_path, f'S{sub}', f'Seq{seq}',
                                    'images', f'{cam}', f'img_{i}.png')
            img = cv2.imread(img_path)
            if img is None:
                print('** ' + img_path + ' **')
                img = np.zeros([224, 224, 3], dtype='float32')
                imgs.append(img)
                continue
            pose_2d = poses_2d[0][i - init_frame] // 4    # crop image around these points
            min_x, max_x = np.min(pose_2d, 0)[0], np.max(pose_2d, 0)[0]
            min_y, max_y = np.min(pose_2d, 0)[1], np.max(pose_2d, 0)[1]
            min_x = max( 0, min_x)
            min_y = max( 0, min_y)
            max_x = max( 2, max_x)
            max_y = max( 2, max_y)
            max_x = min( 512, max_x)
            max_y = min( 512, max_y)
            min_x = min( 510, min_x)
            min_y = min( 510, min_y)
            scale = max(max_x - min_x, max_y - min_y) * 1.33
            center = np.array( [(min_x + max_x)/2, (min_y + max_y)/2]  )
            img = Crop(img, center, scale)
            #if img.shape[0] != 224 or img.shape[1] != 224:
            #    img = cv2.resize(img, (224, 224))
            img = img.astype('float32')
            img = img / 256.
            img = ( img - self.img_mean ) / self.std_dev
            #viz_img(img, i)
            #print(img.shape)
            imgs.append(img)
        imgs = torch.from_numpy(np.array(imgs))
        return imgs

    def __getitem__(self, index):
        poses_3d = self.annot[index+1][1]

        #poses_3d = self._kps_to_h36m(poses_3d.reshape(-1, 28, 3))
        #poses_2d = self._kps_to_h36m(poses_2d.reshape(-1, 28, 2))

        poses_3d, init_frame = self.get_chunk(poses_3d, self.max_length)

        #poses_2d = self.get_chunk(poses_2d, init_frame)
        #imgs = self.get_images(index, init_frame, poses_2d)
        #imgs = np.ones((224, 224, 3)) #None

        if self.max_length==-1:
            imgs = self.annot[index+1][0]
            poses_3d = self.annot[index+1][1]
        else:
            imgs = self.annot[index+1][0][init_frame : init_frame + self.max_length]

        #if self.split=='train':
        #  sims = np.load(open(self.data_path + '{}_sims_train.npy'.format(index+1), 'rb'))
        #else:
        #  sims = np.load(open(self.data_path + '{}_sims_val.npy'.format(index+1), 'rb'))

        #sims = sims[ init_frame : init_frame + self.max_length]
        #sims = sims[ :, init_frame : init_frame + self.max_length]
        
        # start_token = np.ones((1, 17, 3))
        # poses_3d = np.concatenate((start_token, poses_3d), axis=0)
        # padded_poses, pad_vec = self.pad_poses(poses_3d)
        # pad_vec = np.ones((self.max_length, 1))

        return imgs, poses_3d #, sims  #, pad_vec, poses_2d

    def __len__(self):
        return self.n_samples

if __name__ == "__main__":
    mpi  = MPI_INF("mpi_inf_embeds/", split='train', max_length=-1)
    dataloader = torch.utils.data.DataLoader(mpi, batch_size=1, shuffle=True)
    max_len = 0
    for i, (imgs, p_3d) in enumerate(dataloader):
        #print(p_3d.shape)
        #imgs = imgs.cuda()
        print(p_3d.shape)
        #import pdb; pdb.set_trace()
        #print(p_2d.shape)
        #print(p_2d)
        #continue
        #if i == 20:
        #    break
        len = p_3d.shape[1]
        if len > max_len:
          max_len = len
    print("Maximum length : ", max_len)

