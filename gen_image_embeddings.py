import torch
import torch.nn as nn
import pickle
from torchvision.models import resnet50
from tqdm import tqdm

class ResNet(nn.Module):
  def __init__(self):
    super(ResNet, self).__init__()
    resnet = resnet50(pretrained=True)
    modules = list(resnet.children())[:-1]
    self.resnet = nn.Sequential(*modules)

  def forward(self, x):
    return self.resnet(x)

import random
import cv2
import torch
import torch.utils.data as data
import numpy as np
import os
import math
from img_utils import Crop

class MPI_INF():
    def __init__(self, data_path, split='train', max_length=1000):
        print('==> initializing MPI-INF {} data.'.format(split))

        if split == 'train':
            file_path = os.path.join(
                                data_path,
                                'mpi_inf_videowise_annot.pkl')
        else:
            file_path = os.path.join(
                                data_path,
                                'mpi_inf_videowise_annot_val.pkl')

        with open(file_path, 'rb') as f:
            annot = pickle.load(f)

        self.data_path = data_path
        self.annot = annot
        self.split = split
        self.max_length = max_length

        self.n_samples = len(self.annot['kp_2d'])
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

    def get_chunk(self, poses, init_frame=0):
        '''
            poses: n x n_joints x 2 | 3
        '''
        self.n_frames = poses.shape[0]
        #if init_frame is None:
        #    init_frame = random.randint(0, n_frames - self.max_length )
        chunk = poses[init_frame: init_frame + self.max_length ]
        return chunk, init_frame

    def get_images(self, index, init_frame, poses_2d):
        imgs = []
        for i in range(init_frame, init_frame + self.max_length):
            if i >= self.n_frames:
                break
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

            img = img.astype('float32')
            img = img / 256.

            img = ( img - self.img_mean ) / self.std_dev

            #viz_img(img,i)
            imgs.append(img)
        imgs = torch.from_numpy(np.array(imgs))
        return imgs

    def __getitem__(self, index, init_frame=0):
        poses_3d = self.annot['kp_3d'][index]
        poses_2d = self.annot['kp_2d'][index]

        poses_3d = self._kps_to_h36m(poses_3d.reshape(-1, 28, 3))
        poses_2d = self._kps_to_h36m(poses_2d.reshape(-1, 28, 2))

        poses_3d, _ = self.get_chunk(poses_3d, init_frame)
        poses_2d = self.get_chunk(poses_2d, init_frame)

        imgs = self.get_images(index, init_frame, poses_2d)
        # start_token = np.ones((1, 17, 3))
        # poses_3d = np.concatenate((start_token, poses_3d), axis=0)
        # padded_poses, pad_vec = self.pad_poses(poses_3d)
        pad_vec = np.ones((self.max_length, 1))
        return imgs, poses_3d  #, pad_vec, poses_2d

    def len(self):
        return self.n_samples

    def get_len(self, index):
        return self.annot['kp_3d'][index].shape[0]

def main():

  b = 500
  d = MPI_INF("./data/mpi-inf/", split='train', max_length=b)

  model = ResNet().cuda()
  model.eval()

  dict = {}

  p_3d, emb = [], []
  for i in tqdm(range(d.len())):
    n_frames = d.get_len(i)
    temp_embed = torch.zeros((n_frames, 2048))
    temp_poses = torch.zeros((n_frames, 51))
    import pdb; pdb.set_trace()
    for j in tqdm(range(n_frames//b + 1)):
      with torch.no_grad():

        imgs, pose_3d = d.__getitem__(i, j*b)
        imgs = imgs.float().cuda()
        imgs = imgs.reshape(-1, 224, 224, 3)
        imgs = imgs.permute(0, 3, 1, 2)
        embeds = model(imgs)

        temp_poses[j*b : (j+1)*b] = torch.tensor(pose_3d).float().reshape(-1, 51)
        temp_embed[j*b : (j+1)*b] = embeds.detach().cpu().squeeze()

    #temp_embed = torch.tensor(temp_embed)
    #temp_embed =  temp_embed.reshape(-1, 2048)
    #temp_poses = torch.tensor(temp_poses)
    #temp_poses = temp_poses.reshape(-1, 51)

    dict[i+1] = [ temp_embed.numpy(), temp_poses.numpy()]
    print(i)

  print(len(dict))
  f = open('image_embeds_val_mpi.pkl', 'wb')
  pickle.dump(dict, f)

if __name__=='__main__':
  main()
