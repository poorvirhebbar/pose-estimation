## dataloader for h3.6m videos

import torch
import torch.utils.data as data
import numpy as np
import os
import math
import torchvision
import pickle
import ref
import random
from data_loader.utils import load_annot_file

train_subjects = [1, 5, 6] #, 7, 8]
test_subjects = [9] #, 11]


class H36M(data.Dataset):
    def __init__(self, split, max_len=100):
        print('==> initializing H36M {} data.'.format(split))
        print('data dir {}'.format(ref.h36mDataDir))

        if split=='train':
            subjects = train_subjects
        else:
            subjects = test_subjects

        data_dir = './h36m_embeds'
        print("Using Masked Images")
        video_embed_files = [ os.path.join(ref.h36mDataDir, 'img_mask_embeds_h36m_s_'+str(x)+'_new.pkl') for x in subjects ]
        # sim_embed_files =   [ os.path.join(ref.h36mDataDir, 'img_embeds_h36m_s_'+str(x)+'_sims.pkl') for x in subjects]

        annot_file = os.path.join(data_dir, 'annot_cam_' + ('train' if split == 'train' else 'test') + '.pickle')

        self.annot = load_annot_file(annot_file)
        print(video_embed_files)

        self.embeds_loc = []
        self.embeds = []
        self.sims = []

        self.start_frames = []
        cum_frames = 0
        self.index_to_key = []
        self.key_to_index = {}
        count = 0
        for vid_embed_file in video_embed_files:
            f = open(vid_embed_file, 'rb')
            dict = pickle.load(f)
            for key in dict.keys():
                #print(key)
                #self.embeds_loc.append( ( vid_embed_file, key ))
                self.index_to_key.append(key)
                self.key_to_index[key] = count
                self.embeds.append(dict[key])
                n_frames = dict[key].shape[0]
                self.start_frames.append([cum_frames, n_frames])
                cum_frames += n_frames
                count+=1


        self.split = split

        #self.n_samples = len(self.embeds_loc)
        self.n_samples = len(self.embeds)
        self.max_len = max_len
        print('Loaded {} with {} labelled samples'.format(split, self.n_samples))


    def preprocess_poses(self, pose, root=6):
        pose = pose - pose[:, root:root+1]
        pose = pose / 1000
        return pose

    def get_chunk(self, embeds, poses, length, return_full=False):
        n_frames = embeds.shape[0]
        #print(n_frames)
        if not return_full:
          init_frame = random.randint(0, n_frames - length)
          chunk = embeds[init_frame: init_frame + length ]
          pose_chunk = poses[init_frame: init_frame + length]
        else:
          init_frame = 0
          chunk = embeds
          pose_chunk = poses

        pose_chunk = self.preprocess_poses(pose_chunk)
        #print(sims_chunk)
        return chunk, pose_chunk, init_frame


    def __getitem__(self, index):

        #file, key = self.embeds_loc[index]
        #dict = pickle.load(open(file, 'rb'))
        #embeds = dict[key]

        start, n_frames = self.start_frames[index]

        embeds = self.embeds[index]
        vid_poses = self.annot['joint_3d_mono'][start : start+n_frames]

        key = self.index_to_key[index]
        ca = int(key[-1])
        cameras = [1,2,3,4]
        cameras.remove(ca)

        idx = random.randint(0,2)
        new_key = key[:-1] + str(cameras[idx])
        new_index = self.key_to_index[new_key]

        #print(key)
        #print(new_key)
        new_start, new_n_frames = self.start_frames[new_index]
        new_embeds = self.embeds[new_index]
        new_vid_poses = self.annot['joint_3d_mono'][new_start : new_start+new_n_frames]
        new_vid_poses = self.preprocess_poses(new_vid_poses)


        if self.max_len==-1:
          img_embeds, poses, init_frame = self.get_chunk( embeds, vid_poses, self.max_len, True)
        else:
          imgs, poses, init_frame = self.get_chunk( embeds, vid_poses, self.max_len, False)
          imgs = torch.tensor(imgs).unsqueeze(0)
          poses = torch.tensor(poses).unsqueeze(0)
          new_imgs = torch.tensor(new_embeds[init_frame : init_frame + self.max_len]).unsqueeze(0)
          new_poses =  torch.tensor(new_vid_poses[init_frame : init_frame + self.max_len]).unsqueeze(0)

          img_embeds = torch.cat([imgs, new_imgs], 0)
          poses = torch.cat([poses, new_poses], 0)

        sub_id = self.annot['subject'][start]

        if sub_id == 1:
            sub_mask = 1
        else:
            sub_mask = 0

        #imgs, poses, init_frame = self.get_chunk( embeds, vid_poses, self.max_len)
        #names = names[init_frame : init_frame + self.max_len]

        return img_embeds, poses, sub_mask #, names
        '''
        start, n_frames = self.start_frames[index]
        
        action = self.annot['action'][start]
        subaction = self.annot['subaction'][start]
        camera = self.annot['camera'][start]
        subject = self.annot['subject'][start]
        print("{}_{}_{}_{}".format(action, subaction, camera, subject))
        
        vid_poses = self.annot['joint_3d_mono'][start: start+n_frames]
        embeds = self.embeds[index]
        if self.max_len==-1:
          imgs, poses, init_frame = self.get_chunk( embeds, vid_poses, self.max_len, True)
        else:
          imgs, poses, init_frame = self.get_chunk( embeds, vid_poses, self.max_len, False)

        sub_id = self.annot['subject'][start]
        if sub_id == 1:
            sub_mask = 1
        else:
            sub_mask = 0
        return imgs, poses, sub_mask
        '''

    def __len__(self):
        return self.n_samples



if __name__ == "__main__":
    #opt = opts.opts().parse()
    h36m = H36M(split='test', max_len=150)
    dataloader = torch.utils.data.DataLoader(h36m, batch_size=1, shuffle=True)
    saved = {}
    max_len = -1
    for i, (imgs, poses, x) in enumerate(dataloader):
        #print(imgs.shape)
        len = imgs.shape[1]
        if len > max_len:
          max_len = len
        print('-'*15)
        #print(poses.shape)
        '''
        img = img[0] * 256
        p_3d = p_3d[0]
        saved[i] = [ img.detach().cpu().numpy(), p_3d.detach().cpu().numpy() ]
        #print(meta['pose'].shape)
        #import pdb; pdb.set_trace()
        #print(meta['pose_2d'].shape)
        '''
        if i==10:
            break

    print("Number of videos : ", i+1)
    print("Maximum length : ", max_len)
    #f = open("saved_h36m2.pickle", "wb")
    #pkl.dump(saved, f)
    #f.close()
