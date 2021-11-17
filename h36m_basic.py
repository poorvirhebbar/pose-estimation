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

train_subjects = [1] #, 5, 6, 7, 8]
test_subjects = [9, 11]

class H36M(data.Dataset):
    def __init__(self, split, max_len=200):
        print('==> initializing H36M {} data.'.format(split))
        print('data dir {}'.format(ref.h36mDataDir))

        if split=='train':
            subjects = train_subjects
        else:
            subjects = test_subjects

        video_embed_files = [ os.path.join(ref.h36mDataDir, 'img_embeds_h36m_s_'+str(x)+'.pkl') for x in subjects ]
        sim_embed_files =   [ os.path.join(ref.h36mDataDir, 'img_embeds_h36m_s_'+str(x)+'_sims.pkl') for x in subjects]

        print(video_embed_files)
        print(sim_embed_files)

        self.embeds_loc = []
        self.embeds = []
        self.sims = []
        for vid_embed_file, sim_embed_file in zip(video_embed_files, sim_embed_files):
            f = open(vid_embed_file, 'rb')
            f2 = open(sim_embed_file, 'rb')
            dict = pickle.load(f)
            temp = pickle.load(f2)
            for key in dict.keys():
                #self.embeds_loc.append( ( vid_embed_file, key ))
                self.embeds.append(dict[key])
                self.sims.append(temp[key])

        self.split = split

        assert len(self.embeds) == len(self.sims)

        #self.n_samples = len(self.embeds_loc)
        self.n_samples = len(self.embeds)
        self.max_len = max_len
        print('Loaded {} with {} labelled samples'.format(split, self.n_samples))


    def get_chunk(self, embeds, sims, length):
        n_frames = embeds.shape[0]
        #print(n_frames)
        init_frame = random.randint(0, n_frames - length)
        chunk = embeds[init_frame: init_frame + length ]
        sims_chunk = sims[init_frame : init_frame + length ]
        sims_chunk = sims_chunk[:, init_frame : init_frame + length]
        #print(sims_chunk)
        return chunk, sims_chunk, init_frame


    def __getitem__(self, index):

        #file, key = self.embeds_loc[index]
        #dict = pickle.load(open(file, 'rb'))
        #embeds = dict[key]
        embeds = self.embeds[index]
        sims = self.sims[index]
        imgs, sims, init_frame = self.get_chunk( embeds, sims, self.max_len)
        #print(imgs.shape)
        #print(sims.shape)
        return imgs, sims

    def __len__(self):
        return self.n_samples



if __name__ == "__main__":
    #opt = opts.opts().parse()
    h36m = H36M(split='train', max_len=400)
    dataloader = torch.utils.data.DataLoader(h36m, batch_size=2, shuffle=True)
    saved = {}
    for i, (imgs, sims) in enumerate(dataloader):
        print(imgs.shape)
        print(sims.shape)
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

    #f = open("saved_h36m2.pickle", "wb")
    #pkl.dump(saved, f)
    #f.close()
