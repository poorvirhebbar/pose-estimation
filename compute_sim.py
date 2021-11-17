import os
import pickle
import numpy as np

import torch
import torch.nn as nn

def get_mpjpe( ref_pose, rest):
  '''
    PARAMETERS:
      ref_pose : [1, nJoints, 3]
      rest     : [seq_len, nJoints, 3]
    RETURN:
      mpjpe_scores : [seq_len]
  '''
  mpjpe_scores = ((ref_pose - rest)**2).sum(-1).sqrt().mean(-1)
  return mpjpe_scores

#subj_files = os.listdir('/mnt/data/srijon/h36m_embeds')
subj_files = ['image_embeds_val_mpi.pkl']

#cos = nn.CosineSimilarity(1)

saved_sims = {}
#soft = nn.Softmax(0)

done = [] #[1, 5, 7, 8, 11]

for i, subj in enumerate(subj_files):
  if '.pkl' not in subj or 'sims' in subj:
    continue

  #flag = 0
  #for name in done:
  #  if str(name) in subj:
  #    flag = 1
  #    break
  #if flag==1:
  #  continue
  print(subj)
  f = open('mpi_inf_embeds/'+subj, 'rb')
  #f.seek(0)
  d = pickle.load(f)
  subj = subj.split('.')[0]

  for key in d.keys():
    print(key)
    pose_embed = torch.tensor(d[key][1]).cuda()
    sims = torch.zeros(pose_embed.shape[0], pose_embed.shape[0]).cuda()
    rest = pose_embed.reshape(-1, 17, 3)
    for j in range(pose_embed.shape[0]):
      #print(j)
      ref = pose_embed[j, :].reshape(1, 17, 3)
      sim = get_mpjpe(ref, rest)
      sims[j] = sim
      #print(sim.shape)
      #print(sim[:100])
      #if j==10:
      #  exit()
    sims = sims.cpu().int().numpy()
    with open('mpi_inf_embeds/{}_sims_val.npy'.format(key), 'wb') as f:
      np.save(f, sims)
    #pickle.dump(sims, open('mpi_inf_embeds/{}_sims_val.pkl'.format(key), 'wb'))

    #print(sims[0, :100])
    #print(sim.shape)
    #exit()
    #print(sims.sum())
    #print()
    #saved_sims[key] = sims.cpu().int().numpy()

  #exit()
  #pickle.dump(saved_sims, open('{}_sims.pkl'.format(subj), 'wb'))
  #saved_sims = {}



