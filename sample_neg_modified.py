import torch
import torch.nn as nn
import random

def get_mpjpe( poses):

  bsz, n_masked, fsz = poses.shape
  mpjpe_scores = torch.zeros((bsz, n_masked, n_masked))
  poses = poses.view(bsz, n_masked, -1, 3)
  for i in range(n_masked):
    mpjpe_scores[:, i] = ((poses[:, i:i+1] - poses)**2).sum(-1).sqrt().mean(-1)
  return mpjpe_scores

def get_cos_sim(y):
  # y.shape = [BS, n_masked, fsz]

  bsz, n_masked, fsz = y.shape
  sims = torch.zeros((bsz, n_masked, n_masked))
  for i in range(n_masked):
    logits = torch.cosine_similarity(y[:, i:i+1], y, dim=-1)
    sims[:, i] = 1 - logits

  return sims


def sample_negs( quant_rep, poses, n_negatives=10, use_pose_sampling=True):
  '''
  PARAMS:
   quant_rep.shape = [ BS, seq_len, embed_dim ]
   poses      = [BS, n_masked_frames, pose_dim]
   n_negatives= int, how many distractors sampled
   use_pose_sampling = whether to use pose or image feature cosine based sampling

  TO RETURN:
   negs = [ BS, n_masked_frames, embed_dim, n_distractors]
   neg_idxs

  '''

  #import pdb; pdb.set_trace()
  bsz, seq_len, fsz = quant_rep.shape
  if use_pose_sampling:
    sim_scores = get_mpjpe(poses)
  else:
    sim_scores = get_cos_sim(quant_rep)   # selecting negative from quantized frames only

  n_masked = sim_scores.size(1)
  top_k_frames = int(0.3 * n_masked)  # n_negatives

  idxs = sim_scores.sort(2).indices[:, :, -top_k_frames:]  # bsz, n_masked, top_k

  random_sel = random.sample(range(top_k_frames), n_negatives)
  idxs = idxs[:, :, random_sel]
  #idxs.shape = [BS, n_masked, top_k]

  idxs = idxs.reshape(bsz, -1)
  negs = torch.zeros(( bsz, n_negatives * n_masked, fsz)).cuda()

  for i in range(bsz):
    negs[i] = quant_rep[i][idxs[i]]

  negs = negs.view( bsz, n_masked, n_negatives, fsz).permute(0, 1, 3, 2)

  return negs, idxs

  # plan of action:
  #  1. get top k% dissimilar frames
  #  2. randomly/uniformly sample 5 frames from each
  #  3. These are our distractors
