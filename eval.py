#import sys
#sys.path.insert(0,'/home/project/multipose/src_rishabh_train_coco')
import numpy as np
import numpy.ma as ma
import ref
from utils import AverageMeter
import cv2
from img import Transform
#from soft_argmax import SoftArgmax2D
# from  utils.pyTools import Show3d
from opts import opts
import torch
import torch.nn as nn
from torch.nn import functional as F
# from visualise_maps import visualise_map



def getPreds(hm):
  assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
  res = hm.shape[2]
  nJoints = hm.shape[1]

  hm = hm.reshape(( -1, nJoints, res*res))
  idx = np.argmax(hm, axis = 2)
  #hm = F.softmax(hm, 2)
  #hm = hm.reshape(( -1, nJoints, res, res))
  '''
  #print(hm.shape)
  vec_x = hm.sum(dim=3)
  vec_y = hm.sum(dim=2)
  #print(vec_x)
  #print(vec_y)
  vec_x = vec_x * torch.arange(1,res+1).type(torch.cuda.FloatTensor)
  vec_y = vec_y * torch.arange(1,res+1).type(torch.cuda.FloatTensor)
  vec_x = vec_x.sum(2) - 1
  vec_y = vec_y.sum(2) - 1
  vec_x = vec_x.cpu().numpy()
  vec_y = vec_y.cpu().numpy()
  '''
  #argmax = SoftArgmax2D()
  #coords = argmax(hm.float())
  #coords = coords.cpu().numpy()

  preds = np.zeros((hm.shape[0], hm.shape[1], 2))
  for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
      preds[i, j, 0], preds[i, j, 1] = idx[i, j] % res, idx[i, j] / res
  return preds

def calcDists(preds, gt, normalize):
  #dists = np.zeros((preds.shape[1], preds.shape[0]))
  dists = np.zeros((preds.shape[0], preds.shape[1]))
  for i in range(preds.shape[0]):
    for j in range(preds.shape[1]):
      if gt[i, j, 0] > 0 and gt[i, j, 1] > 0:
        dists[i][j] = ((gt[i][j] - preds[i][j]) ** 2).sum() ** 0.5 / normalize[i]
      else:
        dists[i][j] = -1
  return dists

def distAccuracy(dist, thr = 0.5):
  dist = dist[dist != -1]
  if len(dist) > 0:
    return 1.0 * (dist < thr).sum() / len(dist)
  else:
    return -1

def pck(output, target):
  """
  Calculates PCK@0.2 accuracy
  Predicted Keypoints within 0.2*torso_diameter of the ground truth keypoints
  """
  return PCKh(output, target, thresh=0.2, hm_flipped=False, len='torso')

  '''
  # output and target both have MPII style annotations
  preds = getPreds(output)*4
  gt = getPreds(target)*4
  threshold = np.array([0.5*((((gt[i][8] - gt[i][9])**2).sum(-1))**0.5) for i in range(preds.shape[0])])

  dist = calcDists(preds, gt, threshold)
  acc_count, total_count = 0, 0

  for i in range(preds.shape[0]):
    for j in range(preds.shape[1]):
      if dist[i][j]!=-1 and dist[i][j]<=1:
        acc_count += 1
      if dist[i][j]!=-1:
        total_count += 1
  return acc_count / total_count
  '''

def Accuracy(output, target):
  preds = getPreds(output)
  gt = getPreds(target)
  dists = calcDists(preds, gt, np.ones(preds.shape[0]) * ref.outputRes / 10)
  acc = np.zeros(len(ref.accIdxs))
  avgAcc = 0
  badIdxCount = 0

  for i in range(len(ref.accIdxs)):
    acc[i] = distAccuracy(dists[ref.accIdxs[i]])
    if acc[i] >= 0:
      avgAcc = avgAcc + acc[i]
    else:
      badIdxCount = badIdxCount + 1

  if badIdxCount == len(ref.accIdxs):
    return 0
  else:
    return avgAcc / (len(ref.accIdxs) - badIdxCount)



def visualise_maps(hm_preds,paf_preds,depthmaps,indexes,fusion):

    return 0

'''
def visualise_3d(pred,gt,gt_ind,index,dir_path,epoch):

    pred_root_rel = pred[:,:3] - pred[ref.root,:3]
    gt_length=0
    len_pred=0
    tot_cnt=0
    for e in ref.edges:
        if pred_root_rel[e[0]][0]!=0 and pred_root_rel[e[0]][1]!=0 and pred_root_rel[e[1]][1]!=0 and pred_root_rel[e[1]][1]!=0:
            len_pred += ((pred_root_rel[e[0]][:2] - pred_root_rel[e[1]][:2]) ** 2).sum() ** 0.5
            gt_length += ((gt[e[0]][:2] - gt[e[1]][:2]) ** 2).sum() ** 0.5
        else:
            tot_cnt=tot_cnt+1

    gt_root   =  gt[ref.root]
#    print('gt_root is ,',gt_root)
    for j in range(ref.nJoints):
        pred_root_rel[j] = ((pred_root_rel[j]) / len_pred) * gt_length + gt_root
    data={}
    data['joint']=pred_root_rel
    data['gt']=gt
    Show3d(data,index,gt_ind,dir_path,epoch)
    return 0
def PCk_multi(hm_preds,paf_preds,depthmaps,indexes,fusion,iter_ind,vis_trainIters,epoch):
    #without batch

    pckh_list=[];
    mpjpe_list=[]
    mpjpe_itm=[]
    pck_itm=[];
    for i in range(0,ref.datasets):
        pckh_list.append(AverageMeter())
        pck_itm.append(0)
        mpjpe_list.append(AverageMeter())
        mpjpe_itm.append(0)

    for hm_pred,paf_pred,depthmap,index in zip(hm_preds,paf_preds,depthmaps,indexes):
        kpt_3d=fusion.get_kpt3d(index);


        hm_pred=hm_pred.view(1,hm_pred.shape[0],hm_pred.shape[1],hm_pred.shape[2])
        paf_pred=paf_pred.view(1,paf_pred.shape[0],paf_pred.shape[1],paf_pred.shape[2])

        oriImg = np.zeros((hm_pred.shape[2],hm_pred.shape[3]))

        subset,candidate = part_affinity(oriImg,hm_pred,paf_pred)
        people_3d = np.zeros((subset.shape[0],ref.nJoints,3))
        depthmap = depthmap.numpy()
        depthmap = np.squeeze(depthmap)

        for i in range(subset.shape[0]):
            for j in range(subset.shape[1]):
                if j<16:
                    if not int(subset[i][j])==-1:
                        people_3d[i][j][0],people_3d[i][j][1]=candidate[int(subset[i][j])][0],candidate[int(subset[i][j])][1]
                        x,y=int(people_3d[i][j][0]),int(people_3d[i][j][1])
                        if np.count_nonzero(kpt_3d[:,2]) > 0:
                            people_3d[i][j][2]=(depthmap[j][y][x] + 1) * ref.outputRes / 2

        pckh_lis = []
        pck_ind_lis = []
        mpjpe_lis=[]
        for gt_person in kpt_3d:
            max_PCKh=0
            pck_ind=0;
            max_PCKh=0
            tmp_mpjpe=100000;
            for ind,person in enumerate(people_3d):
                tmp_gt_person=gt_person.copy()
                tmp_person=person.copy()
                tmp_pck=PCKh(tmp_person, tmp_gt_person)
                tmp_mpjpe=min(tmp_mpjpe,MPJPE(tmp_person,tmp_gt_person))
                if tmp_pck > max_PCKh:
                    pck_ind=ind
                    max_PCKh=tmp_pck
            mpjpe_lis.append(tmp_mpjpe)
            pckh_lis.append(max_PCKh)
            pck_ind_lis.append(pck_ind)

        avg_PCKh = sum(pckh_lis) / len(pckh_lis)
        avg_mpjpe = sum(mpjpe_lis) / len(mpjpe_lis)
        index_img=fusion.get_dataset_ind(index)
        pckh_list[index_img].update(avg_PCKh,1)
        pck_itm[index_img]=pck_itm[index_img]+1

        mpjpe_list[index_img].update(avg_mpjpe,1)
        mpjpe_itm[index_img]=mpjpe_itm[index_img]+1


        if (iter_ind%vis_trainIters) == 0:
            dir_path=ref.visDir[fusion.get_dataset_ind(index)]
            for gt_ind,gt in enumerate(kpt_3d):
                if pck_ind_lis[gt_ind] < people_3d.shape[0] and index_img!=2 :
                    visualise_3d(gt,people_3d[pck_ind_lis[gt_ind]],gt_ind,index,dir_path,epoch)
            img=fusion.get_img(index)
            dataset_ind=fusion.get_dataset_ind(index)
            visualise_map(hm_pred,paf_pred,img,dataset_ind,index,dir_path,epoch)


    return pckh_list, pck_itm,mpjpe_list,mpjpe_itm
'''


def flip_pose(pose):
    """
        Performs a horizontal flip of the pose
        Parameters:
            pose: n x 16 x 2
        returns:
            flipped_pose: n x 16 x 2
    """
    flip_map = [[0,5], [1,4], [2,3], [10,15], [11,14], [12,13]]
    flipped_pose = pose.copy()

    flipped_pose[..., 0] = 256 - flipped_pose[..., 0]
    for pair in flip_map:
        p1 = pair[0]
        p2 = pair[1]
        flipped_pose[:, p1, 0] = 256 - pose[:, p2, 0]
        flipped_pose[:, p2, 0] = 256 - pose[:, p1, 0]

    return flipped_pose

def PCKh4( kpt, gt, thresh, meta, split):

    if split=='train':
        return PCKh( kpt, gt, thresh)

    bs = kpt.shape[0]    # kpt.shape = (bs, 16, 64 ,64)
    gt = getPreds(gt)
    kpt = getPreds(kpt)  # kpt.shape = (bs, 16, 2)

    tf_pred = np.zeros((bs, ref.nJoints, 2))
    tf_gt = meta['pos_gt'].cpu().numpy()  # np.zeros((bs, ref.nJoints, 2))
    assert tf_gt.shape == (bs, ref.nJoints, 2)

    for i in range(bs):
        for j in range(ref.nJoints):
            tf_pred[i][j] = Transform( kpt[i][j], meta['center'][i], meta['scale'][i], 0, res=ref.outputRes, invert=True)
            #tf_gt[i][j]   = Transform( gt[i][j], center[i], scale[i], 0, res=ref.outputRes, invert=True)

    jnt_visible = 1 - meta['jnt_missing'].cpu().numpy()
    uv_error = tf_gt - tf_pred
    uv_err = np.linalg.norm(uv_error, axis=2)

    headsize = meta['headbox'][:, 1, :] - meta['headbox'][:, 0, :]
    headsize = np.linalg.norm(headsize, axis=1, keepdims = True)
    headsize *= 0.6
    #print(headsize.shape)

    scale = np.multiply(headsize, np.ones((1, 16)))
    #print(scale.shape)

    #scale = headsize
    scaled_uv_err = np.divide(uv_err, scale)
    scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
    #print("Joint visible shape : ", jnt_visible.shape)
    #jnt_count = np.sum( jnt_visible, axis=1)
    jnt_count = jnt_visible.sum(axis=1)
    less_than_threshold = np.multiply((scaled_uv_err < thresh), jnt_visible)

    PCKh = np.divide(100. * less_than_threshold.sum(axis=1), jnt_count)
    #print(PCKh)
    return np.mean(PCKh), jnt_visible.sum()

def PCKh(hm, gt, thresh = ref.PCKthresh, hm_flipped=False, len='head'):
    # Assuming the kpt and gt are all in heatmap dimensions
    gt = getPreds(gt) * 4
    kpt = getPreds(hm) * 4

    # import pdb; pdb.set_trace()
    if hm_flipped is not False:
        flipped_kpt = getPreds(hm_flipped) * 4
        kpt2 = flip_pose(flipped_kpt)
        # kpt[..., 0] = (kpt[..., 0] + kpt2[..., 0]) / 2
        kpt = (kpt + kpt2) / 2

    # Assuming that the kpt dimensions are  16 x 2
    if len=='head':
        neck, head = 8, 9
    elif len=='torso':
        neck, head = 12, 13
    # PCKh theshold is the thresh times the size of the head
    head_len = ((gt[:, head, :] - gt[:, neck,:]) ** 2).sum() ** 0.5
    #head_len = 6.4
    PCKh_thresh = head_len * thresh
    dist = ((kpt - gt)**2).sum(-1)**0.5
    if len=='torso':
        tmp1 = dist[:, :6]    # LSP doesn't contain pelvis and thorax joints
        tmp2 = dist[:, 8:]
        dist = np.concatenate([tmp1, tmp2], 1)
        # dist.shape = [bs, 14, 1]
    '''
    dist=np.zeros(ref.nJoints)
    valid_j=0
    for i in range(ref.nJoints):
        if gt[i][0]!=0 and gt[i][1]!=0:
            valid_j=valid_j+1
            dist[i] = (((gt[i][0] - kpt[i][0]) ** 2) + ((gt[i][1] - kpt[i][1]) ** 2)) ** 0.5
    '''

    inliers = ma.masked_less_equal(dist, PCKh_thresh).mask
    # inliers = ma.masked_where((dist < PCKh_thresh) & (dist > 1.0), dist).mask
    # invalid = ma.masked_where(dist < 1.0, dist).mask
    valid = dist.shape[0]*dist.shape[1] # - invalid.astype('Float32').sum()
    nInliers = inliers.astype('Float64').sum()
    # nInliers=nInliers-(ref.nJoints)
    if valid!=0:
        return nInliers / valid, valid
    else:
        return 0, 0

def PCKh2(hm, gt, thresh = ref.PCKthresh):
    # Assuming the kpt and gt are 2D coordinates
    #kpt = hm * 4
    kpt = hm
    gt = gt
    # Assuming that the kpt dimensions are  16 x 2
    neck, head = 8, 9
    # PCKh theshold is the thresh times the size of the head
    head_len = ((gt[:, head, :] - gt[:, neck,:]) ** 2).sum() ** 0.5
    PCKh_thresh = head_len * thresh
    dist = ((kpt - gt)**2).sum(-1)**0.5

    inliers = ma.masked_less_equal(dist, PCKh_thresh).mask
    valid = dist.shape[0]*dist.shape[1] # - invalid.astype('Float32').sum()
    nInliers = inliers.astype('Float64').sum()

    if valid!=0:
        return nInliers / valid, valid
    else:
        return 0, 0

def PCKh3(hm, gt, thresh = ref.PCKthresh, hm_flipped=False):
    # Assuming the kpt and gt are all in [batchsize, 16, 3]
    #if gt.shape==(16,2):
    #    gt = np.concatenate( [gt, np.zeros((16,1))], axis=1)
    #assert gt.shape==(16,3)

    hm = hm*4
    gt = gt
    kpt = hm
    if hm_flipped is not False:
        flipped_kpt = getPreds(hm_flipped) * 4
        kpt2 = flip_pose(flipped_kpt)
        # kpt[..., 0] = (kpt[..., 0] + kpt2[..., 0]) / 2
        kpt = (kpt + kpt2) / 2

    #print(hm[0])
    #print(gt[0])
    neck, head = 8, 9
    # PCKh theshold is the thresh times the size of the head
    head_len = ((gt[:, head, :2] - gt[:, neck, :2]) ** 2).sum() ** 0.5
    PCKh_thresh = head_len * thresh
    dist = ((kpt[:, :, :2] - gt[:, :, :2])**2).sum(-1)**0.5

    inliers = ma.masked_less_equal(dist, PCKh_thresh).mask

    valid = dist.shape[0]*dist.shape[1]        # - invalid.astype('Float32').sum()
    nInliers = inliers.astype('Float64').sum()

    if valid!=0:
        return nInliers / valid, valid
    else:
        return 0, 0


def MPJPE(pred3D, kpt_3d):
    '''
    Calculates MPJPE for the given predictions and the targets. Assuming
    the inputs are in heatmap dimensions and require Zhou's scaling
    params:
        pred3D: A 16 x 3 Tensor of predicted pose
        kpt_3d: A 16 x 3 Tensor of ground truth pose
    returns:
        mpjpe: The mean perjoint error

    '''
    p = pred3D.copy()
    gt_length=0
    len_pred=0
    tot_cnt=0
    for e in ref.edges:
        gt_length += ((kpt_3d[e[0]][:] - kpt_3d[e[1]][:]) ** 2).sum() ** 0.5
        len_pred += ((pred3D[e[0]][:] - pred3D[e[1]][:]) ** 2).sum() ** 0.5

    ref.root = 6
    pRoot = p[ref.root].copy()
    kptRoot = kpt_3d[ref.root].copy()

    for j in range(ref.nJoints):
        p[j] = ((p[j] - pRoot) / len_pred) * gt_length + kptRoot

    #print("Pred")
    #print(p)
    #print("GT")
    #print(kpt_3d)

    mpjpe = 0
    dpth_check = np.count_nonzero(kpt_3d[:,2])
    if dpth_check>0:
      for j in range(ref.nJoints):
        dis = ((p[j] - kpt_3d[j]) ** 2).sum() ** 0.5
        mpjpe += dis / ref.nJoints

    return mpjpe

def get_MPJPE(hm, dm, gt):

    hm = hm.cpu().detach().numpy()
    dm = dm.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    pred2D = getPreds(hm)
    idx = pred2D.copy()
    idx = idx.astype('int')
    #z = np.zeros((hm.shape[0], hm.shape[1]))

    #for i in range(hm.shape[0]):
    #   for j in range(hm.shape[1]):
    #       z[i,j] = dm[i,j,idx[i,j,1],idx[i,j,0]]

    #z = (z + 1) * ref.outputRes / 2
    dm = (dm + 1) * ref.outputRes / 2

    pred3D = np.concatenate((pred2D, dm[:, :, np.newaxis]), axis=-1)
    mpjpe = 0
    for i in range(pred3D.shape[0]):
        mpjpe += MPJPE(pred3D[i], gt[i])

    return mpjpe / pred3D.shape[0]
