import sys
sys.path.append('/home/rahul/weak_supervision_eccv/src/')

import os
import numpy as np
import torch
import torch.utils.data as data
import pickle
from opts import opts
from tqdm import tqdm
from scipy import spatial
from common.error_metrics import cal_avg_l2_jnt_dist, scale_norm_pose, cal_p_mpjpe
import matplotlib.pyplot as plt
import argparse


def get_down_sampled_annot_ids(annot, down_sample=3):
    cam_ids = [1, 2, 3, 4]

    n_samples = annot['id'].shape[0]
    down_sample_ids = np.empty(0, dtype='int32')

    for cam_id in cam_ids:
        unq_cam_mask = annot['camera'] == cam_id
        unq_cam_ids = np.arange(n_samples)[unq_cam_mask]

        n_cam_samples = unq_cam_ids.shape[0]

        down_sampled_cam_ids = np.arange(0, n_cam_samples, down_sample)

        down_sample_ids = np.append(down_sample_ids, unq_cam_ids[down_sampled_cam_ids].astype('int32'))

    return down_sample_ids


def get_down_sampled_annot(annot, sampled_annot_ids):

    annot_down = {}

    for key in annot.keys():
        annot_down[key] = annot[key][sampled_annot_ids]

    return annot_down


def get_query_database_masks(annot_ids, q_subj, db_subj, annot):

    q_sub_mask = np.zeros(annot_ids.shape[0], dtype='bool')
    for q_s in q_subj:
        q_sub_mask = np.logical_or(q_sub_mask, annot['subject'][annot_ids] == q_s)

    q_sub_ids = np.arange(0, annot_ids.shape[0])[q_sub_mask]

    db_sub_mask = np.zeros(annot_ids.shape[0], dtype='bool')
    for db_s in db_subj:
        db_sub_mask = np.logical_or(db_sub_mask, annot['subject'][annot_ids] == db_s)

    db_sub_ids = np.arange(0, annot_ids.shape[0])[db_sub_mask]

    return q_sub_ids, db_sub_ids


def get_gt_pose_ret_mean_mpjpe(query_poses, db_poses, k_list=[1, 2, 5, 10, 20]):

    no_q_poses = query_poses.shape[0]

    mean_mpjpe = {}
    for k in k_list:
        mean_mpjpe[k] = 0.0

    db_poses_flat = db_poses.reshape(db_poses.shape[0], -1)

    kd_tree = spatial.cKDTree(db_poses_flat)

    for i in tqdm(range(no_q_poses), ascii=True):
        q_pose = query_poses[i:i+1, :, :]
        q_pose_flat = q_pose.reshape(1, -1)

        # temp = cal_p_mpjpe(np.tile(q_pose, (pose_database.shape[0], 1, 1)), pose_database, avg=False)
        # temp = np.sort(temp)

        k_max = max(k_list)
        
        ret_dist, ret_ids = kd_tree.query(q_pose_flat, k_max+1)
        ret_poses = db_poses[ret_ids]

        q_pose_ext = np.tile(q_pose, (k_max+1, 1, 1))

        ret_mpjpe = cal_avg_l2_jnt_dist(q_pose_ext, ret_poses, avg=False)
        
        for k in k_list:
            
            # ret_dist = ret_dist / 16.
            
            mean_mpjpe[k] = mean_mpjpe[k] + ret_mpjpe[1:k+1].mean()

    for k in k_list:
        mean_mpjpe[k] = mean_mpjpe[k] / float(no_q_poses)

    return mean_mpjpe


def get_emb_pose_ret_mean_mpjpe(q_embs, db_embs, q_poses, db_poses, k_list=[1, 2, 5, 10, 20]):

    no_q_poses = q_poses.shape[0]

    mean_mpjpe = {}
    min_mpjpe = []
    for k in k_list:
        mean_mpjpe[k] = 0.0

    kd_tree = spatial.cKDTree(db_embs)

    for i in tqdm(range(no_q_poses), ascii=True):
        q_emb = q_embs[i:i+1, :]
        q_pose = q_poses[i:i+1, :, :]

        k_max = max(k_list)

        ret_dist, ret_ids = kd_tree.query(q_emb, k_max+1)
        ret_poses = db_poses[ret_ids]

        q_pose_ext = np.tile(q_pose, (k_max+1, 1, 1))

        # import pdb; pdb.set_trace()
        ret_mpjpe = cal_avg_l2_jnt_dist(q_pose_ext, ret_poses, avg=False)

        for k in k_list:
            mean_mpjpe[k] = mean_mpjpe[k] + ret_mpjpe[1:k+1].mean()

    for k in k_list:
        min_mpjpe.append(np.min(mean_mpjpe))
        mean_mpjpe[k] = mean_mpjpe[k] / float(no_q_poses)

    return mean_mpjpe, min_mpjpe


def get_mpjpe_retrieval_orig(emb, annot, k_list, split, do_cross=False):

    n_samples = annot['id'].shape[0]
    poses = annot['joint_3d_rel']

    assert emb.shape[0] == n_samples

    mean_mpjpe_emb = {}
    mean_mpjpe_gt = {}

    for k in k_list:
        mean_mpjpe_emb[k] = []
        mean_mpjpe_gt[k] = []

    do_cross = do_cross
    if split == 'train':
        subj_list = [1, 5, 6, 7, 8]
    else:
        subj_list = [9, 11]

    for q_subj in subj_list:
        db_subjs = subj_list

        if do_cross is True:
            db_subjs = []

            for i in subj_list:
                if i != q_subj:
                    db_subjs.append(i)

        # getting oracle retrievals
        annot_ids = np.arange(n_samples)
        q_subj_mask, db_subj_mask = get_query_database_masks(annot_ids, [q_subj], db_subjs, annot)

        q_annot_ids = annot_ids[q_subj_mask]
        q_emb = emb[q_subj_mask]
        q_poses = poses[q_annot_ids]

        db_annot_ids = annot_ids[db_subj_mask]
        db_emb = emb[db_subj_mask]
        db_poses = poses[db_annot_ids]

        print('Cal gt retrievals sub: {}'.format(q_subj))
        q_mean_mpjpe_gt = get_gt_pose_ret_mean_mpjpe(q_poses, db_poses, k_list)

        print('Cal emb retrievals sub: {}'.format(q_subj))
        q_mean_mpjpe_emb, q_min_mpjpe_emb = get_emb_pose_ret_mean_mpjpe(q_emb, db_emb, q_poses, db_poses, k_list)

        for k in k_list:
            mean_mpjpe_gt[k].append(q_mean_mpjpe_gt[k])
            mean_mpjpe_emb[k].append(q_mean_mpjpe_emb[k])

    for k in k_list:
        mean_mpjpe_gt[k] = sum(mean_mpjpe_gt[k]) / len(mean_mpjpe_gt[k])
        mean_mpjpe_emb[k] = sum(mean_mpjpe_emb[k]) / len(mean_mpjpe_emb[k])
        mean_mpjpe_emb[k] = mean_mpjpe_emb[k] - mean_mpjpe_gt[k]

        print('K: {} pa_mpjpe: {}'.format(k, mean_mpjpe_emb[k]))

    return mean_mpjpe_emb



def get_mpjpe_retrieval(train_emb, train_poses, val_emb, val_poses, k_list, split, do_cross=False, file_dir='./'):

    n_samples = train_emb.shape[0]

    assert train_poses.shape[0] == n_samples
    assert val_poses.shape[0] == val_emb.shape[0]

    mean_mpjpe_emb = {}
    min_mpjpe_emb = {}
    mean_mpjpe_gt = {}

    for k in k_list:
        mean_mpjpe_emb[k] = []
        mean_mpjpe_gt[k] = []

    do_cross = do_cross
    if split == 'train':
        subj_list = [1]
    else:
        subj_list = [9, 11]

    for q_subj in subj_list[:1]:
        db_subjs = subj_list

        # getting oracle retrievals
        #annot_ids = np.arange(n_samples)

        # q_subj_mask, db_subj_mask = get_query_database_masks(annot_ids, [q_subj], db_subjs, annot)
        # q_annot_ids = annot_ids[q_subj_mask]
        # q_emb = emb[subj_mask]
        # q_poses = poses[q_annot_ids]
        # db_annot_ids = annot_ids[db_subj_mask]
        # db_emb = emb[db_subj_mask]
        # db_poses = poses[db_annot_ids]
        # print('Cal gt retrievals sub: {}'.format(q_subj))
        # q_mean_mpjpe_gt = get_gt_pose_ret_mean_mpjpe(q_poses, db_poses, k_list)

        print('Cal emb retrievals sub: {}'.format(q_subj))
        q_mean_mpjpe_emb, q_min_mpjpe_emb = get_emb_pose_ret_mean_mpjpe(val_emb, train_emb, val_poses, train_poses, k_list)

        for k in k_list:
            # mean_mpjpe_gt[k].append(q_mean_mpjpe_gt[k])
            mean_mpjpe_emb[k].append(q_mean_mpjpe_emb[k])
            # min_mpjpe_emb[k].append(q_min_mpjpe_emb[k])

    # min_mpjpe_emb = [0]*len(k_list)
    for k in k_list:
        # mean_mpjpe_gt[k] = sum(mean_mpjpe_gt[k]) / len(mean_mpjpe_gt[k])
        # import pdb; pdb.set_trace()
        min_mpjpe_emb = np.min(mean_mpjpe_emb[k])
        mean_mpjpe_emb[k] = sum(mean_mpjpe_emb[k]) / len(mean_mpjpe_emb[k])
        # mean_mpjpe_emb[k] = mean_mpjpe_emb[k] - mean_mpjpe_gt[k]

        s = 'K: {} pa_mpjpe: {}'.format(k, mean_mpjpe_emb[k]) + '\n'
        print(s)
        with open(file_dir + 'results.txt', 'a') as f:
          f.write(s)
        # print('K: {} pa_mpjpe: {}'.format(k, min_mpjpe_emb))

    return mean_mpjpe_emb, min_mpjpe_emb


def find_closest(ref, dataset):
    '''
        ref: 1 x 30 x 51
        dataset: n x 30 x 51
    '''
    sims = torch.cosine_similarity(ref, dataset, 2) # n x 30
    mean_sims = sims.mean(-1)
    max_sim = torch.argmax(mean_sims)

    return max_sim

def calc_mpjpe(ref_pose, ret_pose):
    '''
        ref_pose: 1 x 30 x 51
        ret_pose: 1 x 30 x 51
    '''
    ref_pose = ref_pose.squeeze().reshape(-1, 17, 3)
    ret_pose = ret_pose.squeeze().reshape(-1, 17, 3)

    mpjpe = ((ref_pose - ret_pose)**2).sum(-1).sqrt().mean()
    return mpjpe
    
def motion_retrieval(embs, poses, idx):
    '''
        embs: n x 30 x edim
        poses: n x 30 x 51
    '''
    embs = torch.from_numpy(embs)
    poses = torch.from_numpy(poses)
    ref = embs[idx:idx+1]
    dataset = torch.cat((embs[:idx], embs[idx+1:]))
    c_idx = find_closest(ref, dataset)
    ret_mpjpe = calc_mpjpe(poses[idx:idx+1], poses[c_idx])
    print(idx, ret_mpjpe)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='../data/annot_h36m_cam_reg/', help='data directory')
    parser.add_argument('--load_dir', type=str, help='directory to load saved embedding')
    parser.add_argument('--do_cross', action='store_true', help='use cross subject')
    parser.add_argument('--chunk_size', type=int, default=12, help='chunk size')
    parser.add_argument('--exp_id', default='default', help='name of folder containing embeddings in save_embeds')
    opt = parser.parse_args()

    k_list = [1, 2] #, 5, 10, 20]

    # For Train

    #import pdb; pdb.set_trace()
    # emb_train = np.load('train_embedding.npy')

    #emb_train = np.load('./save_embeds/mpi_embed_train_new.npy')
    #poses_train = np.load('./save_embeds/mpi_poses_train_new.npy')

    f_dir = './save_embeds/{}/'.format(opt.exp_id)

    emb_train = np.load(f_dir + 'embed_train_new.npy')
    poses_train = np.load(f_dir + 'poses_train_new.npy')
    emb_val = np.load(f_dir + 'embed_val_new.npy')
    poses_val = np.load(f_dir + 'poses_val_new.npy')

    # annot_ids_train = np.load(os.path.join(opt.load_dir, 'train_annot_ids.npy'))
    
    # assert annot_ids_train.shape[0] == emb_train.shape[0]
    
    # annot_file_train = os.path.join(opt.data_dir, 'annot_cam_train.pickle')
    
    # annot_train = load_annot_file(annot_file_train)
    
    # assert emb_train.shape[0] == annot_train['id'].shape[0]
    # down_sample_ids_train = get_down_sampled_annot_ids(annot_train, down_sample=3)
    
    # emb_train = emb_train.reshape(-1, 2048)
    print(poses_train.shape)
    print(emb_train.shape)
    print(poses_val.shape)
    print(emb_val.shape)

    #import pdb; pdb.set_trace()
    n_joints = 16   # 16 for human3.6, 17 for MPI-INF

    if len(poses_train.shape)==3:
      b, n, _ = poses_train.shape
      b_val, n_val, _ = poses_val.shape
    else:
      b, n, _, _ = poses_train.shape
      b_val, n_val, _, _ = poses_val.shape

    #poses_train = poses_train.reshape(b, n, n_joints, 3)
    #poses_train = poses_train - poses_train[:, :, 14:15]

    #for i in range(900):
    #    motion_retrieval(emb_train, poses_train, i)

    _, _, fsz = emb_train.shape
    emb_train = emb_train.reshape(b*n, fsz)  #2048)
    emb_val = emb_val.reshape(b_val*n_val, fsz)

    mask_train = emb_train == 99999*np.ones((2048))
    mask_train = mask_train[:, 0]
    emb_train = emb_train[~mask_train]

    mask_val = emb_val == 99999*np.ones((2048))
    mask_val = mask_val[:, 0]
    emb_val = emb_val[~mask_val]

    #poses_train = poses_train.reshape(b, n, n_joints*3)
    poses_train = poses_train.reshape(b*n, n_joints*3)
    poses_train = poses_train.reshape(b*n, n_joints, 3)
    poses_train = poses_train[~mask_train]
    #poses_train = poses_train - poses_train[:, 14:15]

    poses_val = poses_val.reshape(b_val*n_val, n_joints*3)
    poses_val = poses_val.reshape(b_val*n_val, n_joints, 3)
    poses_val = poses_val[~mask_val]
    # poses_train = poses_train.reshape(-1, 51)

    print("after masking off padding")
    print("Train embedding : ", emb_train.shape)
    print("Val embedding : ", emb_val.shape)

    
    print('downsampling data')
    emb_train_down = emb_train  #[::10]
    emb_val_down = emb_val[::10]    #[::3]
    poses_train_down = poses_train  #[::10]
    poses_val_down = poses_val[::10]  #[::3]
    emb_train_down = emb_train_down  #[:18000] #6000]
    emb_val_down = emb_val_down  #[:18000] #6000]
    poses_train_down = poses_train_down   #[:18000] #6000]
    poses_val_down = poses_val_down   #[:18000] #6000]
   
    # annot_train_down = get_down_sampled_annot(annot_train, down_sample_ids_train)
    
    cam_pair_count = 0
    
    print('retrieving')
    mean_mpjpe_emb_train, min_mpjpe_emb_train = get_mpjpe_retrieval(emb_train_down, poses_train_down, emb_val_down, poses_val_down, k_list, 'train', file_dir=f_dir)

    '''
    For Test
    emb_test = np.load(os.path.join(opt.load_dir, 'test_pred.npy'))
    annot_ids_test = np.load(os.path.join(opt.load_dir, 'test_annot_ids.npy'))

    assert annot_ids_test.shape[0] == emb_test.shape[0]

    print('==> initializing H36M {} data.'.format('test'))
    print('data dir {}'.format(opt.data_dir))
    annot_file_test = os.path.join(opt.data_dir, 'annot_cam_test.pickle')

    annot_test = load_annot_file(annot_file_test)

    assert emb_test.shape[0] == annot_test['id'].shape[0]

    down_sample_ids_test = get_down_sampled_annot_ids(annot_test, down_sample=3)
    annot_test_down = get_down_sampled_annot(annot_test, down_sample_ids_test)

    emb_test_down = emb_test[down_sample_ids_test, :]

    cam_pair_count = 0

    mean_mpjpe_emb_test = get_mpjpe_retrieval(opt, emb_test_down, annot_test_down, k_list, 'test')
    '''

if __name__ == '__main__':
    main()
