import torch
import numpy as np
from common.camera_utils import rotate_emb


def process_batch_chunks_reg(opt, batch_size_reg, inp_reg, tar_3d, tar_3d_un, meta, n_joints):
    subject_reg = meta['subj_reg'].view(batch_size_reg * opt.chunk_size)

    tar_3d_norm = tar_3d.view(batch_size_reg, opt.chunk_size, -1)
    tar_3d_un_norm = tar_3d_un.view(batch_size_reg, opt.chunk_size, -1)
    # tar_3d_un_flat = tar_3d_un.view(batch_size_reg, opt.chunk_size, n_joints * 3).numpy()

    inp_reg_ten = inp_reg.permute(0, 3, 1, 2).float()  # .to(torch.device("cuda:0"))

    subject_reg_ten = subject_reg.long()

    return inp_reg_ten, tar_3d_norm, tar_3d_un_norm, subject_reg_ten


# def process_batch_chunks_met(opt, batch_size_met, batch_ratio, inp_met, meta):

#     n_channels = inp_met.shape[4]
#     height = inp_met.shape[2]
#     width = inp_met.shape[3]

#     inp_met = inp_met.view(batch_size_met * opt.chunk_size * batch_ratio * 2,
#                            height, width, n_channels)

#     img_met_cp = meta['img_met'].view(batch_size_met * opt.chunk_size * batch_ratio * 2,
#                                       height, width, n_channels)
#     subject_met = meta['subj_met'].view(batch_size_met * opt.chunk_size * batch_ratio * 2)
#     unq_id_met = meta['unq_id_met'].view(batch_size_met * opt.chunk_size * batch_ratio * 2)
#     pose_un_met = meta['pose_un_met'].view(batch_size_met * opt.chunk_size * batch_ratio * 2, 16, 3)
#     pose_met = meta['pose_met'].view(batch_size_met * opt.chunk_size * batch_ratio * 2, 16, 3)
#     cam_id_met = meta['cam_id_met'].view(batch_size_met * opt.chunk_size * batch_ratio * 2)
#     cam_rot_met = meta['cam_rot_met'].view(batch_size_met * opt.chunk_size * batch_ratio * 2, 4)

#     subject_met_ten = subject_met.long()
#     cam_id_met_ten = cam_id_met.long()

#     # separating views for weak supervision
#     inp_a_met = torch.FloatTensor(batch_size_met * opt.chunk_size * batch_ratio,
#                                   height, width, n_channels)
#     inp_p_met = torch.FloatTensor(batch_size_met * opt.chunk_size * batch_ratio,
#                                   height, width, n_channels)
#     inp_a_met_cp = torch.FloatTensor(batch_size_met * batch_ratio, opt.chunk_size,
#                                      height, width, n_channels)
#     inp_p_met_cp = torch.FloatTensor(batch_size_met * batch_ratio, opt.chunk_size,
#                                      height, width, n_channels)

#     pose_a_met = torch.FloatTensor(batch_size_met * batch_ratio, opt.chunk_size, 16, 3)
#     pose_p_met = torch.FloatTensor(batch_size_met * batch_ratio, opt.chunk_size, 16, 3)

#     subj_met_a = torch.IntTensor(batch_size_met * batch_ratio, opt.chunk_size)
#     subj_met_p = torch.IntTensor(batch_size_met * batch_ratio, opt.chunk_size)

#     cam_rot_met_a = torch.FloatTensor(batch_size_met * batch_ratio, opt.chunk_size, 4)
#     cam_rot_met_p = torch.FloatTensor(batch_size_met * batch_ratio, opt.chunk_size, 4)

#     chunk_count = 0
#     for b in range(0, batch_size_met * 2 * opt.chunk_size * batch_ratio, 2 * opt.chunk_size):
#         inp_a_met[chunk_count * opt.chunk_size:(chunk_count + 1) * opt.chunk_size, :, :, :] = \
#             inp_met[b:b + opt.chunk_size, :, :, :]
#         inp_a_met_cp[chunk_count, :, :, :, :] = \
#             img_met_cp[b:b + opt.chunk_size, :, :, :]

#         pose_a_met[chunk_count, :, :, :] = pose_met[b:b + opt.chunk_size]
#         subj_met_a[chunk_count, :, ] = subject_met[b:b + opt.chunk_size]
#         cam_rot_met_a[chunk_count, :, :] = cam_rot_met[b:b + opt.chunk_size]

#         inp_p_met[chunk_count * opt.chunk_size:(chunk_count + 1) * opt.chunk_size, :, :, :] = \
#             inp_met[b + opt.chunk_size:b + 2 * opt.chunk_size, :, :, :]
#         inp_p_met_cp[chunk_count, :, :, :, :] = \
#             img_met_cp[b + opt.chunk_size:b + 2 * opt.chunk_size, :, :, :]

#         pose_p_met[chunk_count, :, :, :] = pose_met[b + opt.chunk_size:b + 2 * opt.chunk_size]
#         subj_met_p[chunk_count, :, ] = subject_met[b + opt.chunk_size:b + 2 * opt.chunk_size]
#         cam_rot_met_p[chunk_count, :, :] = cam_rot_met[b + opt.chunk_size:b + 2 * opt.chunk_size]

#         chunk_count = chunk_count + 1

#     assert chunk_count == batch_size_met * batch_ratio

#     return inp_a_met, inp_p_met, inp_a_met_cp, inp_p_met_cp, pose_a_met, pose_p_met, \
#            subj_met_a, subj_met_p, cam_rot_met_a, cam_rot_met_p


def process_batch_chunks_met(opt, batch_size_met, batch_ratio, inp_met, meta):

    n_channels = inp_met.shape[4]
    height = inp_met.shape[2]
    width = inp_met.shape[3]
    chunk_size = opt.chunk_size

    inp_met = inp_met.view(-1, 2, height, width, n_channels)
    img_met_cp = meta['img_met'].view(-1, 2, height, width, n_channels)

    subject_met = meta['subj_met'].view(-1, 2)
    unq_id_met = meta['unq_id_met'].view(-1, 2)
    pose_un_met = meta['pose_un_met'].view(-1, 2, 16, 3)
    pose_met = meta['pose_met'].view(-1, 2, 16, 3)
    cam_id_met = meta['cam_id_met'].view(-1, 2)
    cam_rot_met = meta['cam_rot_met'].view(-1, 2, 4)

    subject_met_ten = subject_met.long()
    cam_id_met_ten = cam_id_met.long()

    # separating views for weak supervision
    inp_a_met = inp_met[:, 0, :, :, :]
    inp_p_met = inp_met[:, 1, :, :, :]
    
    img_a_met_cp = img_met_cp[:, 0, :, :, :].view(-1, chunk_size, height, width, n_channels)
    img_p_met_cp = img_met_cp[:, 1, :, :, :].view(-1, chunk_size, height, width, n_channels)

    pose_a_met = pose_met[:, 0, :, :].view(-1, chunk_size, 16, 3)
    pose_p_met = pose_met[:, 1, :, :].view(-1, chunk_size, 16, 3)

    pose_un_a_met = pose_un_met[:, 0, :, :].view(-1, chunk_size, 16, 3)
    pose_un_p_met = pose_un_met[:, 1, :, :].view(-1, chunk_size, 16, 3)

    subj_met_a = subject_met[:, 0].view(-1, chunk_size)
    subj_met_p = subject_met[:, 1].view(-1, chunk_size)

    cam_id_met_a = cam_id_met[:, 0].view(-1, chunk_size)
    cam_id_met_p = cam_id_met[:, 1].view(-1, chunk_size)
    
    cam_rot_met_a = cam_rot_met[:, 0, :].view(-1, chunk_size, 4)
    cam_rot_met_p = cam_rot_met[:, 1, :].view(-1, chunk_size, 4)

    chunk_count = 0
    for b in range(0, batch_size_met * batch_ratio):
        assert subj_met_a[b, 0] == subj_met_a[b, chunk_size-1] 
        assert subj_met_p[b, 0] == subj_met_p[b, chunk_size-1]
        assert subj_met_a[b, 0] == subj_met_p[b, 0]

        pose_a_glb = rotate_emb(pose_un_a_met[b, 0:1, :, :], cam_rot_met_a[b, 0:1, :])
        pose_p_glb = rotate_emb(pose_un_p_met[b, 0:1, :, :], cam_rot_met_p[b, 0:1, :])
    
        assert torch.abs((pose_a_glb - pose_p_glb)).max() < 1e-01
    
    
    meta_out = dict()
    
    meta_out['img_a_met_cp'] = img_a_met_cp
    meta_out['img_p_met_cp'] = img_p_met_cp
    
    meta_out['subj_met_a'] = subj_met_a
    meta_out['subj_met_p'] = subj_met_p

    meta_out['cam_id_met_a'] = cam_id_met_a
    meta_out['cam_id_met_p'] = cam_id_met_p

    return inp_a_met, inp_p_met, pose_a_met, pose_p_met, \
            pose_un_a_met, pose_un_p_met, cam_rot_met_a, cam_rot_met_p, meta_out
