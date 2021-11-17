import torch
import numpy as np


def un_normalise_pose(normalised_pose, mean, std):
    # accepts numpy arrays

    chunk_size = normalised_pose.shape[0]

    mean_chunk = mean.repeat(chunk_size, 1)
    std_chunk = std.repeat(chunk_size, 1)

    if normalised_pose.is_cuda is True:
        mean_chunk = mean_chunk.to(torch.device("cuda:0"))
        std_chunk = std_chunk.to(torch.device("cuda:0"))

    un_norm_pose = torch.mul(normalised_pose, std_chunk) + mean_chunk

    return un_norm_pose


def normalise_pose(un_normalised_pose, mean, std):
    e = 1e-04

    chunk_size = un_normalised_pose.shape[0]

    mean_chunk = mean.repeat(chunk_size, 1)
    std_chunk = std.repeat(chunk_size, 1)

    if un_normalised_pose.is_cuda is True:
        mean_chunk = mean_chunk.to(torch.device("cuda:0"))
        std_chunk = std_chunk.to(torch.device("cuda:0"))

    norm_pose = torch.div(un_normalised_pose - mean_chunk, std_chunk + e)

    return norm_pose


def cal_avg_l2_jnt_dist(pose_1, pose_2, avg=True):
    n_joints = pose_1.shape[1]
    
    # if n_joints > 16:
    #     n_joints = n_joints // 3
    batch_size = pose_1.shape[0]

    pose_1 = np.copy(pose_1).reshape(batch_size, n_joints, 3)
    pose_2 = np.copy(pose_2).reshape(batch_size, n_joints, 3)

    diff = pose_1-pose_2

    diff_sq = diff ** 2

    dist_per_joint = np.sqrt(np.sum(diff_sq, axis=2))

    dist_per_sample = np.average(dist_per_joint, axis=1)

    if avg is True:
        dist_avg = np.average(dist_per_sample)
    else:
        dist_avg = dist_per_sample

    return dist_avg


def scale_norm_pose(pred, label):
    batch_size = pred.shape[0]

    pred_vec = pred.view(batch_size, -1)
    gt_vec = label.view(batch_size, -1)
    dot_pose_pose = torch.sum(torch.mul(pred_vec, pred_vec), 1, keepdim=True)
    dot_pose_gt = torch.sum(torch.mul(pred_vec, gt_vec), 1, keepdim=True)

    s_opt = dot_pose_gt / dot_pose_pose

    return s_opt.expand_as(pred) * pred


def cal_p_mpjpe(predicted, target, avg=True):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    n_joints = predicted.shape[1]
    if n_joints > 16:
        n_joints = n_joints // 3
    batch_size = predicted.shape[0]

    predicted = np.copy(predicted).reshape(batch_size, n_joints, 3)
    target = np.copy(target).reshape(batch_size, n_joints, 3)

    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    pmpjpe_batch = np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=1)

    if avg is True:
        return np.mean(pmpjpe_batch)
    else:
        return pmpjpe_batch


def get_noisy_gt(pose, mean, std, factor=0.1):
    batch_size = pose.shape[0]

    pose_flat = pose.view(batch_size, -1)

    dim_size = pose_flat.shape[1]

    max_noise = factor * torch.ones(1, dim_size).float()

    # rand_noise = torch.rand(batch_size, dim_size) - 0.5

    rand_noise = 2 * torch.bernoulli(torch.rand(batch_size, dim_size)).float() - 1

    rand_noise = torch.mul(rand_noise, max_noise.repeat(batch_size, 1))

    pose_noise = pose_flat + rand_noise

    return pose_noise
