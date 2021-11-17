import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import cv2

edges = [[0,1], [1,2], [2,6], [6,3], [3,4], [4,5],
         [6,7], [7,8], [8,9], [10,11], [11,12], [12,8],
         [13,8], [13,14], [14,15]]

def vizualize(heatmaps,idx, img=None):
    """
        heatmaps: n_joints x hm_w x hm_h
    """
    hm = heatmaps.detach().cpu().numpy()
    count = 1
    if img:
        img = img.cpu().numpy().transpose(1,2,0)
    for i in range(heatmaps.shape[0]):
        ax1 = plt.subplot(8,4,count)
        ax1.imshow(hm[i])
        # ax1.scatter(pose_2d[i, 0], pose_2d[i, 1], marker='x', c='white')
        # ax1.scatter(gt_2d[i, 0], gt_2d[i, 1], marker='+', c='red')
        # save_dir = os.path.join('./exp', cfg.exp_id, 'vis/')
        # os.system(f'mkdir -p {save_dir}')
        # plt.savefig(os.path.join(save_dir, f'hm_{i}.png'))
        count += 1
    save_path = os.path.join(f'./viz/viz_{idx}.png')
    plt.savefig(save_path)
    plt.close()

def overlay(cfg, img, heatmaps, pose_2d, gt_2d, idx):
    hm = heatmaps.detach().cpu().numpy()
    pose_2d = 4 * pose_2d.detach().cpu().numpy()
    gt_2d = 4 * gt_2d.cpu().numpy()
    img = img.cpu().numpy().transpose(1,2,0)
    img = img.sum(-1)
    count = 1
    for i in range(pose_2d.shape[0]):
        ax = plt.subplot(4,4,count)
        hm_i = cv2.resize(hm[i], (256, 256))
        show_img = 0.5*img + 0.02*hm_i
        ax.imshow(show_img)
        ax.scatter(pose_2d[i, 0], pose_2d[i, 1], marker='x', c='white')
        ax.scatter(gt_2d[i, 0], gt_2d[i, 1], marker='+', c='red')
        # save_dir = os.path.join('./exp', cfg.exp_id, 'vis/')
        # os.system(f'mkdir -p {save_dir}')
        # plt.savefig(os.path.join(save_dir, f'hm_{i}.png'))
        count += 1
    save_dir = os.path.join('./exp', cfg.exp_id, 'visualize/')
    plt.savefig(os.path.join(save_dir, f'img_overlay_{idx}.png'))
    plt.close()

def viz_img(img, idx):
    #img = img.detach().cpu().numpy().transpose(1,2,0)
    save_dir = os.path.join('./imgs')
    plt.imshow(img)
    plt.savefig(os.path.join(save_dir, f'img_{idx}.png'))

def viz_2d(cfg, img, pose_2d, idx):
    # pose_2d = 4*pose_2d.detach().cpu().numpy()
    # gt_2d = 4*gt_2d.cpu().numpy()
    #img = img.cpu().numpy().transpose(1,2,0)

    plt.imshow(img)
    plt.scatter(pose_2d[:,0], pose_2d[:,1], marker = 'o', c='red')

    for edge in edges:
        p1 = edge[0]
        p2 = edge[1]
        plt.plot([pose_2d[p1,0], pose_2d[p2,0]], [pose_2d[p1,1], pose_2d[p2,1]])

    plt.savefig(f'./viz/pose_overlay_{idx}.png')
    plt.close()
