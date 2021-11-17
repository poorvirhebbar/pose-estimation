import numpy as np
from utils.lr_scheduler import WarmupMultiStepLR
from utils.average_meter import AverageMeter
from opts import opts
import sys
import torch
import transformers as tf
from transformers.modeling_bert import BertEncoder, BertEmbeddings, BertModel
from tqdm import tqdm 
from model_posebert import PoseBert
from data_loader.h36m_basic_noisy import H36M

def mask_lifter_inputs(cfg, poses_2d, poses_3d):
    '''
        Make the first element CLS character and mask out frames just
        like in BERT
    '''
    pose_dim = poses_2d.shape[-1]
    mask_token = 0.0
    label_masks = torch.zeros(poses_2d[..., 0:1].shape)
    probability_matrix = torch.full(poses_2d[..., 0].shape, cfg.mask_prob).to(poses_2d.device)
    masked_idx = torch.bernoulli(probability_matrix).bool()
    label_masks[masked_idx] = 1.0
    labels = poses_3d * label_masks

    indices_replaced = torch.bernoulli(torch.full(poses_2d[..., 0].shape, 0.8)).bool() & masked_idx
    poses_2d[indices_replaced] = mask_token

    indices_random = torch.bernoulli(torch.full(poses_2d[..., 0].shape, 0.5)).bool() & masked_idx & (~indices_replaced.byte()).bool()
    random_frames = torch.randint(poses_2d[..., 0].numel(), poses_2d[..., 0].shape, dtype=torch.long)
    poses_2d[indices_random] = poses_2d.view(-1, pose_dim)[random_frames[indices_random]]


    return poses_2d, labels, label_masks

def mask_poses(cfg, poses):
    '''
        Make the first element CLS character and mask out frames just
        like in BERT
    '''
    pose_dim = poses.shape[-1]
    mask_token = 0.0
    label_masks = torch.zeros(poses[..., 0:1].shape)
    probability_matrix = torch.full(poses[..., 0].shape, cfg.mask_prob).to(poses.device)
    masked_idx = torch.bernoulli(probability_matrix).bool()
    label_masks[masked_idx] = 1.0
    labels = poses * label_masks

    indices_replaced = torch.bernoulli(torch.full(poses[..., 0].shape, 0.8)).bool() & masked_idx
    poses[indices_replaced] = mask_token

    indices_random = torch.bernoulli(torch.full(poses[..., 0].shape, 0.5)).bool() & masked_idx & (~indices_replaced.byte()).bool()
    random_frames = torch.randint(poses[..., 0].numel(), poses[..., 0].shape, dtype=torch.long)
    poses[indices_random] = poses.view(-1, pose_dim)[random_frames[indices_random]]


    return poses, labels, label_masks

def flip_poses(poses_2d, poses_3d):
    B, K, _ = poses_2d.shape
    flip_2d = poses_2d.reshape(B, K, -1, 2).clone()
    flip_3d = poses_3d.reshape(B, K, -1, 3).clone()

    flip_2d[..., 0] *= -1
    flip_3d[..., [0,2]] *= -1

    return flip_2d.reshape(B, K, -1), flip_3d.reshape(B, K, -1)


def get_mpjpe(preds, gt, masks):
    B, K, _ = preds.shape
    preds = preds.reshape(B, K, -1, 3)
    gt = gt.reshape(B, K, -1, 3)
    diff = ((preds - gt)**2).sum(-1).sqrt() * masks # .reshape(B, K, -1, 3)
    mpjpe = diff.sum() / masks.sum() / 16

    return mpjpe

def step(model, data_loader, criterion, optimizer, split, scheduler=None, noise=False):
    if split == 'train':
        model.train()
    else:
        model.eval()

    Loss = AverageMeter()
    MPJPE = AverageMeter()
    for i, meta in enumerate(tqdm(data_loader, ascii=True)):
        pose_3d = meta['pose_un'].squeeze().float() / 1000
        pose_2d = meta['pose_2d'].squeeze().float() / 112
        if noise:
            noisy_2d = meta['noisy_2d'].squeeze().float() / 112
            noisy_3d_gt = meta['noisy_3d_gt'].squeeze().float() / 1000
            positions = torch.Tensor(list(range(1,K+1))).unsqueeze(0).expand(2*B, K).long().cuda()
            if split == 'train':
                pose_2d = torch.cat((pose_2d, noisy_2d))
                pose_3d = torch.cat((pose_3d, noisy_3d_gt))
            else:
                pose_2d = noisy_2d
                positions = torch.Tensor(list(range(1,K+1))).unsqueeze(0).expand(B, K).long().cuda() 
        else:
            positions = torch.Tensor(list(range(1,K+1))).unsqueeze(0).expand(B, K).long().cuda() 

        if split == 'train':
            flip_2d, flip_3d = flip_poses(pose_2d, pose_3d)
            pose_2d = torch.cat((pose_2d, flip_2d))
            pose_3d = torch.cat((pose_3d, flip_3d)) 
            positions = torch.cat((positions, positions))

        pose_3d = pose_3d.reshape(-1, K, 48)
        pose_2d = pose_2d.reshape(-1, K, 32)
        average = pose_3d.mean(1).unsqueeze(1).expand_as(pose_3d)
        # pose_3d = pose_2d.reshape(-1, K, J+1, 3)
        # pose_3d = pose_2d.transpose(1,0,2,3)
        # pose_2d = torch.cat(pose_2d[:,0], pose_2d[:,1], pose_2d[:,2], pose_2d[:,3]).cuda()

        # input, labels, label_masks = mask_poses(cfg, pose_3d)
        input, labels, label_masks = mask_lifter_inputs(cfg, pose_2d, pose_3d)
        input = input.cuda()
        labels = labels.cuda()
        label_masks = label_masks.cuda()
        pose_2d = pose_2d.cuda()
        pose_3d = pose_3d.cuda()
        # import pdb; pdb.set_trace()

        # output = model(position_ids=positions,
        #                inputs_embeds=input)

        output = model(pose_2d, positions)

        # output = output[0] * label_masks

        # loss = criterion(output, labels)
        loss = criterion(output, pose_3d)
        
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            scheduler.step()

        Loss.update(loss.detach().cpu().numpy())
        # mpjpe = get_mpjpe(output[0].detach().cpu(), labels.detach().cpu(), label_masks.detach().cpu())
        mpjpe = get_mpjpe(output.detach().cpu(), pose_3d.detach().cpu(), 1 + 0.0*label_masks.detach().cpu())
        MPJPE.update(mpjpe.numpy())

    print(f'{split} epoch: {epoch} | iter: {i} | loss: {Loss.avg} | mpjpe: {MPJPE.avg}')


if __name__ == "__main__": 
    B = 24 
    K = 16
    cfg = opts().parse()
    dataset_train = H36M(cfg, split='train',
                    train_stats={}, K=K,
                    allowed_subj_list_reg=[1,5,6,7,8])

    data_loader = torch.utils.data.DataLoader(
                    dataset_train, batch_size=B*K,
                    shuffle=False, num_workers=4, drop_last=True)
    
    dataset_val = H36M(cfg, split='val',
                    train_stats={}, K=K,
                    allowed_subj_list_reg=[9,11])

    data_loader_val = torch.utils.data.DataLoader(
                    dataset_val, batch_size=B*K,
                    shuffle=False, num_workers=4, drop_last=True)

    config = tf.BertConfig(vocab_size=48)
    # model = BertModel(config).cuda()
    model = PoseBert(config).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.MSELoss().cuda()
    milestones = [30, 70]
    milestones = [60000, 140000]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.25)
    scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=0.25, warmup_factor=1.0 / 3,
                                 warmup_iters=2500, last_epoch=-1)


    for epoch in range(cfg.n_epochs): 
        step(model, data_loader, criterion, optimizer, split='train', scheduler=scheduler, noise=False)
        if (epoch + 1) % cfg.val_intervals == 0:
            step(model, data_loader_val, criterion, optimizer, split='val', noise=False)
            print( '\n' )
            
