import numpy as np
from utils.lr_scheduler import WarmupMultiStepLR
from utils.average_meter import AverageMeter
from opts import opts
import sys
import torch
import transformers as tf
from transformers.modeling_bert import BertEncoder, BertEmbeddings, BertModel, BertForMaskedLM
from tqdm import tqdm 
from model_posebert import PoseBert
from dataloader.ntu_rgbd import NTURGBD
from dataloader.mpi_inf import MPI_INF
from ntu_dataset import fetch_dataloader

## IGNORE THIS FUNCTION
def mask_tokens( cfg, inputs):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, cfg.mask_prob)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = 2000

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(2000, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


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

def get_mpjpe(preds, gt, masks, pad_vec):
    # import pdb; pdb.set_trace()
    B, K, _ = preds.shape
    preds = preds.reshape(B, K, -1, 3)
    gt = gt.reshape(B, K, -1, 3)
    diff = ((preds - gt)**2).sum(-1).sqrt() * masks # .reshape(B, K, -1, 3)
    mpjpe = diff.sum() / masks.sum() / 25

    return mpjpe

def step(model, data_loader, criterion, optimizer, split, scheduler=None):
    if split == 'train':
        model.train()
    else:
        model.eval()

    Loss = AverageMeter()
    MPJPE = AverageMeter()
    for i, (pose_3d, pad_vec) in enumerate(tqdm(data_loader, ascii=True)): 
        pose_3d = pose_3d - pose_3d[:, :, 14:15, :]
        pose_3d = pose_3d.reshape(B, K, 17*3).float() / 1000
        '''
        labels = meta['labels'].long()
        labels = labels.reshape(B, K)
        c_cent = meta['cluster_centers'][0]
        # pose_3d = pose_2d.reshape(-1, K, J+1, 3)
        # pose_3d = pose_2d.transpose(1,0,2,3)
        # pose_2d = torch.cat(pose_2d[:,0], pose_2d[:,1], pose_2d[:,2], pose_2d[:,3]).cuda()
        input, labels = mask_tokens(cfg, labels)
        label_masks = 1 - (input < 2000).float()
        '''

        input, labels, label_masks = mask_poses(cfg, pose_3d.clone())
        input = input.cuda()
        labels = labels.cuda()
        label_masks = label_masks.cuda()
        # import pdb; pdb.set_trace()

        output, embedding = model(input, positions)
        output = output * label_masks

        loss = criterion(output, labels)
        
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            scheduler.step()

        Loss.update(loss.detach().cpu().numpy())
        mpjpe = get_mpjpe(output.detach().cpu(), pose_3d.detach().cpu(), 
                            label_masks.detach().cpu(), pad_vec)
        MPJPE.update(mpjpe.numpy())

        if cfg.DEBUG > 0 and i % cfg.display == 0:
            print(f'{split} epoch: {epoch} | iter: {i} | loss: {Loss.avg} | mpjpe: {MPJPE.avg}')

    print(f'{split} epoch: {epoch} | iter: {i} | loss: {Loss.avg} | mpjpe: {MPJPE.avg}')

if __name__ == "__main__": 
    cfg = opts().parse()
    B = cfg.train_batch
    K = cfg.max_len

    mpi_train  = MPI_INF('/home/datasets/mpi-inf-3dhp/mpi_inf_3dhp/', 
                            split='train', max_length=cfg.max_len)
    dataloader_train = torch.utils.data.DataLoader(mpi_train, 
                                        batch_size=cfg.train_batch, 
                                        shuffle=True, drop_last=True)
    # dataloader_train = fetch_dataloader('train', cfg)
    # dataloader_val = fetch_dataloader('test', cfg)

    mpi_val  = MPI_INF('/home/datasets/mpi-inf-3dhp/mpi_inf_3dhp/', 
                            split='val', max_length=cfg.max_len)
    dataloader_val = torch.utils.data.DataLoader(mpi_val, batch_size=B, 
                                        shuffle=True, drop_last=True)

    config = tf.BertConfig(vocab_size=51, num_hidden_layers=2)
    # model = BertModel(config).cuda()
    # model = BertForMaskedLM(config).cuda()
    model = PoseBert(config, n_joints=17).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.0001)
    criterion = torch.nn.MSELoss().cuda()
    milestones = [40000, 80000, 120000, 160000]
    scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=0.25, warmup_factor=1.0 / 3,
                                 warmup_iters=1500, last_epoch=-1)

    positions = torch.Tensor(list(range(1,K+1))).unsqueeze(0).expand(B, K).long().cuda()

    for epoch in range(cfg.n_epochs): 
        step(model, dataloader_train, criterion, optimizer, split='train', scheduler=scheduler)
        if (epoch + 1) % cfg.val_intervals == 0:
            step(model, dataloader_val, criterion, optimizer, split='val')
        if (epoch + 1) % cfg.save_intervals == 0:
            torch.save(model, f'exp/model_{epoch}.pth')

            
