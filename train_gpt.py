import numpy as np
from opts import opts
from utils.lr_scheduler import WarmupMultiStepLR
from torch.optim.lr_scheduler import StepLR
from utils.average_meter import AverageMeter
import sys
import os
from logger import Logger
import torch
from tqdm import tqdm
from mpi_inf import MPI_INF
from model_gpt import GPT2

def get_mpjpe(preds, gt, nJoints):
    # preds.shape == [bsz, seq_len, 17, 3]
    return (((preds - gt)**2).sum(-1).sqrt() ).mean()

def step(model, data_loader, criterion, optimizer, split, scheduler=None, get_embed=False):    # for MPI_INF
    if split == 'train':
        model.train()
    else:
        model.eval()

    Loss = AverageMeter()
    MPJPE = AverageMeter()

    poses, embeddings = [], []
    p = 0.95
    rand =  torch.bernoulli(torch.tensor(p))

    for i, (img_embeds, pose_3d) in enumerate(tqdm(data_loader, ascii=True)):
        #import pdb; pdb.set_trace()
        # pose_3d = meta['pose_un'] / 1000
        # Shape is n_frames x 48... 80 frames, 16 joints, 3d coords

        pose_3d = pose_3d.reshape(*pose_3d.shape[:2], nJoints, 3)
        pose_3d = pose_3d - pose_3d[:, :, 14:15]

        img_embeds = img_embeds.float()
        pose_3d = pose_3d[:, 301:350].float().cuda()
        pose_3d = pose_3d.reshape(*pose_3d.shape[:2], -1)

        source_seq = img_embeds[:, :300].cuda()
        target_seq = img_embeds[:, 300:].cuda()
        target_pose = torch.zeros((pose_3d.size(0), pose_3d.size(1), nJoints*3)).cuda()

        for j in range(pose_3d.size(1)):
          if j==0:
            output_embed, pose, past_key_val = model( source_seq, None)
          else:
            if epoch % 10==0:
              p *= 0.9
            rand = torch.bernoulli(torch.tensor(p))
            if rand==0:
              output_embed, pose, past_key_val = model( output_embed, past_key_val)  # feeding its own output
            else:
              output_embed, pose, past_key_val = model( target_seq[:, j].unsqueeze(1), past_key_val) # feeding GT embed

          output_embed = output_embed.unsqueeze(1).detach()
          target_pose[:, j] = pose

        loss = criterion(target_pose, pose_3d)
        #loss = contrastive_loss + 0.1 * diversity_loss
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        import pdb; pdb.set_trace()
        Loss.update( loss.detach().cpu().numpy())
        mpjpe = get_mpjpe( target_pose.detach().cpu(), pose_3d.detach().cpu(), nJoints)
        MPJPE.update(mpjpe.numpy())

        if cfg.DEBUG > 0 and (i+1) % cfg.display == 0:
            print(f'{split} epoch: {epoch} | iter: {i} | loss: {Loss.avg} | mpjpe: {MPJPE.avg}')

    if get_embed:
        #import pdb; pdb.set_trace()
        #poses_npy = torch.cat(poses).numpy()
        embed = torch.cat(embeddings, 0)
        poses = torch.cat(poses, 0)
        #np.save(split+'_embed.npy', embed_npy)
        return embed, poses

    s = f'{split} epoch: {epoch} | iter: {i} | loss: {Loss.avg} | mpjpe: {MPJPE.avg}\n'
    return s, MPJPE.avg



if __name__ == "__main__":
    cfg = opts().parse()
    B = cfg.train_batch
    seq_len = cfg.seq_len

    if cfg.dataset=='h36m':
        nJoints = 17
        dataset_train = H36M(split='train', max_len=seq_len)
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=B, shuffle=True)

        dataset_val = H36M(split='val', max_len=seq_len)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=B, drop_last=True)

    else:

        nJoints = 17
        dataset_train = MPI_INF("./", split='train', max_length=seq_len)
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=B, shuffle=True)

        dataset_val = MPI_INF("./", split='val', max_length=seq_len)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=B)

    if cfg.load_model is not None:
        model = torch.load(cfg.load_model)
        print("Loaded model : ", cfg.load_model)
    else:
        model = GPT2(n_decoders=cfg.n_decoders, nJoints=17).cuda()

    positions = torch.Tensor(list(range(1, seq_len+1))).unsqueeze(0).expand(B, seq_len).long().cuda()

    '''
    if cfg.get_embed:
        print('getting embeddings')
        embeds_train, embeds_val, poses_train, poses_val = [], [], [], []
        for i in range(5):
            #import pdb; pdb.set_trace()
            embeds, poses = step_h36m(model, dataloader_train, None, None, 'train', None, True)
            embeds_train.append(embeds)
            poses_train.append(poses)
            embeds_v, poses_v = step_h36m(model, dataloader_val, None, None, 'val', None, True)
            embeds_val.append(embeds_v)
            poses_val.append(poses_v)

        embeds_train = torch.cat(embeds_train, 0).numpy()
        embeds_val = torch.cat(embeds_val, 0).numpy()
        poses_train = torch.cat(poses_train, 0).numpy()
        poses_val = torch.cat(poses_val, 0).numpy()

        print(embeds_train.shape)
        print(poses_train.shape)

        np.save('save_embeds/mpi_poses_train.npy', poses_train)
        np.save('save_embeds/mpi_poses_val.npy', poses_val)
        np.save('save_embeds/mpi_embed_train.npy', embeds_train)
        np.save('save_embeds/mpi_embed_val.npy', embeds_val)

        exit()
    '''

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.0001)
    criterion = torch.nn.MSELoss().cuda()
    #milestones = [40000, 80000, 120000]
    #scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=0.25, warmup_factor=1.0 / 3,
    #                             warmup_iters=1500, last_epoch=-1)
    scheduler = StepLR( optimizer, step_size=50, gamma=0.8)

    #positions = torch.Tensor(list(range(1,seq_len+1))).unsqueeze(0).expand(4, seq_len).long().cuda()
    logger = Logger('exp/'+str(cfg.exp_id))

    with open('exp/{}/opts.txt'.format(cfg.exp_id), 'a') as f:
      d = vars(cfg)
      for key in d:
        f.write(str(key) + ' : '+str(d[key]) + '\n')
      f.write('\n\n')

    best_val_mpjpe = 9999999
    for epoch in range(cfg.n_epochs):
        flag = 0
        if cfg.dataset=='h36m':
            s = step_h36m(model, dataloader_train, criterion, optimizer, split='train', scheduler=scheduler)
        else:
            s, mpjpe = step(model, dataloader_train, criterion, optimizer, split='train', scheduler=scheduler)
        logger.write(s, 'train')
        

        if (epoch + 1) % cfg.val_intervals == 0:
            if cfg.dataset=='h36m':
                s = step_h36m(model, dataloader_val, criterion, optimizer, split='val')
            else:
                s, val_mpjpe = step(model, dataloader_val, criterion, optimizer, split='val')
            logger.write(s, 'val')
            if val_mpjpe < best_val_mpjpe:
                best_val_mpjpe = val_mpjpe
                flag = 1
        if (epoch + 1) % cfg.save_intervals == 0 or flag==1:
            torch.save(model, f'exp/{cfg.exp_id}/model_{epoch}.pth')
