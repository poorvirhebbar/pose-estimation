import numpy as np
from utils.lr_scheduler import WarmupMultiStepLR
from utils.average_meter import AverageMeter
from opts import opts
import sys
import os
from logger import Logger
import torch
import transformers as tf
from transformers.models.bert.modeling_bert import BertEncoder, BertEmbeddings, BertModel, BertForMaskedLM
from tqdm import tqdm
from model_posebert import JointBert
from h36m import H36M
from mpi_inf import MPI_INF
from common.error_metrics import cal_p_mpjpe


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

    indices_replaced = torch.bernoulli(torch.full(poses[..., 0].shape, 1.0)).bool() & masked_idx
    poses[indices_replaced] = mask_token

    # indices_random = torch.bernoulli(torch.full(poses[..., 0].shape, 0.5)).bool() & masked_idx & (~indices_replaced.byte()).bool()
    # random_frames = torch.randint(poses[..., 0].numel(), poses[..., 0].shape, dtype=torch.long)
    # poses[indices_random] = poses.view(-1, pose_dim)[random_frames[indices_random]]
    return poses, labels, label_masks

def get_mpjpe(preds, gt):
    # import pdb; pdb.set_trace()
    B, K, _ = preds.shape
    preds = preds.reshape(B, K, -1, 3)
    gt = gt.reshape(B, K, -1, 3)
    nJoints = gt.shape[2]
    diff = ((preds - gt)**2).sum(-1).sqrt()       #* masks # .reshape(B, K, -1, 3)
    mpjpe = diff.mean()                           #  .sum() / nJoints

    pmpjpe = cal_p_mpjpe(preds.reshape(-1, 16, 3).detach().cpu().numpy(), 
                         gt.reshape(-1, 16, 3).detach().cpu().numpy())

    return mpjpe, pmpjpe

def step_h36m(model, data_loader, criterion, optimizer, split, scheduler=None, get_embed=False):
    if split == 'train':
        model.train()
    else:
        model.eval()

    Loss, c_l, d_l, j_l = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    MPJPE = AverageMeter()

    poses_list, embeddings = [], []

    for i, (img_embeds, poses, sub_mask) in enumerate(tqdm(data_loader, ascii=True)):
        # pose info not available for H3.6M
        img_embeds = img_embeds.float().cuda()
        poses = poses.float().cuda()
        bsz, seq_len, n_joints, _ = poses.shape
        poses = poses.reshape(bsz, seq_len, n_joints*3)
        #import pdb; pdb.set_trace()

        if get_embed:
            #print("In get embed")
            with torch.no_grad():
                model.eval()
                embedding = model(img_embeds, poses, positions, None, True)
            embeddings.append( embedding.detach().cpu())
            poses_list.append( poses.detach().cpu())
            continue

        if split=='train':
            embedding, pred_poses, contrastive_loss, diversity_loss, mask = model(img_embeds, poses, positions, None)
        else:
            with torch.no_grad():
                embedding, pred_poses, contrastive_loss, diversity_loss, mask = model(img_embeds, poses, positions, None)

        #import pdb; pdb.set_trace()
        target_poses = poses   #[mask].view(bsz, -1, n_joints*3)
        assert target_poses.shape == pred_poses.shape
        #sub_mask = sub_mask.bool()

        mpjpe, pmpjpe = get_mpjpe(pred_poses, target_poses)
        # print(pmpjpe)
        pred_poses = pred_poses[mask]
        target_poses = target_poses[mask]

        loss = contrastive_loss + 0.2 * diversity_loss

        reg_loss = criterion(pred_poses, target_poses)
        if split=='train':
          reg_loss = reg_loss[sub_mask]
        reg_loss = reg_loss.mean()
        if sub_mask.sum()==0:
          reg_loss = 0
        #reg_loss = torch.nan_to_num(reg_loss)

        loss += 5 * reg_loss
	
        #mpjpe = get_mpjpe(pred_poses.detach().cpu(), target_poses.detach().cpu())

        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        Loss.update(loss.detach().cpu().numpy())
        c_l.update(contrastive_loss.detach().cpu().numpy())
        d_l.update(diversity_loss)
        j_l.update(reg_loss)
        MPJPE.update(pmpjpe)

        if cfg.DEBUG > 0 and (i+1) % cfg.display == 0:
            print(f'{split} epoch: {epoch} | iter: {i} | contrastive: {c_l.avg:03f} | diversity: {d_l.avg:03f} | Joint Loss: {j_l.avg:03f} | loss: {Loss.avg:04f} | mpjpe: {MPJPE.avg:03f}')

    if get_embed:
        #import pdb; pdb.set_trace()
        #poses_npy = torch.cat(poses).numpy()
        embed = torch.cat(embeddings, 0)
        poses = torch.cat(poses_list, 0)
        #np.save(split+'_embed.npy', embed_npy)
        return embed, poses

    s = f'{split} epoch: {epoch} | iter: {i} | contrastive: {c_l.avg:03f} | diversity: {d_l.avg:03f} | Joint loss: {j_l.avg:03f} | loss: {Loss.avg:04f} | mpjpe: {MPJPE.avg:03f} \n'
    print(s)
    return s, Loss.avg


def step(model, data_loader, criterion, optimizer, split, scheduler=None, get_embed=False):    # for MPI_INF
    if split == 'train':
        model.train()
    else:
        model.eval()

    Loss, c_l, d_l = AverageMeter(), AverageMeter(), AverageMeter()
    MPJPE = AverageMeter()

    poses, embeddings = [], []

    for i, (img_embeds, pose_3d) in enumerate(tqdm(data_loader, ascii=True)):
        # pose_3d = meta['pose_un'] / 1000
        # Shape is n_frames x 48... 80 frames, 16 joints, 3d coords

        # pose info not available for H3.6M

        pose_3d = pose_3d.reshape(*pose_3d.shape[:2], 17, 3)
        pose_3d = pose_3d - pose_3d[:, :, 14:15]
        pose_3d = pose_3d.reshape(*pose_3d.shape[:2], 51)

        # Shape is now n_cams x n_frames x 48, 20*4*48
        # pose_3d = pose_3d.permute(1, 0, 2) #4,20,48, 4 videos, each video becomes an element, 20 is the sequence length for every video.

        img_embeds = img_embeds.float().cuda()
        pose_3d = pose_3d.float()

        #input, labels, label_masks = mask_poses(cfg, pose_3d.clone())
        #input = input.cuda()
        #labels = labels.cuda()
        #label_masks = label_masks.cuda()

        if get_embed:
            with torch.no_grad():
                model.eval()
                embedding = model(img_embeds, pose_3d, positions, None, True)
            embeddings.append( embedding.detach().cpu())
            poses.append( pose_3d)
            continue

        if split=='train':
            embedding, contrastive_loss, diversity_loss = model(img_embeds, pose_3d, positions, None)
        else:
            with torch.no_grad():
                embedding, contrastive_loss, diversity_loss = model(img_embeds, pose_3d, positions, None)

        #loss = criterion(output, labels)
        loss = contrastive_loss +  0.2 * diversity_loss
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        Loss.update(loss.detach().cpu().numpy())
        c_l.update(contrastive_loss.detach().cpu().numpy())
        d_l.update(diversity_loss) #.detach().cpu().numpy())

        #mpjpe = get_mpjpe(output.detach().cpu(), pose_3d.detach().cpu(),
        #                    label_masks.detach().cpu(), pad_vec)
        #MPJPE.update(mpjpe.numpy())

        if cfg.DEBUG > 0 and (i+1) % cfg.display == 0:
            print(f'{split} epoch: {epoch} | iter: {i} | contrastive: {c_l.avg:03f} | diversity: {d_l.avg:03f} | loss: {Loss.avg:04f} | mpjpe: {MPJPE.avg}')

    if get_embed:
        #import pdb; pdb.set_trace()
        #poses_npy = torch.cat(poses).numpy()
        embed = torch.cat(embeddings, 0)
        poses = torch.cat(poses, 0)
        #np.save(split+'_embed.npy', embed_npy)
        return embed, poses

    s = f'{split} epoch: {epoch} | iter: {i} | contrastive: {c_l.avg:03f} | diversity: {d_l.avg:03f} | loss: {Loss.avg:04f} | mpjpe: {MPJPE.avg}\n'
    print(s)
    return s, Loss.avg


if __name__ == "__main__":
    cfg = opts().parse()
    B = cfg.train_batch
    seq_len = cfg.seq_len

    if cfg.dataset=='h36m':
        if not cfg.test:
          dataset_train = H36M(split='train', max_len=seq_len)
          dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=B, shuffle=True)

        dataset_val = H36M(split='val', max_len=seq_len)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=B)
        nJoints = 16

    else:
        if not cfg.test:
          dataset_train = MPI_INF("./mpi_inf/", split='train', max_length=seq_len)
          dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=B, shuffle=True)

        dataset_val = MPI_INF("./mpi_inf/", split='val', max_length=seq_len)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=B)
        nJoints = 17


    bert_config = tf.BertConfig(hidden_size=cfg.hidden_dim, num_attention_heads=4,    #8
                           num_hidden_layers=cfg.n_encoders)

    if cfg.load_model is not None:
        model = torch.load(cfg.load_model)
        print("Loaded model : ", cfg.load_model)
    else:
        model = JointBert(bert_config, cfg, n_joints=nJoints).cuda()

    positions = torch.Tensor(list(range(1, seq_len+1))).unsqueeze(0).expand(B, seq_len).long().cuda()
    criterion = torch.nn.MSELoss(reduce=False).cuda()

    if cfg.test:
      print("Running Eval ")
      epoch = 0
      s, _ = step_h36m(model, dataloader_val, criterion, None, 'val', None, False)
      print(s)
      exit()

    if cfg.get_embed:
        print('getting embeddings')
        print("Saving embeddings in srijon directory")
        try:
          os.mkdir('./save_embeds/{}'.format(cfg.exp_id))
        except:
          pass

        embeds_train, embeds_val, poses_train, poses_val = [], [], [], []
        for i in range(5):   #30
            #import pdb; pdb.set_trace()

            if cfg.dataset=='h36m':
                embeds, poses = step_h36m(model, dataloader_train, None, None, 'train', None, True)
                embeds_v, poses_v = step_h36m(model, dataloader_val, None, None, 'val', None, True)

            else:
                embeds, poses = step(model, dataloader_train, None, None, 'train', None, True)
                embeds_v, poses_v = step(model, dataloader_val, None, None, 'val', None, True)

            embeds_train.append(embeds)
            poses_train.append(poses)
            embeds_val.append(embeds_v)
            poses_val.append(poses_v)

        embeds_train = torch.cat(embeds_train, 0).numpy()
        embeds_val = torch.cat(embeds_val, 0).numpy()
        poses_train = torch.cat(poses_train, 0).numpy()
        poses_val = torch.cat(poses_val, 0).numpy()

        print(embeds_train.shape)
        print(poses_train.shape)

        np.save('./save_embeds/{}/poses_train.npy'.format(cfg.exp_id), poses_train)
        np.save('./save_embeds/{}/poses_val.npy'.format(cfg.exp_id), poses_val)
        np.save('./save_embeds/{}/embed_train.npy'.format(cfg.exp_id), embeds_train)
        np.save('./save_embeds/{}/embed_val.npy'.format(cfg.exp_id), embeds_val)

        exit()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.0001)
    criterion = torch.nn.MSELoss(reduce=False).cuda()
    milestones = [40000, 80000, 120000]
    scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=0.25, warmup_factor=1.0 / 3,
                                 warmup_iters=1500, last_epoch=-1)

    #positions = torch.Tensor(list(range(1,seq_len+1))).unsqueeze(0).expand(4, seq_len).long().cuda()
    save_dir = './exp/' + str(cfg.exp_id)
    logger = Logger(save_dir)

    with open(save_dir+'/opts.txt', 'a') as f:
      d = vars(cfg)
      for key in d:
        f.write(str(key) + ' : '+str(d[key]) + '\n')
      f.write('\n\n')

    best_loss_val = 99999
    for epoch in range(cfg.n_epochs):
        if cfg.dataset=='h36m':
            s, loss = step_h36m(model, dataloader_train, criterion, optimizer, split='train', scheduler=scheduler)
        else:
            s, loss = step(model, dataloader_train, criterion, optimizer, split='train', scheduler=scheduler)
        logger.write(s, 'train')

        if (epoch + 1) % cfg.val_intervals == 0:
            if cfg.dataset=='h36m':
                s, loss_val = step_h36m(model, dataloader_val, criterion, optimizer, split='val')
            else:
                s, loss_val = step(model, dataloader_val, criterion, optimizer, split='val')
            logger.write(s, 'val')
        if loss_val < best_loss_val:
            best_loss_val = loss_val
            torch.save(model, f'{save_dir}/best_model.pth')
            with open(save_dir+'/best_model_epoch.txt', 'a') as f:
                line = 'epoch : {} '.format(epoch) + '\n'
                f.write(line)
        if (epoch + 1) % cfg.save_intervals == 0:
            torch.save(model, f'{save_dir}/model_{epoch}.pth')
