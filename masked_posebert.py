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
from model_posebert import ImBert
# from h36m_basic import H36M
from h36m import H36M
#from torchvision.models import resnet50
from mpi_inf import MPI_INF
from ntu_loader import NTURGBD
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

def get_mpjpe(preds, gt, masks, pad_vec):
    # import pdb; pdb.set_trace()
    B, K, _ = preds.shape
    preds = preds.reshape(B, K, -1, 3)
    gt = gt.reshape(B, K, -1, 3)
    diff = ((preds - gt)**2).sum(-1).sqrt() * masks # .reshape(B, K, -1, 3)
    mpjpe = diff.sum() / masks.sum() / 25

    pmpjpe = cal_p_mpjpe(preds.reshape(-1, 16, 3).detach().cpu().numpy(),
                         gt.reshape(-1, 16, 3).detach().cpu().numpy())

    return mpjpe, pmpjpe

def step_h36m(model, data_loader, criterion, optimizer, split, scheduler=None, get_embed=False):
    if split == 'train':
        model.train()
    else:
        model.eval()

    Loss, c_l, d_l = AverageMeter(), AverageMeter(), AverageMeter()
    MPJPE = AverageMeter()

    poses_list, embeddings = [], []

    for i, (img_embeds, poses, sim_logits) in enumerate(tqdm(data_loader, ascii=True)):

        img_embeds = img_embeds.float().cuda()
        poses = poses.float().cuda()

        #if not get_embed:
        #  img_embeds = img_embeds[0]
        #  poses = poses[0]

        b, seq_len, njoints, _ = poses.shape
        poses = poses.view(b, seq_len, -1)

        #bsz, seq_len, n_joints, _ = poses.shape
        #poses = poses.reshape(bsz, seq_len, n_joints*3)

        if get_embed:
          #print("******")
          #try:
            #import pdb; pdb.set_trace()
            with torch.no_grad():
                model.eval()
                embedding = model(img_embeds, poses, True)

            #embeddings.append( embedding.detach().cpu())
            #poses_list.append( poses.detach().cpu())
            #embedding = embedding.squeeze()
            #poses = poses.squeeze()

            e = torch.cat([embedding.detach().cpu(), 99999*torch.ones((1, 1300-embedding.shape[1], 2048))], 1)  # 48
            p = torch.cat([poses.detach().cpu(), 99999*torch.ones((1, 1300-poses.shape[1], poses.shape[2]))], 1)
            #print(e.shape)
            #print(p.shape)
            embeddings.append(e)
            poses_list.append(p)
            continue
          #except RuntimeError:
          #  continue

        if split=='train':
            embedding, contrastive_loss, diversity_loss = model(img_embeds, poses)
        else:
            with torch.no_grad():
                embedding, contrastive_loss, diversity_loss = model(img_embeds, poses)

        #loss = criterion(output, labels)
        loss = contrastive_loss + 0.2 * diversity_loss
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        Loss.update(loss.detach().cpu().numpy())
        c_l.update(contrastive_loss.detach().cpu().numpy())
        d_l.update(diversity_loss.detach().cpu().numpy())

        if cfg.DEBUG > 0 and (i+1) % cfg.display == 0:
            print(f'{split} epoch: {epoch} | iter: {i} | contrastive: {c_l.avg:03f} | diversity: {d_l.avg:03f} | loss: {Loss.avg:04f} | mpjpe: {MPJPE.avg}')

    if get_embed:
        #import pdb; pdb.set_trace()
        #poses_npy = torch.cat(poses).numpy()
        embed = torch.cat(embeddings, 0)
        poses = torch.cat(poses_list, 0)
        #np.save(split+'_embed.npy', embed_npy)
        return embed, poses

    s = f'{split} epoch: {epoch} | iter: {i} | contrastive: {c_l.avg:03f} | diversity: {d_l.avg:03f} | loss: {Loss.avg:04f} | mpjpe: {MPJPE.avg}\n'
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

        img_embeds = img_embeds  #.float().cuda()
        pose_3d = pose_3d.float().cuda()

        #input, labels, label_masks = mask_poses(cfg, pose_3d.clone())
        #input = input.cuda()
        #labels = labels.cuda()
        #label_masks = label_masks.cuda()

        if get_embed:
            with torch.no_grad():
                model.eval()
                embedding = model(img_embeds, pose_3d, True)

            e = torch.cat([embedding.detach().cpu(), 99999*torch.ones((1, 13000-embedding.shape[1], 51))], 1)
            p = torch.cat([pose_3d.detach().cpu(), 99999*torch.ones((1, 13000-pose_3d.shape[1], pose_3d.shape[2]))], 1)
            #print(e.shape)
            #print(p.shape)
            embeddings.append(e)
            poses.append(p)
            continue

        if split=='train':
            embedding, contrastive_loss, diversity_loss = model(img_embeds, pose_3d)
        else:
            with torch.no_grad():
                embedding, contrastive_loss, diversity_loss = model(img_embeds, pose_3d)

        #loss = criterion(output, labels)
        loss = contrastive_loss +  0.2 * diversity_loss
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        Loss.update(loss.detach().cpu().numpy())
        c_l.update(contrastive_loss.detach().cpu().numpy())
        d_l.update(diversity_loss)  #.detach().cpu().numpy())

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

    if cfg.get_embed:
      seq_len = -1
      shuffle_bool = False
      B = 1
    else:
      shuffle_bool = True

    if cfg.dataset=='h36m':
        #dataset_train = H36M(split='train', max_len=seq_len)
        dataset_train  = NTURGBD(split='train')
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=B, shuffle=shuffle_bool)

        #dataset_val = H36M(split='val', max_len=seq_len)
        dataset_val  = NTURGBD(split='val')
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=B, shuffle=shuffle_bool)
        nJoints = 25

    else:
        dataset_train = MPI_INF("./mpi_inf_embeds/", split='train', max_length=seq_len)
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=B, shuffle=True)

        dataset_val = MPI_INF("./mpi_inf_embeds/", split='val', max_length=seq_len)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=B, shuffle=shuffle_bool)
        nJoints = 17

    bert_config = None #tf.BertConfig(hidden_size=cfg.hidden_dim, num_attention_heads=4, #3 8
                       #         num_hidden_layers=cfg.n_encoders, max_position_embeddings=2048)

    if cfg.load_model is not None:
        model = torch.load(cfg.load_model)
        model = model.cuda()
        print("Loaded model : ", cfg.load_model)
    else:
        model = ImBert(bert_config, cfg, n_joints=nJoints).cuda()

    #positions = torch.Tensor(list(range(1, seq_len+1))).unsqueeze(0).expand(B, seq_len).long().cuda()

    if cfg.get_embed:
        print('getting embeddings')
        try:
          os.mkdir('./save_embeds/{}'.format(cfg.exp_id))
        except:
          pass

        embeds_train, embeds_val, poses_train, poses_val = [], [], [], []
        for i in range(1):  #30
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

        np.save('./save_embeds/{}/poses_train_new.npy'.format(cfg.exp_id), poses_train)
        np.save('./save_embeds/{}/poses_val_new.npy'.format(cfg.exp_id), poses_val)
        np.save('./save_embeds/{}/embed_train_new.npy'.format(cfg.exp_id), embeds_train)
        np.save('./save_embeds/{}/embed_val_new.npy'.format(cfg.exp_id), embeds_val)

        exit()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.0001)
    criterion = torch.nn.MSELoss().cuda()
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
        if (epoch + 1) % cfg.save_intervals == 0:
            torch.save(model, f'{save_dir}/model_{epoch}.pth')
