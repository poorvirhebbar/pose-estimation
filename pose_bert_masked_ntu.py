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
from pose_bert_ntu import ImBert
#from data_loader.h36m_multiview import H36M
#from ntu_loader import NTURGBD
from mpi_inf import MPI_INF
#from torchvision.models import resnet50


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

    return mpjpe

def step(model, data_loader, criterion, optimizer, split, scheduler=None, get_embed=False):
    if split == 'train':
        model.train()
    else:
        model.eval()

    Loss = AverageMeter()
    MPJPE = AverageMeter()

    embeddings = []
    poses = []

    for i, (imgs, seed_seq, label) in enumerate(tqdm(data_loader, ascii=True)):
        import pdb; pdb.set_trace()
        pose_3d = seed_seq  #meta['pose_un'] / 1000
        pose_3d = pose_3d.reshape( *pose_3d.shape[:2], -1)
        pose_3d = pose_3d - pose_3d[:, 14:15, :]
        # Shape is n_frames x 48... 80 frames, 16 joints, 3d coords
        #b*n*1,
        #pose_3d = pose_3d.reshape(-1, 4, 48)
        # Shape is now n_cams x n_frames x 48, 20*4*48
        #pose_3d = pose_3d.permute(1, 0, 2) #4,20,48, 4 videos, each video becomes an element, 20 is the sequence length for every video.

        #img = img.float().cuda()

        pose_3d = pose_3d.float().cuda()

        if get_embed:
            with torch.no_grad():
                model.eval()
                embedding = model(None, pose_3d, positions, True)
            poses.append(pose_3d.detach().cpu().numpy())
            embeddings.append( embedding.detach().cpu().numpy())
            continue

        # import torch 
        # import numpy as np
        '''
        # import matplotlib.pyplot as plt 
        # from numpy import random
        def get_sin_seq(l, n):
            arr= np.linspace(l, 100, num=n)
            a = torch.FloatTensor(arr) 
            b1 = torch.sin(a) 

            # plt.plot(a, b1.numpy(), color = 'red', marker = "o")  
            # plt.title("torch.sin")  
            # plt.xlabel("X")  
            # plt.ylabel("Y")  

            # plt.show() 
            return b1


        def get_b_sin_seq(b,n):
            r=random.randint(100)
            arr1=[]
            for i in range(b):
                arr1= np.append(arr1, get_sin_seq(i,n))
            return arr1 #size=b*n


        sin_input=get_b_sin_seq(48*80)
        sin_input = sin_input.reshape(-1, 4, 48)
        sin_input=sin_input.permute(1, 0, 2)
        sin_input=sin_input.float().cuda()
        print(sin_input.size)

        '''

        if split=='train':
            embedding, contrastive_loss, diversity_loss = model(None, pose_3d, positions)
        else:
            with torch.no_grad():
                embedding, contrastive_loss, diversity_loss = model(None, pose_3d, positions)

        #loss = criterion(output, labels)
        loss = contrastive_loss + 0.1 * diversity_loss

        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        Loss.update(loss.detach().cpu().numpy())
        #mpjpe = get_mpjpe(output.detach().cpu(), pose_3d.detach().cpu(),
        #                    label_masks.detach().cpu(), pad_vec)
        #MPJPE.update(mpjpe.numpy())

        if cfg.DEBUG > 0 and i % cfg.display == 0:
            print(f'{split} epoch: {epoch} | iter: {i} | contrastive: {contrastive_loss:03f} | diversity: {diversity_loss:03f} | loss: {Loss.avg} | mpjpe: {MPJPE.avg}')

    if get_embed:
        #import pdb; pdb.set_trace()
        return poses, embeddings
        '''
        poses_npy = np.array(poses)
        embed_npy = np.array(embeddings)
        return poses_npy, embed_npy
        #np.save(split+'_posevec_mpi_embed.npy', embed_npy)
        #np.save(split+'_poses_mpi.npy', poses_npy)
        #return
        '''

    s = f'{split} epoch: {epoch} | iter: {i} | contrastive: {contrastive_loss:03f} | diversity: {diversity_loss:03f} | loss: {Loss.avg} | mpjpe: {MPJPE.avg} \n'
    print(s)
    return s


if __name__ == "__main__":
    cfg = opts().parse()
    B = 2
    seq_len = 500  # B // 4

    '''
    dataset_train = H36M(cfg, split='train',
                    train_stats={}, K=4,
                    allowed_subj_list_reg=[1,5,6,7,8])
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=B, shuffle=False, drop_last=True)
    dataset_val = H36M(cfg, split='val',
                    train_stats={}, K=4,
                    allowed_subj_list_reg=[9,11])
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=B, shuffle=False, drop_last=True)
    '''

    dataset_train = NTURGBD(split='train', benchmark='view', seed_seq_len=seq_len)
    # dataset_train = MPI_INF("./data/mpi-inf/", split='train', max_length=50)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=B, shuffle=True, drop_last=True)

    dataset_val = NTURGBD(split='val', benchmark='view', seed_seq_len=seq_len)
    # dataset_val = MPI_INF("./data/mpi-inf/", split='val', max_length=50)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=B, drop_last=True)

    config = tf.BertConfig(hidden_size=51, num_attention_heads=3,  #8
                           num_hidden_layers=cfg.n_encoders)

    if cfg.load_model is not None:
        model = torch.load(cfg.load_model)
    else:
        model = ImBert(config, cfg, n_joints=17).cuda()

    positions = torch.Tensor(list(range(1, seq_len+1))).unsqueeze(0).expand(B, seq_len).long().cuda()

    if cfg.get_embed:
        print('getting embeddings')
        poses_train, embeds_train = [], []
        poses_val, embeds_val = [], []
        for i in range(3):
          p_train, e_train = step(model, dataloader_train, None, None, 'train', None, True)
          p_val, e_val = step(model, dataloader_val, None, None, 'val', None, True)
          poses_train.extend(p_train)
          embeds_train.extend(e_train)
          poses_val.extend(p_val)
          embeds_val.extend(e_val)

        poses_train = np.array(poses_train)
        poses_val = np.array(poses_val)
        embeds_train = np.array(embeds_train)
        embeds_val = np.array(embeds_val)

        print(poses_train.shape)
        print(poses_val.shape)
        print(embeds_train.shape)
        print(embeds_val.shape)

        np.save('mpi_poses_train_3.npy', poses_train)
        np.save('mpi_poses_val_3.npy', poses_val)
        np.save('mpi_embed_train_3.npy', embeds_train)
        np.save('mpi_embed_val_3.npy', embeds_val)

        exit()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.0001)
    criterion = torch.nn.MSELoss().cuda()
    milestones = [40000, 80000, 120000]
    scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=0.25, warmup_factor=1.0 / 3,
                                 warmup_iters=1500, last_epoch=-1)

    #positions = torch.Tensor(list(range(1,seq_len+1))).unsqueeze(0).expand(4, seq_len).long().cuda()
    logger = Logger('exp/'+str(cfg.exp_id))

    with open('exp/{}/opts.txt'.format(cfg.exp_id), 'a') as f:
      d = vars(cfg)
      for key in d:
        f.write(str(key) + ' : '+str(d[key]) + '\n')
      f.write('\n\n')


    for epoch in range(3000): #cfg.n_epochs):
        s = step(model, dataloader_train, criterion, optimizer, split='train', scheduler=scheduler)
        logger.write(s, 'train')
        if (epoch + 1) % cfg.val_intervals == 0:
            s = step(model, dataloader_val, criterion, optimizer, split='val')
            logger.write(s, 'val')
        if (epoch + 1) % cfg.save_intervals == 0:
            torch.save(model, f'exp/{cfg.exp_id}/model_{epoch}.pth')
