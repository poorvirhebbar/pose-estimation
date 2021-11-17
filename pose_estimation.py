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
from model_posebert import ImBert, FineTuneBert
#from h36m_basic import H36M
from h36m import H36M
#from torchvision.models import resnet50
from common.error_metrics import cal_p_mpjpe
from mpi_inf import MPI_INF
from utils.debugger import Debugger



def get_mpjpe(preds, gt):
    # import pdb; pdb.set_trace()
    B, K, _ = preds.shape
    preds = preds.reshape(B, K, -1, 3)
    gt = gt.reshape(B, K, -1, 3)
    diff = ((preds - gt)**2).sum(-1).sqrt() # * masks # .reshape(B, K, -1, 3)
    mpjpe = diff.mean()

    pmpjpe = cal_p_mpjpe(preds.reshape(-1, 16, 3).detach().cpu().numpy(),
                         gt.reshape(-1, 16, 3).detach().cpu().numpy())

    return mpjpe, pmpjpe

def step_h36m(model, data_loader, criterion, optimizer, split, scheduler=None, get_embed=False):
    if split == 'train':
        model.train()
    else:
        model.eval()

    Loss, c_l, d_l = AverageMeter(), AverageMeter(), AverageMeter()
    MPJPE, PMPJPE = AverageMeter(), AverageMeter()

    poses_list, embeddings = [], []

    for i, (img_embeds, poses, sub_mask) in enumerate(tqdm(data_loader, ascii=True)):
        # pose info not available for H3.6M
        img_embeds = img_embeds[0].float().cuda()
        #import pdb; pdb.set_trace()
        #debugger = Debugger()
        #debugger.addPoint3D(poses[0, 0].numpy()*1000)
        #debugger.show3D()

        poses = poses[0].float().cuda()
        bsz, seq_len, n_joints, _ = poses.shape
        poses = poses.reshape(bsz, seq_len, n_joints*3)
        #import pdb; pdb.set_trace()
        # sim_logits = sim_logits.float().cuda()
        if split=='train':
          pred_pose, embed = model(img_embeds, poses)
        else:
          with torch.no_grad():
            pred_pose, embed = model(img_embeds, poses)

        loss = criterion( pred_pose, poses)

        if get_embed:
            model.eval()
            pred_poses, embeds = model(img_embeds, poses)
            pred_poses = pred_poses.reshape(bsz, seq_len, n_joints, 3)
            pred_poses = pred_poses - pred_poses[:, :, 6:7]
            pred_poses = pred_poses.reshape(bsz, seq_len, -1)
            loss = criterion(pred_poses, poses)
            # embeddings.append( embedding.detach().cpu())
            # poses_list.append( poses.detach().cpu())


        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        Loss.update(loss.detach().cpu().numpy())
        mpjpe, pmpjpe = get_mpjpe(pred_pose, poses)
        MPJPE.update(mpjpe)
        PMPJPE.update(pmpjpe)

        if cfg.DEBUG > 0 and (i+1) % cfg.display == 0:
            print(f'{split} epoch: {epoch} | iter: {i} | loss: {Loss.avg:04f} | mpjpe: {MPJPE.avg:04f} | pmpjpe: {PMPJPE.avg:04f}')

    s = f'{split} epoch: {epoch} | iter: {i} | loss: {Loss.avg:04f} | mpjpe: {MPJPE.avg:04f} | pmpjpe: {PMPJPE.avg:04f}'
    print(s)

    return s, Loss.avg



if __name__ == "__main__":
    cfg = opts().parse()
    B = 1 #cfg.train_batch
    seq_len = cfg.seq_len

    if cfg.dataset=='h36m':
        dataset_train = H36M(split='train', max_len=seq_len)
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=B, shuffle=True)

        dataset_val = H36M(split='val', max_len=seq_len)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=B)

    else:
        dataset_train = MPI_INF("/home/datasets/posebert_exp/mpi_inf/", split='train', max_length=seq_len)
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=B, shuffle=True)

        dataset_val = MPI_INF("/home/datasets/posebert_exp/mpi_inf/", split='val', max_length=seq_len)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=B)


    bert_config = tf.BertConfig(hidden_size=cfg.hidden_dim, num_attention_heads=4, #3
                           num_hidden_layers=cfg.n_encoders, hidden_dropout_prob=0.1, 
                           attention_probs_dropout_prob=0.1)

    if cfg.load_model is not None:
        bert_model = torch.load(cfg.load_model)
        print("Loaded model : ", cfg.load_model)
        model = FineTuneBert(cfg, bert_model, n_joints=16).cuda()
    else:
        bert_model = ImBert(bert_config, cfg, n_joints=16).cuda()
        model = FineTuneBert(cfg, bert_model, n_joints=16).cuda()
        print("Creating a fresh bert model")

    positions = torch.Tensor(list(range(1, seq_len+1))).unsqueeze(0).expand(B, seq_len).long().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.0001)
    criterion = torch.nn.MSELoss().cuda()
    milestones = [8000, 20000, 50000]
    scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=0.25, warmup_factor=1.0 / 3,
                                 warmup_iters=1500, last_epoch=-1)

    #positions = torch.Tensor(list(range(1,seq_len+1))).unsqueeze(0).expand(4, seq_len).long().cuda()
    save_dir = 'exp/' + str(cfg.exp_id)
    logger = Logger(save_dir)

    with open(save_dir+'/opts.txt', 'a') as f:
      d = vars(cfg)
      for key in d:
        f.write(str(key) + ' : '+str(d[key]) + '\n')
      f.write('\n\n')

    best_loss_val = 99999
    for epoch in range(cfg.n_epochs):
        if cfg.dataset=='h36m':
            s, loss = step_h36m(model, dataloader_train, criterion, optimizer, split='train', scheduler=scheduler, get_embed=cfg.get_embed)
        else:
            s, loss = step(model, dataloader_train, criterion, optimizer, split='train', scheduler=scheduler)
        logger.write(s, 'train')

        if (epoch + 1) % cfg.val_intervals == 0:
            if cfg.dataset=='h36m':
                s, loss_val = step_h36m(model, dataloader_val, criterion, optimizer, 
                                            split='val', get_embed=cfg.get_embed)
            else:
                s, loss_val = step(model, dataloader_val, criterion, optimizer, split='val')
            logger.write(s, 'val')
            if loss_val < best_loss_val and epoch > 50:
                best_loss_val = loss_val
                torch.save(model, f'{save_dir}/best_model.pth')
        if (epoch + 1) % cfg.save_intervals == 0:
            torch.save(model, f'{save_dir}/model_{epoch}.pth')
