
import random
import torch.nn as nn
import torch
from torch.nn import CosineSimilarity
import numpy as np
import transformers
from transformers.models.bert.modeling_bert import BertEncoder, BertEmbeddings, BertModel, BertLMPredictionHead
from torchvision.models import resnet50
#from transformer_enc import *
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from gumbal_quant import GumbelVectorQuantizer
from sample_neg import sample_negatives
from sample_neg_modified import sample_negs
from gen_mask import compute_mask_indices
import math

class PoseBert(nn.Module):
    def __init__(self,
                 cfg,
                 n_joints=25,
                 p_dropout=0.2):
        super(PoseBert, self).__init__()

        self.p_dropout = p_dropout
        self.n_joints = n_joints

        self.to_embed = nn.Linear(n_joints*3, cfg.hidden_size)
        self.bert = BertModel(cfg) #this is a transfer, result m embedding milegi
        # self.to_pose = BertLMPredictionHead(cfg)
        self.to_pose = nn.Sequential(
                        nn.Linear(cfg.hidden_size, cfg.hidden_size),
                        nn.ReLU(),
                        nn.Linear(cfg.hidden_size, n_joints*3)
                        ) #embedding ko ismai dala

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, positions):
        # pre-processing
        # import pdb; pdb.set_trace()
        y = self.to_embed(x)
        embedding = self.bert(position_ids=positions, inputs_embeds=y)[0] #intermediate layer ka output

        out = self.to_pose(embedding) #reconstructed pose

        return out, embedding

class ImBert(nn.Module):
    def __init__(self,
                 cfg_bert,
                 cfg,
                 n_joints=25):
        super(ImBert, self).__init__()
        self.n_joints = n_joints
        self.layer_norm = nn.LayerNorm(75)  #cfg.hidden_dim)

        # self.to_embed = nn.Linear(n_joints*3, cfg.hidden_dim)
        #self.bert = BertModel(cfg_bert) #this is a transfer, result m embedding milegi
        # self.to_pose = BertLMPredictionHead(cfg)
        # self.to_pose = nn.Sequential(
        #                 nn.Linear(cfg.hidden_dim, cfg.hidden_dim),  #njoint*3, 48,512
        #                 nn.ReLU(),
        #                 nn.Linear(cfg.hidden_dim, n_joints*3)   #2048, x, 2048
        #                 ) #embedding ko ismai dala

        #self.to_pose = nn.Sequential(
        #                nn.Linear(n_joints*3, cfg.hidden_dim),  #njoint*3, 48,512
        #                nn.ReLU(),
        #                nn.Linear(cfg.hidden_dim, 2048)  #2048, x, 2048
        #                ) #embedding ko ismai dala


        #self.mask_embed = nn.Parameter(torch.FloatTensor(cfg.hidden_dim).uniform_())
        #self.encoder = TransformerEncoder(cfg)
        encoder_layer = TransformerEncoderLayer(75, cfg.encoder_attention_heads)
        self.encoder = TransformerEncoder(encoder_layer, cfg.n_encoders)

        self.kernel_size = 9
        self.pos_conv = nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=self.kernel_size,
                                                  padding=self.kernel_size//2)
        std = math.sqrt((4 * (1.0 - 0.2)) / (self.kernel_size * cfg.hidden_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.use_sim = True

        if not self.use_sim:
          print("Not using similarity score based sampling")
        else:
          self.use_pose_sim = True
          if self.use_pose_sim:
            print("Pose based similarity")
          else:
            print("Cosine feature based similarity")

        #self.project_embed = nn.Linear( cfg.hidden_dim, cfg.output_embed_dim)            # on output of BERT
        self.relu = nn.ReLU(inplace=True)

        self.use_quantizer = True
        if not self.use_quantizer:
          print("Not using quantizer")
          self.project = nn.Linear( cfg.hidden_dim, cfg.hidden_dim)
        else:
          print("Using quantizer")
          self.quantizer = GumbelVectorQuantizer(75,   #2048                # input dim
                                               cfg.n_quantized,        # how many quantized vectors required?
                                               ( 2, 0.5, 0.999995),    # temp for training (start, stop, decay_factor)
                                               cfg.n_codebook,         # number of groups/codebooks
                                               True,                   # whether to combine vectors for all groups
                                               75, #cfg.quant_dim,          # dimensionality of quantized vector
                                               True)                   # time first/ input = [B, T, C]
          self.project_quant = nn.Linear( 75, 75) #cfg.quant_dim, cfg.output_embed_dim)            # projecting quantized vector to dimension of BERT embedding

    def contrastive_loss(self, embedding, quantized, negs):
        # embedding.shape == quantized.shape == [B, n_masked, embed_size]
        # negs.shape = [B, n_masked, embed_size, n_distractors]
        negs = negs.permute(0,1,3,2)
        temp = 0.2
        cos = torch.exp( self.sim_pos( embedding, quantized) / temp)
        # distract = cos.clone()
        embedding = embedding.unsqueeze(2).repeat(1, 1, negs.size(2), 1)
        negs_sim = torch.exp( self.sim_neg( embedding, negs) / temp ).sum(2)
        distract = cos + negs_sim
        loss = -1 * (torch.log( torch.div( cos, distract) ))
        return loss.mean()

    def compute_preds(self, x, y, negatives):
        neg_is_pos = (y == negatives).all(-1)

        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)

        logits /= 0.1  #self.logit_temp

        #if neg_is_pos.any():
        #    logits[1:][neg_is_pos] = float("-inf")

        return logits


    def diversity_loss(self, perplexity, num_vars):
        ''' Implements the diversity loss
            gumbel_softmax: a tensor of shape b*t x V
        '''
        return 1 - perplexity
        return ( num_vars - perplexity ) / num_vars

    def forward(self, x, poses, return_embed = False):
        # import pdb; pdb.set_trace()

        y = self.layer_norm(poses)

        bsz, seq_len, embed_dim = y.shape

        if not return_embed:
            #mask = compute_mask_indices( shape=y.shape[:2], padding_mask=None, mask_prob=0.3,
            #                         mask_length=1, mask_type="static")
            #mask = torch.from_numpy(mask).to(y.device)

            y_unmasked = y.clone()
            #y[mask] = self.mask_embed

        pos_embed = self.pos_conv(y.transpose(1,2))   # adding relative embedding instead of absolute positional embedding
        pos_embed = pos_embed.transpose(1,2)
        y += pos_embed

        #import pdb; pdb.set_trace()
        #embedding = self.bert(inputs_embeds=y)[0] #, position_ids=positions)[0]
        y = y.permute(1,0,2)
        embedding = self.encoder(y)
        embedding = embedding.permute(1, 0, 2)
        if return_embed:
            return embedding

        #embedding = embedding[mask]
        final_embed = embedding.view( bsz, -1, embed_dim)

        n_masked = final_embed.shape[1]

        y_ = y_unmasked  #[mask].view( bsz, n_masked, embed_dim)

        if self.use_quantizer:
          dict_quantized = self.quantizer(y_)
          quantized = dict_quantized['x']
          num_vars = dict_quantized['num_vars']
          quant_rep = self.project_quant(quantized)    # diversity loss doesn't drop without linear layer
        else:
          quant_rep = final_embed.clone()   #[mask].view(bsz, n_masked, embed_dim)  #  self.project(y_)   #y_

        if self.use_sim:
          if self.use_pose_sim:
            negs, neg_idxs = sample_negs(quant_rep, poses, 10, True)  #[mask].view(bsz,n_masked,-1), 10, True)
          else:
            negs, neg_idxs = sample_negs(quant_rep, None, 10, False)
          negs = negs.permute(3,0,1,2)

        else:
          negs, neg_idxs = sample_negatives(quant_rep, 10)  # quant_rep should contain only masked out blocks

        #quant_rep = quant_rep[mask].view(bsz, n_masked, embed_dim)    # uncomment if sampling from everywhere
        #negs = negs[mask].view(bsz, n_masked, embed_dim, -1)          # uncomment if sampling from everywhere

        n_negs = quant_rep.shape[1]
        anchor_idx = random.randint(0, n_negs-1)
        #quant_rep = quant_rep[:, anchor_idx, :].unsqueeze(1)
        #positive = y_ #final_embed  #[:, anchor_idx, :].unsqueeze(1).clone()
        # swp = positive[0, :].clone()     #[2, 1, 2048]
        # positive[0] = positive[1, :]
        # positive[1] = swp
        # positive[0], positive[1] = positive[1], positive[0]
        # final_embed = final_embed[:, anchor_idx, :].unsqueeze(1)
        #positive = final_embed.flip(0)
        #negs = negs[:, :, anchor_idx, :].unsqueeze(2)

        preds = self.compute_preds( final_embed, quant_rep, negs)    # quant_rep
        preds = torch.exp(preds)
        contrastive_loss = -1 * torch.log(torch.div(preds[0] , preds.sum(0)))
        contrastive_loss = contrastive_loss.mean()

        if self.use_quantizer:
          diversity_loss = self.diversity_loss(dict_quantized['prob_perplexity'], num_vars)
        else:
          diversity_loss = 0

        return final_embed, contrastive_loss, diversity_loss



class JointBert(nn.Module):
    def __init__(self,
                 cfg_bert,
                 cfg,
                 n_joints=17):
        super(JointBert, self).__init__()
        self.n_joints = n_joints
        self.layer_norm = nn.LayerNorm(cfg.hidden_dim)

        # self.to_embed = nn.Linear(n_joints*3, cfg.hidden_dim)
        self.bert = BertModel(cfg_bert) #this is a transfer, result m embedding milegi
        # self.to_pose = BertLMPredictionHead(cfg)
        self.to_pose = nn.Sequential(
                         nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),  #njoint*3, 48,512
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(cfg.hidden_dim // 2 , n_joints*3)   #2048, x, 2048
                         )

        self.mask_embed = nn.Parameter(torch.FloatTensor(cfg.hidden_dim).uniform_())

        self.kernel_size = 31
        self.pos_conv = nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=self.kernel_size,
                                                  padding=self.kernel_size//2)

        std = math.sqrt((4 * (1.0 - 0.2)) / (self.kernel_size * cfg.hidden_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.use_sim = True

        if not self.use_sim:
          print("Not using similarity score based sampling")
        else:
          self.use_pose_sim = True
          if self.use_pose_sim:
            print("Pose based similarity")
          else:
            print("Cosine feature based similarity")

        #self.project_embed = nn.Linear( cfg.hidden_dim, cfg.output_embed_dim)            # on output of BERT
        self.relu = nn.ReLU(inplace=True)

        self.use_quantizer = False
        if not self.use_quantizer:
          print("Not using quantizer")
          self.project = nn.Linear( cfg.output_embed_dim, cfg.output_embed_dim)
        else:
          print("Using quantizer")
          self.quantizer = GumbelVectorQuantizer(2048,                       # input dim
                                               cfg.n_quantized,              # how many quantized vectors required?
                                               ( 2, 0.5, 0.999995),          # temp for training (start, stop, decay_factor)
                                               cfg.n_codebook,               # number of groups/codebooks
                                               True,                         # whether to combine vectors for all groups
                                               cfg.quant_dim,                # dimensionality of quantized vector
                                               True)                         # time first/ input = [B, T, C]
          self.project_quant = nn.Linear( cfg.quant_dim, cfg.output_embed_dim)            # projecting quantized vector to dimension of BERT embedding

    def compute_preds(self, x, y, negatives):
        neg_is_pos = (y == negatives).all(-1)

        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)

        logits /= 0.1

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        return logits


    def diversity_loss(self, perplexity, num_vars):
        ''' Implements the diversity loss
            gumbel_softmax: a tensor of shape b*t x V
        '''
        return 1 - perplexity
        return ( num_vars - perplexity ) / num_vars

    def forward(self, x, poses, positions, sim_scores=None, return_embed = False):
        # import pdb; pdb.set_trace()

        y = self.layer_norm(x)

        bsz, seq_len, embed_dim = y.shape

        if not return_embed:
            mask = compute_mask_indices( shape=y.shape[:2], padding_mask=None, mask_prob=0.3,
                                     mask_length=1, mask_type="static")
            mask = torch.from_numpy(mask).to(y.device)

            y_unmasked = y.clone()
            y[mask] = self.mask_embed

        pos_embed = self.pos_conv(y.transpose(1,2))   # adding relative embedding instead of absolute positional embedding
        pos_embed = pos_embed.transpose(1,2)
        y += pos_embed

        #import pdb; pdb.set_trace()
        embedding = self.bert(inputs_embeds=y)[0]

        if return_embed:
            return embedding

        pred_poses = self.to_pose(embedding)
        embedding = embedding[mask]
        final_embed = embedding.view( bsz, -1, embed_dim)

        #pred_poses = self.to_pose(final_embed)

        n_masked = final_embed.shape[1]
        if sim_scores is not None:
          sim_scores = sim_scores[mask]
          sim_scores = sim_scores.view( bsz, n_masked, seq_len)

        y_ = y_unmasked[mask].view( bsz, n_masked, embed_dim)

        if self.use_quantizer:
          dict_quantized = self.quantizer(y_)
          quantized = dict_quantized['x']
          num_vars = dict_quantized['num_vars']
          quant_rep = self.project_quant(quantized)    # diversity loss doesn't drop without linear layer
        else:
          quant_rep = y_    # self.project(y_)

        if self.use_sim:
          if self.use_pose_sim:
            negs, neg_idxs = sample_negs(quant_rep, poses[mask].view(bsz,n_masked,-1), 10, True)
          else:
            negs, neg_idxs = sample_negs(quant_rep, None, 10, False)
          negs = negs.permute(3,0,1,2)

        else:
          negs, neg_idxs = sample_negatives(quant_rep, 10)  # quant_rep should contain only masked out blocks

        #quant_rep = quant_rep[mask].view(bsz, n_masked, embed_dim)    # uncomment if sampling from everywhere
        #negs = negs[mask].view(bsz, n_masked, embed_dim, -1)          # uncomment if sampling from everywhere
        n_negs = quant_rep.shape[1]
        anchor_idx = random.randint(0, n_negs-1)
        quant_rep = quant_rep[:, anchor_idx, :].unsqueeze(1)
        final_embed = final_embed[:, anchor_idx, :].unsqueeze(1)
        negs = negs[:, :, anchor_idx, :].unsqueeze(2)

        preds = self.compute_preds( final_embed, quant_rep, negs)
        preds = torch.exp(preds)
        contrastive_loss = -1 * torch.log(torch.div(preds[0] , preds.sum(0)))
        contrastive_loss = contrastive_loss.mean()
        #contrastive_loss = self.contrastive_loss(final_embed, quant_rep, negs)

        if self.use_quantizer:
          diversity_loss = self.diversity_loss(dict_quantized['prob_perplexity'], num_vars)
        else:
          diversity_loss = 0

        return final_embed, pred_poses, contrastive_loss, diversity_loss, mask




class FineTuneBert(nn.Module):
    def __init__(self, cfg, bert_model, emb_size=2048, n_joints=16):
        super(FineTuneBert, self).__init__()
        self.bert_model = bert_model
        self.emb_size = emb_size
        self.n_joints = n_joints

        # Freezing the weights of the pretrained model
        # for module in self.bert_model.parameters():
        #     module.requires_grad = False

        self.to_pose = nn.Sequential(
                                nn.Linear(emb_size, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, n_joints*3))

    def forward(self, x, poses):
        bsz = x.shape[0]
        final_embed = self.bert_model(x, poses, return_embed=True)
        pose_out = self.to_pose(final_embed.reshape(-1, self.emb_size))
        pose_out = pose_out.reshape(bsz, -1, self.n_joints*3) 
        return pose_out, final_embed


