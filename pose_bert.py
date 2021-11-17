import torch.nn as nn
import torch
from torch.nn import CosineSimilarity
import numpy as np
import transformers
from transformers.models.bert.modeling_bert import BertEncoder, BertEmbeddings, BertModel, BertLMPredictionHead
from torchvision.models import resnet50
from gumbal_quant import GumbelVectorQuantizer
from sample_neg import sample_negatives
from sample_neg_modified import sample_negs
from gen_mask import compute_mask_indices
import math
#for my model only this is sufficient

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
                 n_joints=25,
                 p_dropout=0.2):
        super(ImBert, self).__init__()
        self.p_dropout = p_dropout
        self.n_joints = n_joints

        self.layer_norm = nn.LayerNorm(self.n_joints*3)

        self.bert = BertModel(cfg_bert) #this is a transfer, result m embedding milegi

        self.mask_embed = nn.Parameter(torch.FloatTensor(self.n_joints*3).uniform_())

        self.project_quant = nn.Linear( self.n_joints*3, self.n_joints*3)              # projecting quantized vector to dimension of BERT embedding
        #self.project_embed = nn.Linear( self.n_joints*3, self.n_joints*3)              # on output of BERT
        self.relu = nn.ReLU(inplace=True)
        self.pos_conv = nn.Conv1d(self.n_joints*3, self.n_joints*3, kernel_size=9,
                                  padding=4)

        std = math.sqrt((4 * (1.0 - self.p_dropout)) / (9 * self.n_joints * 3))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)

        self.sim = CosineSimilarity(dim=2)

        self.quantizer = GumbelVectorQuantizer(3*self.n_joints,                   # input dim
                                               cfg.n_quantized,        # how many quantized vectors required?
                                               ( 2, 0.5, 0.999995),    # temp for training (start, stop, decay_factor)
                                               cfg.n_codebook,         # number of groups/codebooks
                                               True,                   # whether to combine vectors for all groups
                                               3*self.n_joints,        # dimensionality of quantized vector
                                               True)                   # time first/ input = [B, T, C]


    def compute_preds(self, x, y, negatives):

        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)

        logits /= 0.1  #self.logit_temp

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        return logits


    def __contrastive_loss(self, embedding, quantized, negs):
        # embedding.shape == quantized.shape == [B, seq_len, embed_size]
        # negs.shape = [B, seq_len, embed_size, n_distractors]

        # Put them into the contrastive loss
        # simple contrastive loss

        temp = 0.1
        sim = CosineSimilarity(dim=2)
        cos = torch.exp(sim( embedding, quantized) / temp)

        distract = cos.clone()
        bsz, n_masked, fsz, n_dist = negs.shape
        embedding = embedding.repeat(1, 1, negs.size(-1)).reshape(*negs.shape)
        #.reshape(bsz, n_masked, n_dist, fsz).permute(0,1,3,2)    #.reshape(*negs.shape)
        negs_sim = torch.exp( sim( embedding, negs) / temp ).sum(-1)
        distract = distract + negs_sim

        #for i in range(negs.size(-1)):
        #    distract += torch.exp( sim( embedding[..., i], negs[..., i]) / temp)

        loss = -1 * (torch.log( torch.div(cos, distract)))
        return loss.mean()

    def contrastive_loss(self, embedding, quantized, negs):
        # embedding.shape == quantized.shape == [B, n_masked, embed_size]
        # negs.shape = [B, n_masked, embed_size, n_distractors]
        #import pdb; pdb.set_trace()

        negs = negs.permute(0, 1, 3, 2)
        temp = 0.1
        sim = CosineSimilarity(dim=-1)
        cos = torch.exp( sim( embedding, quantized) / temp)

        # distract = cos.clone()

        embedding = embedding.unsqueeze(2).repeat(1, 1, negs.size(2), 1)
        negs_sim = torch.exp( sim( embedding, negs) / temp ).sum(-1)
        distract = cos + negs_sim

        loss = -1 * (torch.log( torch.div( cos, distract)))

        return loss.mean()

    def diversity_loss(self, perplexity):
        ''' Implements the diversity loss
            gumbel_softmax: a tensor of shape b*t x V
        '''
        return 1 - perplexity

    def forward(self, x, poses, positions, return_embed = False):
        # pre-processing
        # x was image, 224,224,3
        # y = self.resnet(x)  # embedding 2048 1d vector, to ve changed to y= something (poses), 2 linear layers with dropout

        y = self.layer_norm(poses)
        # y = poses
        # import pdb; pdb.set_trace()
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
        #embedding = self.bert(position_ids=positions, inputs_embeds=y)[0] #intermediate layer ka output

        if return_embed:
            return embedding

        embedding = embedding[mask]
        final_embed = embedding.view( y.size(0), -1, y.size(2))

        #import pdb; pdb.set_trace()
        #from utils.debugger import Debugger
        bsz, n_masked, fsz = final_embed.shape
        y_masked = y_unmasked[mask].view( bsz, n_masked, fsz)
        dict_quantized = self.quantizer(y_masked)
        quantized = dict_quantized['x']
        #quant_rep = quantized
        quant_rep = self.project_quant(quantized)

        '''
        qp = quant_rep.reshape(16, -1, 17, 3)
        qpr = qp - qp[:, :, 14:15]
        db = Debugger()
        db.addPoint3D(qpr[0,0]*1000)
        '''

        negs, neg_idxs = sample_negatives(quant_rep)
        #negs, neg_idxs = sample_negs(quant_rep, poses[mask].view( bsz, n_masked, fsz))  # similarity based sampling
        #import pdb; pdb.set_trace()

        #final_embed = self.project_embed(final_embed.view(bsz, n_masked, fsz))   # projecting BERT output
        #final_embed = final_embed.view(bsz, n_masked, fsz)

        #import pdb; pdb.set_trace()
        preds = self.compute_preds( final_embed, quant_rep, negs)
        preds = torch.exp(preds)
        contrastive_loss = - torch.log(torch.div(preds[0] , preds.sum(0)))
        contrastive_loss = contrastive_loss.mean()
        #contrastive_loss = self.contrastive_loss(final_embed, quant_rep, negs)
        diversity_loss = self.diversity_loss( dict_quantized['prob_perplexity'] )

        return final_embed, contrastive_loss, diversity_loss

class PoseBertToken(nn.Module):
    def __init__(self,
                 cfg,
                 n_joints=16,
                 p_dropout=0.2):
        super(PoseBert, self).__init__()

        self.p_dropout = p_dropout
        self.n_joints = n_joints

        self.to_embed = nn.Linear(n_joints*2, cfg.hidden_size)
        self.bert = BertModel(cfg)
        self.to_pose = BertLMPredictionHead(cfg)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, positions):
        # pre-processing
        # import pdb; pdb.set_trace()
        y = self.to_embed(x)
        embedding = self.bert(position_ids=positions, inputs_embeds=y)[0]
        
        out = self.to_pose(embedding)

        return out
