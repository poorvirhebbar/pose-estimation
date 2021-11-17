import torch.nn as nn
import torch
from torch.nn import CosineSimilarity
import numpy as np
import transformers
from transformers.models.bert.modeling_bert import BertEncoder, BertEmbeddings, BertModel, BertLMPredictionHead
from torchvision.models import resnet50
from gumbal_quant import GumbelVectorQuantizer
from sample_neg import sample_negatives
from gen_mask import compute_mask_indices
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
        #resnet = resnet50(pretrained=True)
        #modules = list(resnet.children())[:-1]
        #self.resnet = nn.Sequential(*modules)
        self.layer_norm = nn.LayerNorm(self.n_joints*3) #48)

        #self.to_embed = nn.Linear(n_joints*3, cfg.hidden_dim)
        self.bert = BertModel(cfg_bert) #this is a transfer, result m embedding milegi
        # self.to_pose = BertLMPredictionHead(cfg)
        # self.to_pose = nn.Sequential(
        #                 nn.Linear(cfg.hidden_dim, cfg.hidden_dim),#njoint*3, 48,512
        #                 nn.ReLU(),
        #                 nn.Linear(cfg.hidden_dim, n_joints*3)#2048, x, 2048
        #                 ) #embedding ko ismai dala
        '''
        self.to_pose = nn.Sequential(
                        nn.Linear(n_joints*3, cfg.hidden_dim), #njoint*3, 48,512
                        nn.ReLU(),
                        nn.Linear(cfg.hidden_dim, 2048)        #2048, x, 2048
                        ) #embedding ko ismai dala
        '''

        self.mask_embed = nn.Parameter(torch.FloatTensor(self.n_joints*3).uniform_())

        self.project_quant = nn.Linear( self.n_joints*3, self.n_joints*3)              # projecting quantized vector to dimension of BERT embedding
        self.project_embed = nn.Linear( cfg.hidden_dim, cfg.output_embed_dim)   # on output of BERT
        self.relu = nn.ReLU(inplace=True)

        self.quantizer = GumbelVectorQuantizer(3*self.n_joints,                   # input dim
                                               cfg.n_quantized,        # how many quantized vectors required?
                                               ( 2, 0.5, 0.999995),    # temp for training (start, stop, decay_factor)
                                               cfg.n_codebook,         # number of groups/codebooks
                                               True,                   # whether to combine vectors for all groups
                                               3*self.n_joints,          # dimensionality of quantized vector
                                               True)                   # time first/ input = [B, T, C]

    def contrastive_loss(self, embedding, quantized, negs):
        # embedding.shape == quantized.shape == [B, seq_len, embed_size]
        # negs.shape = [B, seq_len, embed_size, n_distractors]

        # Put them into the contrastive loss
        #import pdb; pdb.set_trace()
        # simple contrastive loss
        #import pdb; pdb.set_trace()

        temp = 0.1
        sim = CosineSimilarity(dim=2)
        cos = torch.exp(sim( embedding, quantized) / temp)

        distract = cos.clone()

        embedding = embedding.repeat(1, 1, negs.size(-1)).reshape(*negs.shape)
        negs_sim = torch.exp( sim( embedding, negs) / temp ).sum(-1)
        distract = distract + negs_sim

        #for i in range(negs.size(-1)):
        #    distract += torch.exp( sim( embedding[..., i], negs[..., i]) / temp)

        loss = -1 * (torch.log( torch.div(cos, distract)))
        return loss.sum(1).mean()
        #return loss.mean()

    def diversity_loss(self, perplexity):
        ''' Implements the diversity loss
            gumbel_softmax: a tensor of shape b*t x V
        '''
        return 1 - perplexity

    def forward(self, x, poses, positions, return_embed = False):
        # pre-processing
        # x was image, 224,224,3
        # y = self.resnet(x)  # embedding 2048 1d vector, to ve changed to y= something (poses), 2 linear layers with dropout
        #y1 = self.resnet(x)

        #y = self.to_pose(poses)
        #print("y", y.shape)
        #y = poses.reshape(-1, 4, 48)
        #y = y.permute(1, 0, 2)  #4,20, 2048
        y = self.layer_norm(poses)
        #import pdb; pdb.set_trace()
        if not return_embed:
            mask = compute_mask_indices( shape=y.shape[:2], padding_mask=None, mask_prob=0.3,
                                     mask_length=5, mask_type="static")
            mask = torch.from_numpy(mask).to(y.device)
            y_unmasked = y.clone()
            y[mask] = self.mask_embed

        embedding = self.bert(position_ids=positions, inputs_embeds=y)[0] #intermediate layer ka output

        if return_embed:
            return embedding

        embedding = embedding[mask]
        final_embed = embedding.view( y.size(0), -1, y.size(2))

        #bool = torch.ones(( y.size(0), 5)).bool()
        #bool[mask] = False
        #import pdb; pdb.set_trace()
        #from utils.debugger import Debugger
        y_masked = y[mask].view( y.size(0), -1, y.size(2))
        dict_quantized = self.quantizer(y_masked)
        quantized = dict_quantized['x']
        quant_rep = self.project_quant(quantized)
        '''
        qp = quant_rep.reshape(16, -1, 17, 3)
        qpr = qp - qp[:, :, 14:15]
        db = Debugger()
        db.addPoint3D(qpr[0,0]*1000)
        '''

        negs, neg_idxs = sample_negatives(quant_rep)  # sampling negatives from everywhere else

        contrastive_loss = self.contrastive_loss( final_embed, quant_rep, negs)
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
