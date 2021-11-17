import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50

from transformers import GPT2Config, GPT2Model

class GPT2(nn.Module):
  def __init__(self, n_decoders, hidden_dim=2048, nJoints=17):
    super(GPT2, self).__init__()
    #resnet = resnet50(pretrained=True)
    #modules = list(resnet.children())[:-1]
    #self.resnet = nn.Sequential(*modules)

    cfg = GPT2Config(n_layer=n_decoders,
                     n_embd=hidden_dim,
                     n_inner = 2*hidden_dim,
                     n_head=8
                     )

    self.gpt = GPT2Model(cfg)

    self.to_pose = nn.Linear( hidden_dim, nJoints*3)

  def forward(self, x, past_keys=None):
    
    #x = self.resnet(x)
    #assert len(x.shape) == 3   # [bsz, seq_len, hidden_dim]

    out = self.gpt(inputs_embeds=x, past_key_values=past_keys)
    past_key_values = out.past_key_values
    embeds = out.last_hidden_state[:, -1]

    pose = self.to_pose(embeds)

    return embeds, pose, past_key_values
