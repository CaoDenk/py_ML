
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from get_activation import _get_activation_fn

class TransformerEncoderlayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu") -> None:
        super().__init__()
        
        self.multihead_attn=nn.MultiheadAttention(d_model,nhead,dropout=dropout)
        
        self.linear1=nn.Linear(d_model,dim_feedforward)
        self.dropout=nn.Dropout(dropout)
        
        self.linear2=nn.Linear(dim_feedforward,d_model)
        
        