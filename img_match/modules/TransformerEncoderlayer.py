
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from get_activation import _get_activation_fn


class TransformerEncoderLayer(nn.Module):
    def __init__(self,d_model,nhead,dim_feedforward=2048,dropout=0.1,activation="relu") -> None:
        super().__init__()
        
        self.self_attn=nn.MultiheadAttention(d_model,nhead,dropout=dropout)
        
        self.lineer1=nn.Linear(d_model,dim_feedforward)
        self.dropout=nn.Dropout(dropout)
        self.linear2=nn.Linear(dim_feedforward,d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
    
    def with_pos_emded(self,tensor,pos:Optional[Tensor]):
        return tensor if pos is None else tensor+pos
    
    def forward(self,
                src,
                src_mask:Optional[Tensor]=None,
                src_key_padding_mask:Optional[Tensor]=None,
                pos:Optional[Tensor]=None):
        q=k=self.with_pos_emded(src,pos)
        src2=self.self_attn(query=q,
                            key=k,
                            value=src,
                            key_padding_mask=src_key_padding_mask,
                            attn_mask=src_mask
                            )[0]
        
        src=src+self.dropout(src2)
        src=self.norm1(src)
        
        lin1=self.lineer1(src)
        acti=self.activation(lin1)
        drop=self.dropout(acti)
        src2=self.linear2(drop)
        src=src+self.dropout2(src2)
        
        src=self.norm2(src)
        return src
        
        