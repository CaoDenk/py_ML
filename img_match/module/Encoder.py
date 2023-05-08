from pyparsing import Optional
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F




class Encoder(nn.Module):
    def __init__(self) -> None:
        self.super(Encoder).__init__()
        # nn.TransformerEncoderLayer()
        
    def forward(left_img:torch.Tensor,right_img:torch.Tensor)->torch.Tensor:
        ...
