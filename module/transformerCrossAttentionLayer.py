import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerCrossAttentionLayer(nn.Module):
    
    def __init__(self,hidden_dim,) -> None:
        super().__init__()