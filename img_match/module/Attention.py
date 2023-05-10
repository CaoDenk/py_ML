import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self) -> None:
        super(Attention,self).__init__()
        
        # self.