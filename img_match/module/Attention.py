import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self) -> None:
        super(Attention).__init__()
        
        # self.