import torch
import torch.nn as nn


class Tversky_loss(nn.Module):
    
    def __init__(self,alpha,beta) -> None:
        super(Tversky_loss).__init__()
        
        self.alpha=alpha
        self.beta=beta
        
        
    def forward(self,x):
        ...