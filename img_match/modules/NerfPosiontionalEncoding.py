import torch.nn as nn
import torch
import math

"""
#out_dim=in_dim*depth*2


"""
class NerfPosiontionalEncoding(nn.Module):
    def __init__(self,depth,sine_type='lin_sine') -> None:
        super().__init__()
        
        if sine_type=='line_sine':
            self.bases=[i+1 for i in range(depth)]
        elif sine_type=='exp_sine':
            self.bases=[2**i for i in range(depth)]
        else:
            self.bases=[i+1 for i in range(depth)]
            

    ##inputs (1,16,32,2)
    """
    
    """
    @torch.no_grad()
    def forward(self,inputs):
        out=torch.cat([torch.sin(i*math.pi*inputs) for i in self.bases] +[torch.cos(i*math.pi*inputs) for i in self.bases],axis=-1)
        assert torch.isnan(out).any()==False
        return out