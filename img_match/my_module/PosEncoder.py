from torch import nn

"""
输入是   n ,c ,h*w,2

"""
class PosEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        
    def forward(self,input):
        n,c,h,w=input.shape
        
        
        