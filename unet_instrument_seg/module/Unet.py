from torch import nn
import torch
from UpSample import UpSample
from DownSample import DownSample




class Unet(nn.Module):
    def __init__(self) -> None:
        super(Unet,self).__init__()
        
        # layer=nn.Sequential(
        
        self.down=DownSample()        
        self.up=UpSample()  
        # self.out_layer=nn.Softmax2d()
        # )
        
    def forward(self,x):
        c64,c128,c256,c512,c1024=self.down(x)
        x=self.up(c64,c128,c256,c512,c1024)
        # x=self.out_layer(x)
        # x=torch.where(x > 0.5, torch.ones_like(x), torch.zeros_like(x))
        return x
        

# u=Unet()
# t=torch.rand((1,3,572,572),dtype=torch.float32)
# out=u(t)
# print(out.shape)