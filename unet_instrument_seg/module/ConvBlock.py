from torch import nn
import torch
from torch.nn import functional as F
class ConvBlock(nn.Module):
    def __init__(self,in_channles:int,out_channles:int) -> None:
        super(ConvBlock,self).__init__()       
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=in_channles,out_channels=out_channles,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=out_channles),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=out_channles,out_channels=out_channles,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=out_channles),
            nn.ReLU(inplace=True)
        )
        """
        h==w  is required
        经过一次卷积块(c*2,h-2,w-2)
        """
    def forward(self,x):
        return self.conv(x)
  
        """在下采样的时候h1,w1必然大于h2,w2
            通道数相加
        """
def make_tuple(x1,x2):
    h1,w1=x1.shape[2],x1.shape[3]
    h2,w2=x2.shape[2],x2.shape[3]
    
    dif_h=h1-h2
    dif_w=w1-w2
    
    pad_left=dif_h//2
    pad_up=dif_w//2
    if dif_h % 2 == 0:
        pad_right=pad_left
    else:
        pad_right=0
        
    if dif_w % 2 == 0:
        pad_down=pad_up
    else:
        pad_down=0
    
    
    # torch.div()
    
    x2=F.pad(x2,(pad_left,pad_right,pad_up,pad_down))
    return torch.concat([x1,x2],dim=1)
    
    
# x1=torch.zeros((1,3,32,32))
# x2=torch.zeros((1,3,32,32))
# ret=make_tuple(x1,x2)
# print(ret.shape)
