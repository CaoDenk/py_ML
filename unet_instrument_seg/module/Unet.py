from torch import nn
import torch
from UpSample import UpSample
from DownSample import DownSample


from torchviz import make_dot


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
        
if __name__ =='__main__':
    u=Unet()
    # summary(u,input_size=(3,224,224),batch_size=1,device="cpu")
    t=torch.rand((1,3,576,576),dtype=torch.float32)
    out=u(t)
    dot = make_dot(out, params=dict(u.named_parameters()))

    # 保存计算图为PDF文件
    dot.render('net')