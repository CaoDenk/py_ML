from torch import nn
import torch

from module import ConvBlock

# from .ConvBlock import ConvBlock

class Unet_test(nn.Module):
    def __init__(self) -> None:
     
        super(Unet_test,self).__init__()
        # self.x=x
        self.conv1=ConvBlock(3,64)
        # self.down2=ConvBlock(32,64)
        self.first_pool=nn.AvgPool2d(2,2)
        
        self.conv2=ConvBlock(64,128)
        self.second_pool=nn.AvgPool2d(2,2)
        
        self.conv3=ConvBlock(128,256)
        self.third_pool=nn.AvgPool2d(2,2)
        
        self.conv4=ConvBlock(256,512)      
        self.fouth_pool=nn.AvgPool2d(2,2)
        
        # self.conv5=nn.Conv2d(512,1021,3,1)
        
        
        
        
        
        
    def forward(self,x):
        
        conv_out1=self.conv1(x)
        pool_out1=self.first_pool(conv_out1)
        
        conv_out2=self.conv2(pool_out1)
        pool_out2=self.second_pool(pool_out2)
        
        conv_out3=self.conv3(pool_out2)
        pool_out3=self.third_pool(conv_out3)
        
        conv_out4=self.conv4(pool_out3)
        pool_out4=self.fouth_pool(conv_out4)
        
        return conv_out1,conv_out2
    
    


if __name__ =='__main__':
    t=torch.rand(size=(1,3,572,572),dtype=torch.float32)

    u=Unet_test()
    t=u(t)
    print(t.shape)
        
        
        


       