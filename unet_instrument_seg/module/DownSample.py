from torch import nn
import torch

from ConvBlock import ConvBlock

# from .ConvBlock import ConvBlock

def print_shape(x):
    print(f"shape:{x.shape}")

class DownSample(nn.Module):
    def __init__(self) -> None:
     
        super(DownSample,self).__init__()
        # self.x=x
        self.conv1=ConvBlock(3,64)
        # self.down2=ConvBlock(32,64)
        self.first_pool=nn.MaxPool2d(2,2)
        
        self.conv2=ConvBlock(64,128)
        self.second_pool=nn.MaxPool2d(2,2)
        
        self.conv3=ConvBlock(128,256)
        self.third_pool=nn.MaxPool2d(2,2)
        
        self.conv4=ConvBlock(256,512)      
        self.fouth_pool=nn.MaxPool2d(2,2)
        
        
        
        self.conv5=ConvBlock(512,1024)
        
        
        
        
        
        # 输入n*3*572*572
    def forward(self,x): 
        
        conv_out1=self.conv1(x)  #conv_out n*64*572*572 
        pool_out1=self.first_pool(conv_out1) #pool_out  64*286*286
        # print_shape(pool_out1)
        
        conv_out2=self.conv2(pool_out1) #128*286*286
        pool_out2=self.second_pool(conv_out2) #128*143*143
        # print_shape(pool_out2)
        
        conv_out3=self.conv3(pool_out2) #256*71*71
        pool_out3=self.third_pool(conv_out3) #256*71*71
        # print_shape(pool_out3)
        
        conv_out4=self.conv4(pool_out3) #512*71*71
        pool_out4=self.fouth_pool(conv_out4)#512*35*35
        # print_shape(pool_out4)
         
        conv_out5=self.conv5(pool_out4) #1024*35*35
        # conv_out5=nn.ReLU()(conv_out5)
        
        return conv_out1,conv_out2,conv_out3,conv_out4,conv_out5
    
    



        

        


       