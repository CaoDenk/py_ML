from torch import nn

from ConvBlock import make_tuple,ConvBlock

def up_block(in_channels,mid_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels,mid_channels,3,1,1),
        # nn.ReLU(),
        nn.Sigmoid(),
        nn.Conv2d(mid_channels,mid_channels,3,1,1),
        # nn.ReLU(),
        nn.Sigmoid(),
        nn.Upsample(scale_factor=2.0,mode="bilinear"),  
        nn.Conv2d(mid_channels,out_channels,2,1,1),
        # nn.ReLU()
        nn.Sigmoid()
        
          
    )


class UpSample(nn.Module):
    
    def __init__(self):
        super(UpSample,self).__init__()
        
        # self.layer=nn.Conv2d(1024,512,3,1,1)
        
        # self.conv=nn.Sequential(
        #     nn.Conv2d(1024,512,3,1,1),
        #     nn.ReLU()        
        # )
     
        # self.up_conv1=nn.Upsample(scale_factor=2.0,mode="bilinear")
        
        # self.up_conv1=nn.Sequential(
        #     nn.Conv2d(1024,1024,3,1,1),
        #     nn.Conv2d(1024,512,2,1,1),
        #     nn.Upsample(scale_factor=2.0,mode="bilinear")
        # )
        self.up_conv1=nn.Sequential(
            nn.Conv2d(1024,1024,3,1,1),
            # nn.ReLU(),
            nn.LeakyReLU(0.2),
            
            nn.Upsample(scale_factor=2.0,mode="bilinear"),  
            nn.Conv2d(1024,512,2,1,1),
            # nn.ReLU(),
            
            nn.LeakyReLU(0.2),
            
        )
        self.up_conv2=up_block(1024,512,256)
        self.up_conv3=up_block(512,256,128)
        self.up_conv4=up_block(256,128,64)
        # self.up_conv4=up_block(128,64,32),
        
        self.final_layer=nn.Sequential(
            nn.Conv2d(128,64,3,1,1),
            # nn.ReLU(),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,64,3,1,1),
            # nn.ReLU(),
            nn.LeakyReLU(0.2),
            # nn.Conv2d(64,1,1,1,1),
            # nn.ReLU(),           
            nn.Conv2d(64,1,3,1,1),
            # nn.ReLU()
        )

        # self.concat
        
        # x5 1024*35*35
        #x4  512*71*71
        #x3 256 143
        #x2  128*286*286
        #x1  64*576*576
    def forward(self,x1,x2,x3,x4,x5):
        
        d4=self.up_conv1(x5) #d4 512*70*70
        xd4=make_tuple(x4,d4) #xd4 1024*71*71       
        
        # print(xd4.shape)
        d3 =self.up_conv2(xd4) #d3  256*142**142
        xd3=make_tuple(x3,d3) #xd3  512*143*143
        
        d2=self.up_conv3(xd3) #d2  128*286*286
        xd2=make_tuple(x2,d2) #xd2 256*286*286
        
        d1 =self.up_conv4(xd2) #d1 64*572*572
        xd1=make_tuple(x1,d1) #xd1 128*572*572
        return self.final_layer(xd1)
        



        
        