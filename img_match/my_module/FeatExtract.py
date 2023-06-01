import torch
from torch import nn
from torchvision.models import resnet50
class FeatExtract(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=resnet50(pretrained=True)
        self.model.conv1=nn.Conv2d(3,64,3,1,0)
        self.model = nn.Sequential(
        *list(self.model.children())[:-2],  # 保留模型中除最后两层之外的所有层
        nn.AdaptiveAvgPool2d((16, 32)),  # 使用自适应平均池化层调整输出大小
        nn.Conv2d(2048, 2048, 1, stride=1, padding=0, bias=False),  # 使用1x1卷积将2048个通道减少到1024个
    )
        
    def forward(self,input:torch.Tensor,mask):
        
      
        
        out =self.model(input)
        return out


if __name__=='__main__':
    input=torch.rand((1,3,288,496),dtype=torch.float32)
    fe=FeatExtract()
    feat=fe(input)
    print(feat.shape)
    

