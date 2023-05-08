import math
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
# 加载预训练的ResNet50模型
# model = models.resnet50(pretrained=True)

# # 删除最后一层全连接层
# model = torch.nn.Sequential(*list(model.children())[:-1])

# # 将模型设置为评估模式
# model.eval()

# # 加载并预处理图像
# img = Image.open('example.jpg')
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# img = transform(img)
# img = img.unsqueeze(0)

# # 将图像输入模型，得到特征表示
# with torch.no_grad():
#     features = model(img)

# # 打印特征表示的大小
# print('Features:', features.shape)

class FeatureExtracor(nn.Module):
    def __init__(self) -> None:
        super(FeatureExtracor,self).__init__()
        
        model = models.resnet50(pretrained=True)

        # 删除最后一层全连接层
        model = torch.nn.Sequential(*list(model.children())[:-1])
        self.resnet50=model
    
    def forward(self,x):
        return self.resnet50(x)
    

def pos_encoding(x):
    _,h,w=x.shape
    d=h*w
    pos_enc = torch.zeros((h, w, 2))
    for i in range(h):
        for j in range(w // 2):
            pos_enc[i, 2*j, 0] = math.sin(i / 10000**(2*j/d))
            pos_enc[i, 2*j+1, 0] = math.cos(i / 10000**(2*j/d))
            pos_enc[i, 2*j, 1] = math.sin(j / 10000**(2*j/d))
            pos_enc[i, 2*j+1, 1] = math.cos(j / 10000**(2*j/d))
            
    return torch.cat([x, pos_enc.permute(2, 0, 1)], dim=0)

 
if __name__=="__main__":
    x=torch.rand((1,3,288,720),dtype=torch.float32)
    
    fe=FeatureExtracor()
    x=fe(x)
    print(x.shape)