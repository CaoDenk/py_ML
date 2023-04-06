
import torch
from __init__ import _load_module
import torchvision.transforms as transforms
from torch import nn, tensor
from torch import optim
from torch.utils.data import dataloader

_load_module("load_dataset")
_load_module("module")

from dataset import get_dataset
from Unet import Unet

def img_to_tensor(img)->torch.Tensor:
    transform = transforms.ToTensor()
    return transform(img)



def train(epoch):
    
    
    img_dir=r"E:\Dataset\Img_seg\images"
    img_mask_dir=r"E:\Dataset\Img_seg\masks"
    batch_size=1
    dataset= get_dataset(img_dir,img_mask_dir)
    dataset=dataloader.DataLoader(dataset=dataset,batch_size=batch_size)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    u=Unet()
    u.to(device=device)

    # u.to(dev)
    optimizer = optim.RMSprop(u.parameters(), lr=0.00001, weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()
    for i in range(epoch):
        u.train()
        for data in dataset:
            optimizer.zero_grad()
            
            
            img=data["img"]
            mask=data["mask"]

            mask=mask.to(device=device)
            img=img.to(device=device,dtype=torch.float32)
            
            img_pred=u(img)
     
            
            loss=criterion(img_pred,mask)        
            loss.backward()
            
            optimizer.step()
            # print(f"img_pred.shape={img_pred.shape}")                     
            # print(f"mask={mask.shape}")   
            print(f"loss={loss}")
            # break
    
    
    torch.save(u,r"E:\Dataset\图像分割\u2.pth")

train(5)