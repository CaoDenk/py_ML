
import torch
from __init__ import _load_module
import torchvision.transforms as transforms
from torch import nn, tensor
from torch import optim
from torch.utils.data import dataloader
import os

_load_module("load_dataset")
_load_module("module")

from dataset import get_dataset
from Unet import Unet

def img_to_tensor(img)->torch.Tensor:
    transform = transforms.ToTensor()
    return transform(img)



def train(epoch,device,net,batch_size=1):
    
    
    img_dir=r"E:\Dataset\Img_seg\images"
    img_mask_dir=r"E:\Dataset\Img_seg\masks"
    dataset= get_dataset(img_dir,img_mask_dir)
    dataset=dataloader.DataLoader(dataset=dataset,batch_size=batch_size)

    net.to(device=device)

    # u.to(dev)
    optimizer = optim.RMSprop(net.parameters(), lr=0.00001, weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()
    for i in range(epoch):
        net.train()
        for data in dataset:
            optimizer.zero_grad()
                      
            img=data["img"]
            mask=data["mask"]

            mask=mask.to(device=device)
            img=img.to(device=device,dtype=torch.float32)
            
            img_pred=net(img)
     
            
            loss=criterion(img_pred,mask)        
            loss.backward()
            
            optimizer.step()
            # print(f"img_pred.shape={img_pred.shape}")                     
            # print(f"mask={mask.shape}")   
            print(f"loss={loss}")
            # break
    
    torch.save(net,r"E:\Dataset\Img_seg\u2.pth")
   

if __name__=='__main__':
    file=r"E:\Dataset\Img_seg\u.pth"
    if os.path.exists(file):
        net=torch.load(file)
    else:
        net=Unet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        train(epoch=5,device=device,net=net,batch_size=1)
    except Exception as e:
        print(e.args)     
    finally:
        torch.save(net,file)
        print("interrupt ,model has been saved")