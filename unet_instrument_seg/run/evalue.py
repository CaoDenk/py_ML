
import torch
from __init__ import _load_module
import torchvision.transforms as transforms
from torch import nn, tensor
from torch import optim
# from torch.nn import n

_load_module("load_dataset")
_load_module("module")

from dataset import get_dataset
from Unet import Unet

def img_to_tensor(img)->torch.Tensor:
    transform = transforms.ToTensor()
    return transform(img)

def print_tensor(t:tensor):
    c,h,w=t.shape
    if c==1:
        for i in range(h):
            for j in range(w):
                if t[0,i,j] !=0:
                    print(f"{i},{j}",t[0,i,j])

def train(epoch):
    
    
    img_dir=r"E:\Dataset\图像分割\images"
    img_mask_dir=r"E:\Dataset\图像分割\masks"

    dataset= get_dataset(img_dir,img_mask_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    u=Unet()
    u.to(device=device)

    # u.to(dev)
    optimizer = optim.RMSprop(u.parameters(), lr=0.00001, weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()
    for i in range(epoch):
        u.train()
        for idx,data in enumerate(dataset):
            optimizer.zero_grad()
            img=data[0]
            # img_mask=data[1]

            img_u=img.resize((572,572))    
            
            
            img_tensor=img_to_tensor(img_u)
            input=img_tensor.unsqueeze(0)
            
            input=input.to(device=device,dtype=torch.float32)
            
            img_bin_true=data[1].convert("1")
            
            img_bin_true=img_bin_true.resize((572,572))
            img_bin_true_tensor=img_to_tensor(img_bin_true)
            
            # print_tensor(img_bin_true_tensor)
            
            # break
            
            img_bin_true_tensor=img_bin_true_tensor.squeeze()
            
            
            
            img_bin_true_tensor=img_bin_true_tensor.to(device=device,dtype=torch.float32)
            
            # print(input.size())
            # print(img_tensor.size())
            img_pred=u(input)
            # img_pred=img_pred.resize(img_bin.size())
                      
            img_bin_pred=img_pred.squeeze()
            # print("pred",img_bin_pred)
            # print("true",img_bin_true_tensor)
            loss=criterion(img_bin_pred,img_bin_true_tensor)        
            loss.backward()
            
            optimizer.step()
            print(f"idx={idx},loss={loss}")
    
    
    torch.save(u,"u.pth")

train(5)