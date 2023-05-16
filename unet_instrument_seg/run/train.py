
import torch
from __init__ import _load_module
import torchvision.transforms as transforms
from torch import nn, tensor
from torch import optim
from torch.utils.data import dataloader
from tqdm import tqdm
import os

_load_module("load_dataset")
_load_module("module")
_load_module("loss")

from dataset import get_dataset
from Unet import Unet
# from Tversky_loss import tversky_loss

def img_to_tensor(img)->torch.Tensor:
    transform = transforms.ToTensor()
    return transform(img)



def train(img_dir,img_mask_dir,epoch,device,net,batch_size=1):
    

    # img_dir=r"E:\Dataset\Img_seg\images"
    # img_mask_dir=r"E:\Dataset\Img_seg\masks"
    dataset= get_dataset(img_dir,img_mask_dir,1)
    dataset=dataloader.DataLoader(dataset=dataset,batch_size=batch_size)

    net.to(device=device)

    # u.to(dev)
    # optimizer = optim.RMSprop(net.parameters(), lr=0.00001, weight_decay=1e-8, momentum=0.9)
    optimizer=torch.optim.Adam(net.parameters(),lr=0.00001)
    optimizer.zero_grad()
    net.train()
    criterion = nn.BCEWithLogitsLoss()
    for i in range(epoch):
 
        for data in tqdm(dataset):
            optimizer.zero_grad()
                      
            img=data["img"]
            mask=data["mask"]

            mask=mask.to(device=device)
            img=img.to(device=device)
            
            img_pred=net(img)
     
            
            # loss=tversky_loss(img_pred,mask)        
            loss=criterion(img_pred,mask)
            
            loss.backward()
            
            optimizer.step()
            # print(f"img_pred.shape={img_pred.shape}")                     
            # print(f"mask={mask.shape}")   
            print(f"epoch={i},loss={loss}")
            # break
    
   
   

if __name__=='__main__':
    file=r"E:\Dataset\Img_seg\u.pth"
    img_dir=r"E:\Dataset\Img_seg\make_dataset\mesad-real\mesad-real\train\images"
    img_mask_dir=r"E:\Dataset\Img_seg\make_dataset\mask"
    # if os.path.exists(file):
    #     net=torch.load(file)
    # else:
    net=Unet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        train(img_dir=img_dir,img_mask_dir=img_mask_dir,epoch=100,device=device,net=net,batch_size=1)
    except Exception as e:
        print(e.args) 
        print("interrupt ,model has been saved")
        torch.save(net,file)  
        # raise e  
    torch.save(net,file)
    print("finished!")
        
       