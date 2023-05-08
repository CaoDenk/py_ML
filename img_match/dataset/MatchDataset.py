import os
import cv2
import torch
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from torch import nn
from PIL import Image


class MatchDataset(data.Dataset):
    def __init__(self,left_dir,right_dir,depth_dir) -> None:
        # super(MatchDataset).__init__()       
        self.left_imgs=os.listdir(left_dir)
        self.right_imgs=os.listdir(right_dir)
        self.depth_imgs=os.listdir(depth_dir)
        
    def __len__(self):
        return len(self.left_imgs)
    
    
    
    def __getitem__(self, index):
        
        left_img=Image.open(self.left_imgs[index])
        right_img=Image.open(self.right_imgs[index])
        depth_img=Image.open(self.depth_imgs[index])
        
        left=transforms.ToTensor()(left_img)
        right=transforms.ToTensor()(right_img)
        depth=transforms.ToTensor()(depth_img)
        return {"left":left,"right":right,"depth":depth}
            
        
        
        
    
        