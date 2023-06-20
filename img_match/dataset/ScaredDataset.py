import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import os
from torchvision.transforms import transforms 
import imageio
import torch.nn.functional as F
import tifffile as tiff

from tqdm import tqdm

"""
不太清楚mask 是scene_data 值为0么？

"""


class ScaredDataset(data.Dataset):
    def __init__(self,left_img_dir,right_img_dir,disp_dir,occ_mask_dir=None):
        self.left_imgs=os.listdir(left_img_dir)        
        self.left_img_dir=left_img_dir
        self.right_img_dir=right_img_dir
        self.disp_dir=disp_dir
        self.occ_mask_dir=occ_mask_dir
        
        
        
        
    def __len__(self):
        return len(self.left_imgs)
    
    def __getitem__(self,index):
        
        img_name=self.left_imgs[index]
        
        disp_name=img_name[:-3]+"tiff"

        disp_full_path=rf"{self.disp_dir}\{disp_name}"
     
        left_img_full_path=rf"{self.left_img_dir}\{img_name}"
        right_img_full_path=rf"{self.right_img_dir}\{img_name}"
        left=Image.open(left_img_full_path)
        right=Image.open(right_img_full_path)
      
        disp=tiff.imread(disp_full_path)
                
        mask=torch.ones(size=disp.shape)
        mask=mask.bool()      
        data={}
        data["left"]=transforms.ToTensor()(left)
        data["right"]=transforms.ToTensor()(right)
        data["disp"]=disp
        data["occ_mask"]=mask
        data["occ_mask_right"]=mask
        return data 
    
    def get_mask(self,tiff_path):
        # tiff.
        arr=tiff.imread(tiff_path)
        mask=np.any(arr!=0,axis=2)
        left_mask=mask[:1024,...]
        right_mask=mask[1024:,...]
        
        return left_mask,right_mask
        