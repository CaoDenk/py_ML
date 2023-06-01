import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import os
from torchvision.transforms import transforms 
import imageio
import torch.nn.functional as F
from tqdm import tqdm


class MyDataset(data.Dataset):
    def __init__(self,left_img_dir,right_img_dir,disp_dir,right_disp_dir):
        self.left_imgs=os.listdir(left_img_dir)        
        self.left_img_dir=left_img_dir
        self.right_img_dir=right_img_dir
        self.disp_dir=disp_dir
        self.right_disp_dir=right_disp_dir
    def __len__(self):
        return len(self.left_imgs)
    
    def __getitem__(self,index):
        
        img_name=self.left_imgs[index]
        disp_name=img_name[:-3]+"png.npy"
        img_name=img_name[:-3]+"jpg"
        

        disp_full_path=rf"{self.disp_dir}\{disp_name}"
        right_disp_full_path=rf"{self.right_disp_dir}\{disp_name}"
        left_img_full_path=rf"{self.left_img_dir}\{img_name}"
        right_img_full_path=rf"{self.right_img_dir}\{img_name}"
        left=Image.open(left_img_full_path)
        right=Image.open(right_img_full_path)
        disp=np.load(disp_full_path)
        disp=torch.from_numpy(disp)
        
        right_disp=np.load(right_disp_full_path)
        right_disp=torch.from_numpy(right_disp)
               
        data={}
        data["left"]=transforms.ToTensor()(left)
        data["right"]=transforms.ToTensor()(right)
        data["disp"]=disp
        data["occ_mask"]=data["disp"]>0
        data["occ_mask_right"]=right_disp>0
        return data 