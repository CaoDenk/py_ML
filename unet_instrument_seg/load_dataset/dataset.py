import os
import cv2
import torch
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms

from PIL import Image



class InstrumentSegDataset(data.Dataset):
    
    def __init__(self,img_dir,img_mask_dir,imgs) -> None:
        # super().__init__()
        self.img_dir=img_dir
        self.img_mask_dir=img_mask_dir
        self.imgs=imgs
        
    def __len__(self):
        return len(self.imgs)
    
    
    def __getitem__(self,index):
        # img_mat=cv2.imread(f"{self.img_dir}/{self.imgs[index]}")
        # img_mask_mat=cv2.imread(f"{self.img_mask_dir}/{self.imgs[index]}")
        # mask_img=self.imgs[index].replace("jpg","png")
        mask_img=self.imgs[index][:-3]+"png"
        
        img_mat = Image.open(f"{self.img_dir}/{self.imgs[index]}")
        img_mask_mat = Image.open(f"{self.img_mask_dir}/{mask_img}")

        # img_mat=img_mat.resize((576,576))
        # img_mask_mat=img_mask_mat.resize((576,576))
        img_mask_mat=img_mask_mat.convert("1")
        
        return {"img":transforms.ToTensor()(img_mat),"mask":transforms.ToTensor()(img_mask_mat)}
        # return img_mat,img_mask_mat


class SegmentDataset(data.Dataset):
    
    def __init__(self,img_dir,img_mask_dir,img_masks) -> None:
        # super().__init__()
        self.img_dir=img_dir
        self.img_mask_dir=img_mask_dir
        self.img_masks=img_masks
        
    def __len__(self):
        return len(self.img_masks)
    
    
    def __getitem__(self,index):

        img_mat = Image.open(f"{self.img_dir}/{self.img_masks[index]}")
        img_mask_mat = Image.open(f"{self.img_mask_dir}/{self.img_masks[index]}")
        
        img_mask_mat=img_mask_mat.convert("1")
        
        return {"img":transforms.ToTensor()(img_mat),"mask":transforms.ToTensor()(img_mask_mat)}

def get_dataset(img_dir,img_mask_dir,dataset_no=0):
    
    dataset=None
    if dataset_no==0:
        imgs=os.listdir(img_dir)
    # assert(len(imgs)>0)    
        dataset = InstrumentSegDataset(img_dir=img_dir,img_mask_dir=img_mask_dir,imgs=imgs)   
    elif  dataset_no==1:
        img_masks=os.listdir(img_mask_dir)
        dataset=SegmentDataset(img_dir,img_mask_dir,img_masks)
    return dataset


# img_dir=r"E:\Dataset\Img_seg\images"
# img_mask_dir=r"E:\Dataset\Img_seg\masks"

# dataset= get_dataset(img_dir,img_mask_dir)

# dataloader=DataLoader(dataset=dataset,batch_size=2)
# for idex in  dataloader:
#     print(type(idex["img"]),idex["img"].shape)
#     break



# mask -> img
# 数据集 mesad-real

if __name__ =='__main__':
    img_dir=r"E:\Dataset\Matching_Reconstruction\Endovis 2017\train\instrument_1_4_training\instrument_dataset_1\left_frames"
    img_mask_dir=r"E:\Dataset\Img_seg\make_dataset\mask"
    dataset= get_dataset(img_dir,img_mask_dir,1)
    
