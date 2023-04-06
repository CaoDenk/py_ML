
import os
import sys
import torch

import tqdm
import torchvision.transforms as transforms
from __init__ import _load_module
_load_module("load_dataset")
_load_module("module")
from dataset import get_dataset



img_dir=r"E:\Dataset\图像分割\images"
img_mask_dir=r"E:\Dataset\图像分割\masks"
dataset= get_dataset(img_dir,img_mask_dir)
# data_loader=dataset()
def img_to_tensor(img)->torch.Tensor:
    transform = transforms.ToTensor()
    return transform(img)


for idex,data in enumerate(dataset):
    # print(idex,data[0].shape)
    # t=img_to_tensor(data[1])
    # img.convert()
    # print(t.shape)
    # break
    img_bin=data[1].convert("1")
    # print(img.size())
    # img.show()
    # break
#     h,w,c=data[0].shape
#     if c==720:
#         count720 +=1
#     if c==768:
#         count768 +=1

# print(f"count720={count720},count768={count768}")
# count720=210,count768=254

