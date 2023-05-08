
import os
import sys
import torch

import tqdm
import torchvision.transforms as transforms
from PIL import Image



# def __():
#     for idex,data in enumerate(dataset):
#         # print(idex,data[0].shape)
#         # t=img_to_tensor(data[1])
#         # img.convert()
#         # print(t.shape)
#         # break
#         img_bin=data[1].convert("1")
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

def open_img():
    img=r"E:\Dataset\OneDrive_1_2022-11-18\rectified01\depth01\0000000000.png"
    img=Image.open(img)
    print(img.size)
    # pixel=img.load()
    for j in range(img.height):
        for i in range(img.width):
            print(img.getpixel((i,j)))
    
open_img()
