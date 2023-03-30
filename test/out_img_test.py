import cv2

# file=r"E:\proj\python\ml\PSMNet\Test_disparity.png"
# mat=cv2.imread(file)
# x,y,c=mat.shape

import torch
x=torch.rand((1,2,3,1))
s=x.squeeze(0)
print(s.shape)


import os

dir=r"E:\proj\python\ml\PSMNet\dataset\data_scene_flow_2015\training"
def listDir(d):
    files=os.listdir(d)
    for f in files:
        print(f)
        # if f.isdir():
        #     print(f)

listDir(dir)
