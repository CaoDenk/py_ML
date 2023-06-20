import cv2
import tifffile as tiff
import json5
import numpy as np
from PIL import Image
from torchvision import transforms

    
from __init__ import _load_module
_load_module("build_cloudpoint")
from show_model_by_vtk import vtk_show_points

def get_disparity(disp):
    return tiff.imread(disp)
    
    
def get_Q(json_file):  
    with open(json_file,"r") as f:
        js=json5.load(f)
        # print(js["reprojection-matrix"])
        matrix=js["reprojection-matrix"]
        nmat=np.asmatrix(matrix)
        return nmat
            # print(nmat)

# cv2.reprojectImageTo3D()

def get3d(dataset):
    disp=dataset['disp']
    proj_json=dataset['json']
    img=dataset['img']
    
    disp=get_disparity(disp)
    Q=get_Q(proj_json)
    print(Q)
    img=transforms.ToTensor()(img)
    img=img.transpose(0,2).transpose(0,1)
    disparity=disp.astype(np.float32)
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    mask=np.where(np.isinf(points_3D[...,0]),False,True)
    return points_3D[mask],img[mask]



class Dataset():
    def __init__(self) -> None:
        pass
    
    def __len__(self):
        pass
    
    
    
    
    
if __name__=='__main__':
    disparity=get_disparity(r"E:\2019\dataset_3\keyframe_1\data\disparity\frame_data000000.tiff")
    
    proj_json=r"E:\2019\dataset_3\keyframe_1\data\reprojection_data\frame_data000000.json"
    img_path=r"E:\2019\dataset_3\keyframe_1\data\left_finalpass\frame_data000000.png"
    
    img=Image.open(img_path) 
    print(img.mode)
    img_t=transforms.ToTensor()(img)
    
    
    Q=get_Q(proj_json)
    print(Q)
    disparity=disparity.astype(np.float32)
    print(disparity.dtype)
    # disparity=cv2.Mat(disparity)
    
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    print(points_3D)
    
    
    mask=np.where(np.isinf(points_3D[...,0]),False,True)
    
    p=points_3D[mask]
    
    # np.concatenate([mask,mask,mask])


    img_t=img_t.transpose(0,2)
    img_t=img_t.transpose(0,1)
    print("img_t shape",img_t.shape)
    img_t=img_t[mask]
    print(img_t.shape)
    print(p.shape)
    # for i in img_t:
    #     print(i)
    
    vtk_show_points(p,img_t*255,False)
    
    
    # print(p.shape)
    # for i in p:
    #     print(i)
