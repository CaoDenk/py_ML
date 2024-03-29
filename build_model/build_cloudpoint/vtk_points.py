import vtk
import cv2
import torch
from rgbd_to_pointcloud import save_pos
from camera_args import load_camera_args
from load_img import get_img_and_depth
from rgbd_to_pointcloud import rgbd_to_pointcloud
from show_model_by_vtk import vtk_show_points
import numpy as np
    
if __name__=='__main__':
    dir=r"E:\Dataset\OneDrive_1_2022-11-18"
    dataset_no=22
    intrinsics_t,extrinsics_t=load_camera_args(dir,dataset_no=dataset_no)
    
    # T=torch.cat((extrinsics_t,torch.Tensor([[0,0,0,1]])))
    # print(T.shape)
    # print(T)
    img_bgr,depth=get_img_and_depth(dir,dataset_no,1,"0000000012.jpg")
    
    narr=np.asarray(depth).astype(np.float32)
    t=torch.from_numpy(narr)
    max=t.max()
    posList,rgbList=rgbd_to_pointcloud(img_bgr,depth,intrinsics_t,extrinsics_t)
    # save_pos(posList,"pos.txt")
    # pointcloud,rgb_collections=rgbd_to_pointcloud(img_bgr,depth,intrinsics_t,extrinsics_t)
    # save_pos(pointcloud,"python_pos.txt")
    # cv2.imshow("img",img_bgr)
    vtk_show_points(posList,rgbList,True)
   
    # cv2.waitKey()