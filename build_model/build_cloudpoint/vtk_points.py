import vtk
import cv2
from camera_args import load_camera_args
from load_img import get_img_and_depth
from rgbd_to_pointcloud import rgbd_to_pointcloud
from show_model_by_vtk import vtk_show_points

    
if __name__=='__main__':
    dir=r"E:\Dataset\OneDrive_1_2022-11-18"
    dataset_no=22
    intrinsics_t,extrinsics_t=load_camera_args(dir,dataset_no=dataset_no)
    
    img_bgr,depth=get_img_and_depth(dir,dataset_no,1,"0000000000.jpg")
    
    
    pointcloud,rgb_collections=rgbd_to_pointcloud(img_bgr,depth,intrinsics_t,extrinsics_t)
    
    cv2.imshow("img",img_bgr)
    vtk_show_points(pointcloud,rgb_collections,True)
   
    cv2.waitKey()