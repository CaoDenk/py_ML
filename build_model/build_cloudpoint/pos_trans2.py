import math
from typing import List, Tuple
import cv2
import torch

from camera_args import load_camera_args


# dir=r"E:\Dataset\OneDrive_1_2022-11-18"
# dataset_no=22
# intrinsics_t,extrinsics_t=load_camera_args(dir,dataset_no=dataset_no)

# intrinsics_t_inv=torch.inverse(intrinsics_t)


# camera_plane_coord=torch.matmul(intrinsics_t_inv,torch.Tensor([[0],[0],[1]]))


# x=camera_plane_coord[0]
# y=camera_plane_coord[1]




# depth=cv2.imread(r"E:\Dataset\OneDrive_1_2022-11-18\rectified22\image01\0000000036.jpg",cv2.IMREAD_ANYDEPTH)


