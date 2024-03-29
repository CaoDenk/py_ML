import math
from typing import List, Tuple
import cv2
import torch
import numpy as np
"""
转化成相机坐标系
P = K^-1 * [i, j, 1]^T * d
"""


"""
    方向
    _________>  y,w
    |
    |
    ﹀
    x,h
    
    vtk空间坐标系
    ︿x
    |
    |
    |___________> y
    /
    z /
    垂直平面向外是z轴 
    
    相机平面
    --------> x
    |
    |
    ﹀
    y
              
Returns:
    _type_: 点云（x,y,z）集合， 与之点对应rgb集合
"""

def save_pos(pos_list,file_name):
    with open(file_name,"w") as f:
        for i in pos_list:
            f.writelines(f"{i[0]},{i[1]},{i[2]},{math.sqrt(i[0]*i[0]+i[1]*i[1]+i[2]*i[2])},true depth:{i[3]}\n")
"""
如果深度图是欧式距离
"""
def rgbd_to_pointcloud_of_Euclidean_distance(img_bgr:cv2.Mat,depth:cv2.Mat,camera_intrinsics:torch.Tensor,camera_extrinsics:torch.Tensor)->Tuple[List,List]:
    
    h,w=depth.shape

    R=camera_extrinsics[:,:3]
    t=camera_extrinsics[:,3:]
    t=t.reshape((-1,1))
    L=torch.matmul(R.T,camera_intrinsics.inverse())
    W=torch.matmul(R.T,t).reshape((-1))
    
    pos_collections=[]
    rgb_collections=[]

    for i in range(h):
        for j in range(w):
            
            d=depth[i,j]
            if d==0:
                continue
            pos=torch.Tensor([[i],[j],[1]])
            P=torch.matmul(L,pos).reshape(-1)
          
            W2=W.dot(W)
            P2=P.dot(P)
            a=(d*d-W2)/P2
            b=P.dot(W)/P2
            z_c=torch.sqrt(a+b*b)+b
            x_w,y_w,z_w=z_c*P-W
          
            pos_collections.append([x_w.float(),y_w.float(),z_w.float(),d])
            red=img_bgr[i,j,2]
            green=img_bgr[i,j,1]
            blue=img_bgr[i,j,0]             
            rgb_collections.append([red,green,blue])
            
    return pos_collections,rgb_collections
"""
如果深度图是z轴坐标
"""          
def rgbd_to_pointcloud(img_bgr:cv2.Mat,depth:cv2.Mat,camera_intrinsics:torch.Tensor,camera_extrinsics:torch.Tensor)->Tuple[List,List]:       
    h,w=depth.shape

    R=camera_extrinsics[:,:3]
    t=camera_extrinsics[:,3:]
    t=t.reshape((-1,1))
    # T=torch.cat([camera_extrinsics,torch.Tensor([[0,0,0,1]])],dim=0)
    L=torch.matmul(R.T,camera_intrinsics.inverse())
    W=torch.matmul(R.T,t).reshape((-1))
    
    pos_collections=[]
    rgb_collections=[]

    for i in range(h):
        for j in range(w):
            
            d=depth[i,j]
            if d==0:
                continue
            pos=torch.Tensor([[i],[j],[1]])
            P=torch.matmul(L,pos).reshape(-1)
          
            z_c=(d+W[2])/P[2]
            x_w,y_w,z_w=z_c*P-W
          
            pos_collections.append([x_w.float(),y_w.float(),z_w.float(),d])
            red=img_bgr[i,j,2]
            green=img_bgr[i,j,1]
            blue=img_bgr[i,j,0]             
            rgb_collections.append([red,green,blue])
            
    return pos_collections,rgb_collections

# def rgbd_to_pointcloud_with_gpu(img_bgr:cv2.Mat,depth:cv2.Mat,camera_intrinsics:torch.Tensor,camera_extrinsics:torch.Tensor,device)->Tuple[List,List]:
#     h,w=depth.shape
#     x,y=torch.meshgrid(torch.arange(h),torch.arange(w))
#     x=x.reshape(1,-1).to(device=device)
#     y=y.reshape(1,-1).to(device=device)
   
#     narr=np.asarray(depth).astype(np.float32)
#     depth_t=torch.from_numpy(narr).to(device=device)
    
#     # depth_v=depth_t.reshape(-1) #将depth转成一个维度
        
#     # print(f"x shape:{x.shape},y shape: {y.shape},z shape:{depth_v.shape}")
#     ones=torch.ones_like(x,device=device)

#     # print(x.shape,y.shape,ones.shape)
#     pos=torch.cat([x,y,ones],dim=0)

#     T=torch.cat((camera_extrinsics,torch.Tensor([[0,0,0,1]])))
    
#     T_invr= torch.inverse(T).to(device=device)
#     K_invr= torch.inverse(camera_intrinsics).to(device=device)
#     pos_w=[]
#     pos=pos.reshape(pos.shape[1],3,1).float()
#     pos.to(device=device)
#     for i in pos:
#         tmp=torch.matmul(K_invr,i)
#         magnitude=torch.norm(tmp)
#         tmp=torch.cat([tmp.cpu(),torch.Tensor([[1]])],dim=0)       
#         tmp=torch.matmul(T_invr,tmp.to(device=device))
#         tmp_pos=tmp*magnitude
#         pos_w.append(tmp_pos.cpu())
#     print(len(pos_w))
        

    
# def save_pose(l:List):
#      with open("pos.txt","w") as f:
#          for i in l:
#              f.write(f"{i[0]},{i[1]},{i[2]}\n")   
    
    
    
# def compute(camera_intrinsics:torch.Tensor,camera_extrinsics,u,v,d):
#     u1=(u-camera_intrinsics[0,2])/camera_intrinsics[0,0]
#     v1=(u-camera_intrinsics[1,2])/camera_intrinsics[1,1]
    
    
    
    
    
    