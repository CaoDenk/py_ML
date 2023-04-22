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
# def rgbd_to_pointcloud(img_bgr:cv2.Mat,depth:cv2.Mat,camera_intrinsics:torch.Tensor,camera_extrinsics:torch.Tensor)->Tuple[List,List]:
   

#     h,w=depth.shape
#     r=camera_extrinsics[:,:-1] # 3*3 旋转矩阵
#     t=camera_extrinsics[:,-1:] #3*1 平移矩阵
    
#     T=torch.cat([camera_extrinsics,torch.tensor([0,0,0,1])],dim=1)
#     print(T.shape)
#     pointcloud=[]
#     rgb_collections=[]

#     for i in range(h):
#         for j in range(w):

#             d=depth[i,j]
            
#             # if d==0:
#             #     continue
#             xc1=(i-camera_intrinsics[0,2])*d/camera_intrinsics[0,0]
#             yc1=(j-camera_intrinsics[1,2])*d/camera_intrinsics[1,1]
#             d1=math.sqrt(xc1*xc1+yc1*yc1)
            
#             kz=d/d1  #归一化的系数
            
#             xc=kz*xc1
#             yc=kz*yc1
#             zc=kz

#             tpos=torch.Tensor([[xc],[yc],[zc]])
            
            
            
            
#             pos3d=r.matmul(tpos)+t
#             pointcloud.append(pos3d.view(-1))
#             red=img_bgr[i,j,2]
#             green=img_bgr[i,j,1]
#             blue=img_bgr[i,j,0]             
#             rgb_collections.append([red,green,blue])
            
#     return pointcloud,rgb_collections


# if __name__ =='__main__':
#     # load_camera_args(14)
#     l,rgb_colls=rgbd_to_pointcloud(20,1,"0000000005.jpg")
#     # print(l)

#     # def save_point(points,fileName):
#     #     with open(fileName,"w") as f:
#     #         for i in l:
#     #             f.writelines(f"{i[0]},{i[1]},{i[2]}\n")
                

#     vtk_show_points(l,rgb_colls)
#     cv2.waitKey()
#     # save_point(l,"test.points")
def save_pos(pos_list,file_name):
    with open(file_name,"w") as f:
        for i in pos_list:
            f.writelines(f"{i[0]},{i[1]},{i[2]}\n")

def rgbd_to_pointcloud(img_bgr:cv2.Mat,depth:cv2.Mat,camera_intrinsics:torch.Tensor,camera_extrinsics:torch.Tensor)->Tuple[List,List]:
    
    h,w=depth.shape

    
    T=torch.cat([camera_extrinsics,torch.Tensor([[0,0,0,1]])],dim=0)
    T_inverse=torch.inverse(T)
    pos_collections=[]
    rgb_collections=[]

    for i in range(h):
        for j in range(w):

            d=depth[i,j]
            
            # if d==0:
            #     continue
            xc1=(i-camera_intrinsics[0,2])/camera_intrinsics[0,0]
            yc1=(j-camera_intrinsics[1,2])/camera_intrinsics[1,1]
            d1=math.sqrt(xc1*xc1+yc1*yc1+1)            
            kz=d/d1  #归一化的系数
            
            xc=kz*xc1
            yc=kz*yc1
            zc=kz
            tpos=torch.Tensor([[xc],[yc],[zc],[1]])
                       
            pos_w=torch.matmul(T_inverse,tpos)
            
            pos_collections.append([pos_w[0,0].float(),pos_w[1,0].float(),pos_w[2,0].float()])
            red=img_bgr[i,j,2]
            green=img_bgr[i,j,1]
            blue=img_bgr[i,j,0]             
            rgb_collections.append([red,green,blue])
            
    return pos_collections,rgb_collections
            
            
            
def rgbd_to_pointcloud_with_gpu(img_bgr:cv2.Mat,depth:cv2.Mat,camera_intrinsics:torch.Tensor,camera_extrinsics:torch.Tensor,device)->Tuple[List,List]:
    h,w=depth.shape
    x,y=torch.meshgrid(torch.arange(h),torch.arange(w))
    x=x.reshape(1,-1).to(device=device)
    y=y.reshape(1,-1).to(device=device)
   
    narr=np.asarray(depth).astype(np.float32)
    depth_t=torch.from_numpy(narr).to(device=device)
    
    # depth_v=depth_t.reshape(-1) #将depth转成一个维度
        
    # print(f"x shape:{x.shape},y shape: {y.shape},z shape:{depth_v.shape}")
    ones=torch.ones_like(x,device=device)

    # print(x.shape,y.shape,ones.shape)
    pos=torch.cat([x,y,ones],dim=0)

    T=torch.cat((camera_extrinsics,torch.Tensor([[0,0,0,1]])))
    
    T_invr= torch.inverse(T).to(device=device)
    K_invr= torch.inverse(camera_intrinsics).to(device=device)
    pos_w=[]
    pos=pos.reshape(pos.shape[1],3,1).float()
    pos.to(device=device)
    for i in pos:
        tmp=torch.matmul(K_invr,i)
        magnitude=torch.norm(tmp)
        tmp=torch.cat([tmp.cpu(),torch.Tensor([[1]])],dim=0)       
        tmp=torch.matmul(T_invr,tmp.to(device=device))
        tmp_pos=tmp*magnitude
        pos_w.append(tmp_pos.cpu())
    print(len(pos_w))
        

    
    
    
    
    
def compute(camera_intrinsics:torch.Tensor,camera_extrinsics,u,v,d):
    u1=(u-camera_intrinsics[0,2])/camera_intrinsics[0,0]
    v1=(u-camera_intrinsics[1,2])/camera_intrinsics[1,1]
    
    
    
    
    
    