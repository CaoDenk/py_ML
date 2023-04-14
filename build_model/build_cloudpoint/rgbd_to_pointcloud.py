from typing import List, Tuple
import cv2
import torch


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
def rgbd_to_pointcloud(img_bgr:cv2.Mat,depth:cv2.Mat,camera_intrinsics:torch.Tensor,camera_extrinsics:torch.Tensor)->Tuple[List,List]:
   

    h,w=depth.shape
    r=camera_extrinsics[:,:-1] # 3*3 旋转矩阵
    t=camera_extrinsics[:,-2:-1] #3*1 平移矩阵
    
    pointcloud=[]
    rgb_collections=[]

    for i in range(h):
        for j in range(w):

            d=depth[i,j]
            # if d==0:
            #     continue
            xc=(i-camera_intrinsics[0,2])*d/camera_intrinsics[0,0]
            yc=(j-camera_intrinsics[1,2])*d/camera_intrinsics[1,1]
            dc=d
            tpos=torch.Tensor([[xc],[yc],[dc]])

            pos3d=r.matmul(tpos)+t
            pointcloud.append(pos3d.view(-1))

            red=img_bgr[i,j,2]
            green=img_bgr[i,j,1]
            blue=img_bgr[i,j,0]
                
            rgb_collections.append([red,green,blue])
            
    return pointcloud,rgb_collections


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
