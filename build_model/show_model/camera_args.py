import cv2
import torch

from show_build_by_vtk import vtk_show_points

dir=r"E:\Dataset\OneDrive_1_2022-11-18 (1)\calibration"

def load_camera_args(no):
 
    intrinsics_path=rf"{dir}\{no}\intrinsics.txt"
    extrinsics_path=rf"{dir}\{no}\extrinsics.txt"
    intrinsics=[]
    extrinsics=[]
    with open(intrinsics_path,"r") as f:
         for i in range(3):
                line=f.readline()
                s=line.strip()
                arr = s.split()
                l=[]
                for n in arr:
                    l.append(float(n))
                intrinsics.append(l)
    with open(extrinsics_path,"r")  as f:

            
            for i in range(3):
                line=f.readline()
                s=line.strip()
                arr = s.split()
                l=[]
                for n in arr:
                    l.append(float(n))
                extrinsics.append(l)
       
    # print(intrinsics)
    # print(extrinsics)
    return intrinsics,extrinsics


def rgbd_to_pointcloud(img_path):
    img_fullpath=rf"E:\Dataset\OneDrive_1_2022-11-18 (1)\rectified18\image01\{img_path}"
    
    depth_fullpath=rf"E:\Dataset\OneDrive_1_2022-11-18 (1)\rectified18\depth01\{img_path[:-3]}png"
  
    depth=cv2.imread(depth_fullpath,cv2.IMREAD_UNCHANGED)
    img_bgr=cv2.imread(img_fullpath,cv2.IMREAD_UNCHANGED)
    # img_rgb=cv2.cvtColor(img_bgr,code=cv2.COLOR_BGR2RGB)
    
    # img_pixel_rgb=cv2.
    # cv2.imshow("img",img_bgr)
    # cv2.waitKey()
    h,w=depth.shape
  
    
    kk,e=load_camera_args(18)
    # print(e)
    k=torch.tensor(kk,dtype=torch.float32)
    tt=torch.tensor(e,dtype=torch.float32)
    
    # print(t.shape)
    r=tt[:,:-1] #
    t=tt[:,-2:-1] #3*1
    
    # print(r.shape)
    # # t=t.view([])
    
    pointcloud=[]
    rgb_collections=[]
    
    # print(t.shape)
    # return
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
        _type_: _description_
    """
    for i in range(h):
        for j in range(w):
            # print(depth[i,j],end=",")

            d=depth[i,j]
            if d==0:
                continue
            xc=(i-k[0,2])*d/k[0,0]
            yc=(j-k[1,2])*d/k[1,1]
            dc=d
            tpos=torch.Tensor([[xc],[yc],[dc]])
            # print(t.shape)
            pos3d=r.matmul(tpos)+t
            pointcloud.append(pos3d.view(-1))

            red=img_bgr[i,j,2]
            green=img_bgr[i,j,1]
            blue=img_bgr[i,j,0]
                
            rgb_collections.append([red,green,blue])
            
    
    return pointcloud,rgb_collections


if __name__ =='__main__':
    # load_camera_args(14)
    l,rgb_colls=rgbd_to_pointcloud("0000000000.jpg")
    # print(l)

    # def save_point(points,fileName):
    #     with open(fileName,"w") as f:
    #         for i in l:
    #             f.writelines(f"{i[0]},{i[1]},{i[2]}\n")
                

    vtk_show_points(l,rgb_colls)
    # save_point(l,"test.points")

