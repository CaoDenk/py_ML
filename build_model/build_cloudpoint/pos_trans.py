import torch

from camera_args import load_camera_args
import cv2

dir=r"E:\Dataset\OneDrive_1_2022-11-18"
dataset_no=22
intrinsics_t,extrinsics_t=load_camera_args(dir,dataset_no=dataset_no)

depth=cv2.imread(r"E:\Dataset\OneDrive_1_2022-11-18\rectified22\image01\0000000036.jpg",cv2.IMREAD_ANYDEPTH)
# 内参矩阵
# K = torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1]])
K=intrinsics_t
# 外参矩阵
R=extrinsics_t[:,:-1] # 3*3 旋转矩阵
t=extrinsics_t[:,-1:] #3*1 平移矩阵
# R = torch.tensor([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
# t = torch.tensor([tx, ty, tz])


Rt = torch.cat((R, t.reshape(3, 1)), dim=1)

# 深度图
# depth = torch.tensor([[d00, d01, ..., d0W-1], [d10, d11, ..., d1W-1], ..., [dH-1, dH-1, ..., dH-1]])
depth=torch.from_numpy(depth).float()
H,W=depth.size()
# 计算相机坐标系下的三维坐标
y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
v = torch.stack((x.reshape(-1), y.reshape(-1), torch.ones(H * W)))
P = torch.matmul(torch.inverse(K), v) * depth.reshape(-1, 1)
P = torch.matmul(torch.inverse(Rt), P)

# 将结果reshape回原来的图像形状
X = P[0].reshape(H, W)
Y = P[1].reshape(H, W)
Z = P[2].reshape(H, W)

with open("pos2.txt") as f:
    for i in range(H):
        for j in range(W):
            f.write(F"({X[i,j],Y[i,j],Z[i,j]})\n")


