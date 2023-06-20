import numpy as np
import json5 
import cv2
def get_K(json_file):  
    with open(json_file,"r") as f:
        js=json5.load(f)
        # print(js["reprojection-matrix"])
        matrix=js["camera-calibration"]["KL"]
        camera_pos=js["camera-pose"]
        
        
        nmat=np.asmatrix(matrix)
        ncamera_pos=np.asmatrix(camera_pos)
        return nmat,ncamera_pos
    
def combine_Rt(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T
    
narr=np.load("E:/corr.npy")
print(narr.shape)


frame0=narr[:,:2]
frame1=narr[:,2:]

file=r"E:\2019\dataset_3\keyframe_1\data\frame_data\frame_data000000.json"
K,pos=get_K(file)
# print(K)

frame0=frame0.reshape((-1,1,2))
frame1=frame1.reshape((-1,1,2))



# # 计算本质矩阵
E, mask = cv2.findEssentialMat(frame0, frame1, K, method=cv2.LMEDS, prob=0.999, threshold=1.0)

# # 从本质矩阵中恢复相机位姿
_, R, t, mask = cv2.recoverPose(E, frame0, frame1, K)

rt=combine_Rt(R,t)
print(rt)

file1=r"E:\2019\dataset_3\keyframe_1\data\frame_data\frame_data000001.json"
K,pos=get_K(file1)
print(pos)

# print(rt-pos)
# print("Rotation Matrix: \n", R)
# print("Translation Vector: \n", t)
