import cv2
import torch





def load_camera_args(dirpath,dataset_no):
 
    intrinsics_path=rf"{dirpath}\calibration\{dataset_no}\intrinsics.txt"
    extrinsics_path=rf"{dirpath}\calibration\{dataset_no}\extrinsics.txt"
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
                    if len(l)>=3:
                        break
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
       

    intrinsics_t = torch.tensor(intrinsics,dtype=torch.float32)
    extrinsics_t = torch.tensor(extrinsics,dtype=torch.float32)
    return intrinsics_t,extrinsics_t



