import torchvision.transforms as transforms
from PIL import Image
import torch
import cv2
import numpy as np
import torch


def Image_to_tensor(img:Image)->torch.tensor:
    transform = transforms.ToTensor()
    return transform(img)


def mat_to_tensor(mat:cv2.Mat)->torch.Tensor:
    transform = transforms.ToTensor()
    return transform(mat)

def mat_to_numpy(mat:cv2.Mat)->np.ndarray:
    np.asarray(mat)


"""_summary_
@param img_mat 输入一个cv2.mat   
@return  返回数组，储存灰度值的像素个数的

"""
def count_pixel(img_mat:cv2.Mat)->cv2.Mat:
    
    img_mat_grey=mat_to_gray_mat(img_mat)
    img_tensor:torch.Tensor = mat_to_tensor(img_mat_grey)
    gray_value_list:list = [0 for i in range(256)]
    _,h,w=img_tensor.shape
    for i in range(h):
        for j in range(w):
            val_tensor:torch.Tensor = img_tensor[0,i,j]*255
            val_number = val_tensor.item()
            val_int = int(val_number)
            gray_value_list[val_int] += 1
    return gray_value_list
    


def draw_line(arr:list,img_size=(600,800),start_point:tuple=(10,570),line_color=(255,0,0),thickness:int=3):
    
    max=np.max(arr)
    ratio=max/(start_point[1]*0.9)
    narr=np.zeros(img_size,np.uint8)
    mat=cv2.Mat(narr)
    
    for i in range(0,256):
        current_start_point=(start_point[0]+i*thickness,start_point[1])
        end_point=(current_start_point[0],current_start_point[1]-int(arr[i]/ratio))
        mat=cv2.line(mat,current_start_point,end_point,line_color,thickness)
        if i%10==0:
            font_start_point=(current_start_point[0],current_start_point[1]+20)
            mat = cv2.putText(mat,str(i),font_start_point,cv2.FONT_HERSHEY_SIMPLEX,0.4,line_color)
    return mat


def mat_to_gray_mat(img_mat:cv2.Mat)->cv2.Mat:
    
    if len(img_mat.shape)==3:
        return cv2.cvtColor(img_mat,cv2.COLOR_BGR2GRAY)
    return img_mat





    


