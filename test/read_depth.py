import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch
def show_img_without_depth_eq_zero(img_path,depth_path):
    img=cv2.imread(img_path)
    depth=cv2.imread(depth_path,cv2.IMREAD_UNCHANGED)
    
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGBA)
    
    h,w=depth.shape
    for i in range(h):
        for j in range(w):
            if depth[i,j]==0:
                img[i,j,3]=255
                
    cv2.imshow("rgba",img)
    cv2.waitKey()
    
def de_occ(img,mask):
    imgrgba=Image.open(img).convert("RGBA")
    imgmask=Image.open(mask)
    
       
    h,w=imgmask.size
    

    array_mask=imgmask.load()
    array=imgrgba.load()
    
    
    
    for i in range(h):
        for j in range(w):
            

            if array_mask[i,j]==0:
                 array[i,j]=(0,0,0,0)

   
    # imgrgba.save("ckcu8ty6z00003b5yzfaezbs5_mask.png")    
    imgrgba.show()
if __name__=='__main__':
    img_path=r"E:\Dataset\OneDrive_1_2022-11-18\rectified24\image01\0000000000.jpg"
    depth_path=r"E:\Dataset\OneDrive_1_2022-11-18\rectified24\depth01\0000000000.png"
    # show_img_without_depth_eq_zero(img_path,depth_path)
    de_occ(img_path,depth_path)
