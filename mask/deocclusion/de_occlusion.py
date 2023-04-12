from PIL import Image
import torchvision.transforms as transforms
import torch

"""
问题来了怎么用gpu处理，提高速度
"""
def de_occ(img,mask):
    imgrgba=Image.open(img).convert("RGBA")
    imgmask=Image.open(mask).convert("1")
    
    
    
    h,w=imgmask.size
    

    array_mask=imgmask.load()
    array=imgrgba.load()
    
    
    
    for i in range(h):
        for j in range(w):
            

            if array_mask[i,j]!=0:
                 array[i,j]=(0,0,0,0)

   
    imgrgba.save("ckcu8ty6z00003b5yzfaezbs5_mask.png")
    
    # imgmask.show()
    





"""
tensor的值默认范围就是(0,1),挖个坑
"""
def de_occ_t(img_t:torch.Tensor,mask_t:torch.Tensor):
    zeros_t=torch.zeros_like(img_t)
    img_t.where()
    
if __name__ =='__main__':
    img=r"E:\Dataset\Img_seg\images\ckcu8ty6z00003b5yzfaezbs5.jpg"
    mask=r"E:\Dataset\Img_seg\masks\ckcu8ty6z00003b5yzfaezbs5.png"
    de_occ(img,mask) 
