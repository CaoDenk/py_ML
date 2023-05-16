import torch
from __init__ import _load_module
import torchvision.transforms as transforms
from torch import nn
from torch import optim
# from torch.nn import n
from PIL import Image
import torchvision.transforms.functional as TF

_load_module("load_dataset")
_load_module("module")

from dataset import get_dataset
from Unet import Unet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
u=torch.load(r"E:\Dataset\Img_seg\u.pth")
# print(type(u))
img=Image.open(r"E:\Dataset\Img_seg\make_dataset\mesad-real\mesad-real\train\images\real1_frame_654.jpg")

# img_u=img.resize((572,572))    
            
            
img_tensor=transforms.ToTensor()(img)
input=img_tensor.unsqueeze(0)
input=input.to(device=device,dtype=torch.float32)


img_pred=u(input)
img_pred=img_pred.squeeze(0)
# print(img_pred)
# img_pred=img_pred*255


img_pred=torch.where(img_pred>0,torch.ones_like(img_pred),torch.zeros_like(img_pred))
img_show=TF.to_pil_image(img_pred)
img_show_rgb=img_show.convert("RGB")


h=img_show_rgb.height
w=img_show_rgb.width
print(h,w)
img_new=Image.new(mode="RGB",size=[w*2,h])
img_new.paste(img_show_rgb,(0,0))
img_new.paste(img,(w,0))
img_new.show()
