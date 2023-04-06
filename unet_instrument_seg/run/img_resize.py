
import torch
from __init__ import _load_module
import torchvision.transforms as transforms
from torch import nn
from torch import optim
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor
import numpy as np

_load_module("load_dataset")
_load_module("module")

from dataset import get_dataset


def img_to_tensor(img)->torch.Tensor:
    transform = transforms.ToTensor()
    return transform(img)

from PIL import Image

file=r"E:\Dataset\图像分割\masks\ckcu8ty6z00003b5yzfaezbs5.png"
img=Image.open(file)
# img.show()


img_tensor=img_to_tensor(img)




img_reshow=TF.to_pil_image(img_tensor)

# narr=img_tensor.numpy()
# print(narr.shape)


# img_reshow=Image.fromarray(narr.astype(np.float32))
img_reshow.show()
# print(img_tensor.shape)

# img=img_tensor.numpy()
# img_show=Image.fromarray(img)
# img_show.show()

