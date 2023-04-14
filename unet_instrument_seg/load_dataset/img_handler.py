"""_summary_
 unused
"""
from typing import Tuple
from PIL import Image

def resize(img:Image,size:Tuple[int,int]):
    # Image.resize(obj, size)
    return img.resize((572,572))
    


if __name__=='__main__':    
    file=r"C:\Users\denk\Pictures\屏幕截图 2023-03-16 205150.png"  
    img=Image.open(file)
    ret = resize(img,(572,572))
    print(ret)



