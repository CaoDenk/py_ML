import torch
import os
from PIL import Image
# from PIL 
import tifffile as tiff
import numpy as np

def create_dir(l):
    for i in l:
        if not os.path.exists(i):
            os.mkdir(i)




dir=r"E:\2019\dataset_1\keyframe_2\data\scene_points"


par_dir=os.path.dirname(dir)

par_par_dir=os.path.dirname(par_dir)

left_dir=rf"{par_par_dir}\left_pos"
right_dir=rf"{par_par_dir}\right_pos"
create_dir((left_dir,right_dir))



files=os.listdir(dir)
for f in files:
    img_path=rf"{dir}\{f}"
    img=tiff.imread(img_path)
    img=np.array(img)
    
    left=img[:1024,...]
    right=img[1024:,...]
    left_save_path=rf"{left_dir}\{f[:-5]}"
    right_save_path=rf"{right_dir}\{f[:-5]}"
    np.save(left_save_path,left)
    np.save(right_save_path,right)

    


