# from PIL 
import tifffile as tiff
import numpy as np
file=r"E:\2019\dataset_1\keyframe_1\left_depth_map.tiff"
img=tiff.imread(file)

narr=np.array(img)



print(narr.shape)

h,w,c=narr.shape
with open("depth.txt","w") as f:
    for i in range(h):
        for j in range(w):
                if np.isnan(narr[i,j,0]):
                    continue
                f.write(f"({narr[i,j,0]},{narr[i,j,1]},{narr[i,j,2]})\n")
            