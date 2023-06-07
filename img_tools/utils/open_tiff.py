# from PIL 
import tifffile as tiff
import numpy as np
file=r"E:\2019\dataset_1\keyframe_1\data\scene_points\scene_points000000.tiff"
img=tiff.imread(file)

narr=np.array(img)


left=narr[:1024,...]
right=narr[1024:,...]

print(narr.shape)

h,w,c=left.shape
with open("tiff.txt","w") as f:
    for i in range(h):
        for j in range(w):
                f.write(f"({left[i,j,0]},{left[i,j,1]},{left[i,j,2]}),({right[i,j,0]},{right[i,j,1]},{right[i,j,2]})\n")
            


# tiff.imshow(img,vmin=0,vmax=1)


# from PIL import Image

# file=r"E:\2019\dataset_1\keyframe_1\data\scene_points\scene_points000000.tiff"
# img=Image.open(file)
# img.show()