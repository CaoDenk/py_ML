import cv2
import numpy as np

from PIL import Image


if __name__=='__main__':
    file=r"E:\Dataset\OneDrive_1_2022-11-18\rectified05\depth01\0000000000.png"
    mat=cv2.imread(file,cv2.IMREAD_UNCHANGED)
    h,w=mat.shape
    narr=np.asarray(mat)
    min_val = np.min(narr)
    max_val = np.max(narr)
    normalized_matrix = 255*(narr - min_val) / (max_val - min_val)

    img=Image.fromarray(normalized_matrix)
    img.show()
    
    # cv2.namedWindow("depth",cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("depth",mat*3)
    # cv2.waitKey()
    
    # with open("save-depth.txt","w") as f:
    #     for i in range(h):
    #         for j in range(w):
    #             f.write(f"{mat[i,j]} ")
    #         f.write("\n")
        