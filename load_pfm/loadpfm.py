# import imageio

# from PIL import Image

# file="0006.pfm"

# arr =imageio.imread(file)
# img=Image.fromarray(img)

# img.show()


import cv2

file=r"E:\Dataset\Compressed\Sampler.tar\Sampler\FlyingThings3D\disparity\0007.pfm"

mat=cv2.imreadmulti(file)

print(mat)
# cv2.imshow("pfm",mat)
# cv2.waitKey()