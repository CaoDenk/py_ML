from pathlib import Path
import numpy as np
import struct

import cv2

def read_pfm(filename):
    with Path(filename).open('rb') as pfm_file:

        line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
        assert line1 in ('PF', 'Pf')
        
        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)

        buffer = pfm_file.read()
        
        
        # j=0
        # for i in buffer:
        #     if j>100:
        #         break
        #     print(i)
        #     j+=1

        
        samples = width * height * channels
        assert len(buffer) == samples * 4
        
        fmt = f'{"<>"[bigendian]}{samples}f'
        print(fmt)
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.flipud(np.reshape(decoded, shape)) * scale


import matplotlib.pyplot as plt
image = read_pfm(r'E:\Dataset\Compressed\Sampler.tar\Sampler\FlyingThings3D\disparity\0006.pfm')
plt.imshow(image)
plt.show()

# mat =cv2.fromarray(image)
# height,width =image.shape

# for i in range(10):
#     for j in range(10):
#         print(image[i,j],end=" ")
#     print()

cv2.normalize(image,image)
image =image*255
cv2.imshow("mat",image)
cv2.waitKey()
