import cv2
from img_tools.utils import mat_to_tensor

pic_dir:str = r"E:\Dataset\dataset\rectified08-003\rectified08"
pic_left:str =rf"{pic_dir}\image01\0000000000.jpg"
pic_right:str =rf"{pic_dir}\image02\0000000000.jpg"

mat_left = cv2.imread(pic_left)
mat_right = cv2.imread(pic_right)

tensor_left = mat_to_tensor(mat_left)
tensor_right = mat_to_tensor(mat_right)

# print(tensor_left.shape)


pic =rf"{pic_dir}\0000"
print(pic)





# for i in range(100):
#      print('{:>05}'.format(i))